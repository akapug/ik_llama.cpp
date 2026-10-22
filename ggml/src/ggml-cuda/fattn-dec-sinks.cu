// Split-K (flash-decoding) attention for head size 512 on Pascal, for decode shapes: one K/V head shared by many Q
// heads, K == V (DeepSeek V4: 64 Q heads, F16 cache, F32 sinks), one Q row per block.
//
// The tile_new kernel packs 8 Q heads per block, so a 64-head decode op reads K and V 8 times each and runs 56 blocks;
// on a GTX 1080 Ti it costs 65-350 us per layer. Here a block owns 32 Q heads (8 warps x 4 heads) and a slice of the KV
// rows. Each 8-row K tile is read from global memory once, converted to F32 in shared memory once, and serves both
// K @ Q and V @ softmax(KQ) (K == V). Per lane: 16 of the 512 dims (4 float4 at 128*j + 4*lane) of Q and of the output
// for its warp's 4 heads stay in registers; the 32 partial dot products of a tile (4 heads x 8 rows) are reduced with
// a transposed butterfly (31 shuffles, lane l ends with head l/8, row l%8). The online softmax state lives per lane.
// Each block writes an unnormalized partial (O, max, sum); the combine kernel merges the splits and adds the sink
// logit (one extra softmax term with no value row) once per head.
//
// With src[5] (the DSV4 selected-row index list from ggml_mask_to_index: the rows whose mask is 0, padded with -1) the
// kernel walks the list instead of every row.
//
// Settings (environment): LLAMA_FA_DEC (default 0 = off: tile_new / the dsa path / the CPU as before), LLAMA_FA_DEC_MAX_Q
// (default 5: the most Q rows of a dense op it takes), LLAMA_FA_DEC_MAX_Q_SPARSE (default 8: the same with a src[5] row
// list), LLAMA_FA_DEC_SPLITS (default 0 = from the SM count).

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-dec-sinks.cuh"

#include <cstdlib>

namespace {

constexpr int dec_D   = 512;   // head size (K and V)
constexpr int dec_HPW = 4;     // Q heads per warp
constexpr int dec_T   = 8;     // KV rows per tile
constexpr int dec_min_rows_per_split = 4*dec_T;
static_assert(dec_HPW*dec_T == WARP_SIZE, "the transposed reduction needs heads x rows == warp size");

struct dec_settings {
    bool    on           = false;
    int64_t max_q        = 5;   // dense ops: faster than tile_new up to 5 Q rows, slower at 8 (GTX 1080 Ti, test-fattn-dec)
    int64_t max_q_sparse = 8;   // ops with a src[5] row list: the alternative is the dsa cuBLAS path, ~10x slower
    int     splits       = 0;
};

const dec_settings & dec_get_settings() {
    static const dec_settings s = [] {
        dec_settings r;
        const char * on     = getenv("LLAMA_FA_DEC");
        const char * max_q  = getenv("LLAMA_FA_DEC_MAX_Q");
        const char * max_qs = getenv("LLAMA_FA_DEC_MAX_Q_SPARSE");
        const char * splits = getenv("LLAMA_FA_DEC_SPLITS");
        r.on     = on && atoi(on) != 0;
        r.max_q  = max_q ? atoll(max_q) : r.max_q;
        r.max_q_sparse = max_qs ? atoll(max_qs) : r.max_q_sparse;
        r.splits = splits ? atoi(splits) : r.splits;
        return r;
    }();
    return s;
}

// warps per block for a head count: 8 (32 heads) or 4 (16 heads); 0 = not supported
int dec_nwarps(const int64_t n_head) {
    return n_head % (8*dec_HPW) == 0 ? 8 : n_head % (4*dec_HPW) == 0 ? 4 : 0;
}

} // namespace

// One step of the transposed butterfly over 2*OFF values per lane: the lanes with bit OFF set keep the upper half,
// the others the lower half, each adding its partner's copy of the half it keeps. A template so that every index
// is a constant and a[] stays in registers.
template <int OFF>
static __device__ __forceinline__ void dec_butterfly(float * a, const int lane) {
    const bool up = lane & OFF;
#pragma unroll
    for (int k = 0; k < OFF; ++k) {
        const float send = up ? a[k] : a[k + OFF];
        const float keep = up ? a[k + OFF] : a[k];
        a[k] = keep + __shfl_xor_sync(0xFFFFFFFF, send, OFF, WARP_SIZE);
    }
    if constexpr (OFF > 1) {
        dec_butterfly<OFF/2>(a, lane);
    }
}

template <int NW>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void fattn_dec_sinks(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ mask,
        const int  * __restrict__ idx,
        float4     * __restrict__ parts,
        float2     * __restrict__ meta,
        const float scale,
        const int n_rows, const int chunk, const int n_head,
        const int64_t nb01, const int64_t nb02, const int64_t nb11, const int64_t nb31, const int64_t s_idx) {
    constexpr int D4 = dec_D/4;
    const int lane   = threadIdx.x;
    const int w      = threadIdx.y;
    const int split  = blockIdx.x;
    const int q      = blockIdx.z;
    const int head0  = blockIdx.y*(NW*dec_HPW) + w*dec_HPW;
    const int r_beg  = split*chunk;
    const int r_end  = min(n_rows, r_beg + chunk);

    __shared__ float4 Ks[2][dec_T][D4];
    __shared__ float  Ms[2][dec_T];
    __shared__ float4 Ps[NW][dec_T];

    float4 qv[dec_HPW][4];
#pragma unroll
    for (int i = 0; i < dec_HPW; ++i) {
        const float4 * Qh = (const float4 *) (Q + q*nb01 + (head0 + i)*nb02);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = Qh[32*j + lane];
            qv[i][j] = make_float4(v.x*scale, v.y*scale, v.z*scale, v.w*scale);
        }
    }

    float4 o[dec_HPW][4];
#pragma unroll
    for (int i = 0; i < dec_HPW; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[i][j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    float m = -FLT_MAX/2.0f; // running max and sum of the head of this lane (head lane/dec_T)
    float l = 0.0f;

    // A tile is fetched into registers (rows w, w + NW, ...), then stored to shared memory as F32 after the compute of
    // the previous tile, so the global load overlaps it.
    constexpr int RPW = dec_T/NW;  // rows per warp
    int4  kr[RPW][2];
    float mr[RPW];
    auto fetch = [&](const int r0) {
#pragma unroll
        for (int k = 0; k < RPW; ++k) {
            const int j = r0 + w + k*NW;
            const int row = j < r_end ? (idx ? idx[q*s_idx + j] : j) : -1;
            if (row >= 0) {
                const int4 * Kr = (const int4 *) (K + row*nb11);
                kr[k][0] = Kr[lane];
                kr[k][1] = Kr[32 + lane];
                mr[k] = __half2float(((const half *) (mask + q*nb31))[row]);
            } else {
                kr[k][0] = make_int4(0, 0, 0, 0);
                kr[k][1] = make_int4(0, 0, 0, 0);
                mr[k] = -INFINITY;
            }
        }
    };
    auto store = [&](const int b) {
#pragma unroll
        for (int k = 0; k < RPW; ++k) {
            const int t = w + k*NW;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const half2 * hp = (const half2 *) &kr[k][h];
                const float2 a = __half22float2(hp[0]);
                const float2 c = __half22float2(hp[1]);
                const float2 d = __half22float2(hp[2]);
                const float2 e = __half22float2(hp[3]);
                Ks[b][t][64*h + 2*lane + 0] = make_float4(a.x, a.y, c.x, c.y);
                Ks[b][t][64*h + 2*lane + 1] = make_float4(d.x, d.y, e.x, e.y);
            }
            if (lane == 0) {
                Ms[b][t] = mr[k];
            }
        }
    };

    if (r_beg < r_end) {
        fetch(r_beg);
        store(0);
    }
    __syncthreads();

    int b = 0;
    for (int r0 = r_beg; r0 < r_end; r0 += dec_T) {
        const bool more = r0 + dec_T < r_end;
        if (more) {
            fetch(r0 + dec_T);
        }

        // K @ Q: partial dot products over this lane's 16 dims, value index i*dec_T + t
        float a[WARP_SIZE];
#pragma unroll
        for (int t = 0; t < dec_T; ++t) {
            float4 kv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                kv[j] = Ks[b][t][32*j + lane];
            }
#pragma unroll
            for (int i = 0; i < dec_HPW; ++i) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s += qv[i][j].x*kv[j].x + qv[i][j].y*kv[j].y + qv[i][j].z*kv[j].z + qv[i][j].w*kv[j].w;
                }
                a[i*dec_T + t] = s;
            }
        }
        // transposed butterfly: after it lane l holds the full dot product of value index l
        dec_butterfly<WARP_SIZE/2>(a, lane);
        const int t_l = lane % dec_T;
        const float s = a[0] + Ms[b][t_l];

        // online softmax per head (groups of dec_T lanes)
        float mx = s;
#pragma unroll
        for (int off = dec_T/2; off >= 1; off >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, off, WARP_SIZE));
        }
        const float m_new = fmaxf(m, mx);
        const float p     = expf(s - m_new);
        const float alpha = expf(m - m_new);
        float ps = p;
#pragma unroll
        for (int off = dec_T/2; off >= 1; off >>= 1) {
            ps += __shfl_xor_sync(0xFFFFFFFF, ps, off, WARP_SIZE);
        }
        l = l*alpha + ps;
        m = m_new;
        ((float *) &Ps[w][t_l])[lane / dec_T] = p;
        __syncwarp();

        // V @ softmax(KQ), V == K
#pragma unroll
        for (int i = 0; i < dec_HPW; ++i) {
            const float al = __shfl_sync(0xFFFFFFFF, alpha, i*dec_T, WARP_SIZE);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[i][j].x *= al; o[i][j].y *= al; o[i][j].z *= al; o[i][j].w *= al;
            }
        }
#pragma unroll
        for (int t = 0; t < dec_T; ++t) {
            const float4 pp = Ps[w][t];
            const float pv[4] = {pp.x, pp.y, pp.z, pp.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 kv = Ks[b][t][32*j + lane];
#pragma unroll
                for (int i = 0; i < dec_HPW; ++i) {
                    o[i][j].x += pv[i]*kv.x; o[i][j].y += pv[i]*kv.y; o[i][j].z += pv[i]*kv.z; o[i][j].w += pv[i]*kv.w;
                }
            }
        }
        __syncwarp();

        if (more) {
            store(b ^ 1);
        }
        __syncthreads();
        b ^= 1;
    }

    const int nsplit = gridDim.x;
#pragma unroll
    for (int i = 0; i < dec_HPW; ++i) {
        const int64_t r = (int64_t(q)*n_head + head0 + i)*nsplit + split;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            parts[r*D4 + 32*j + lane] = o[i][j];
        }
    }
    if (lane % dec_T == 0) {
        meta[(int64_t(q)*n_head + head0 + lane/dec_T)*nsplit + split] = make_float2(m, l);
    }
}

// One block per (head, Q row), one float4 of the output per thread: merge the splits and add the sink logit.
static __global__ void fattn_dec_sinks_combine(
        const float4 * __restrict__ parts,
        const float2 * __restrict__ meta,
        const float  * __restrict__ sinks,
        float4       * __restrict__ dst,
        const int nsplit, const int n_head) {
    constexpr int D4 = dec_D/4;
    const int head = blockIdx.x;
    const int64_t r = int64_t(blockIdx.y)*n_head + head;
    parts += r*nsplit*D4;
    meta  += r*nsplit;

    float M = sinks ? sinks[head] : -FLT_MAX/2.0f;
    for (int s = 0; s < nsplit; ++s) {
        M = fmaxf(M, meta[s].x);
    }
    float  den = sinks ? expf(sinks[head] - M) : 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int s = 0; s < nsplit; ++s) {
        const float  f = expf(meta[s].x - M);
        const float4 v = parts[s*D4 + threadIdx.x];
        den   += f*meta[s].y;
        acc.x += f*v.x; acc.y += f*v.y; acc.z += f*v.z; acc.w += f*v.w;
    }
    const float inv = den > 0.0f ? 1.0f/den : 0.0f;
    dst[r*D4 + threadIdx.x] = make_float4(acc.x*inv, acc.y*inv, acc.z*inv, acc.w*inv);
}

template <int NW>
static void launch_fattn_dec_sinks(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * sel   = dst->src[5];

    cudaStream_t stream = ctx.stream();
    const int id     = ggml_cuda_get_device();
    const int nsm    = ggml_cuda_info().devices[id].nsm;
    const int n_q    = Q->ne[1];
    const int n_head = Q->ne[2];
    const int n_hg   = n_head/(NW*dec_HPW);
    const int n_rows = sel ? sel->ne[0] : K->ne[1];

    int occ = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fattn_dec_sinks<NW>, NW*WARP_SIZE, 0));
    const int max_splits = std::max(1, (n_rows + dec_min_rows_per_split - 1)/dec_min_rows_per_split);
    const int want       = dec_get_settings().splits > 0 ? dec_get_settings().splits
                         : std::max(1, (nsm*std::max(occ, 1))/(n_hg*n_q));
    const int nsplit_req = std::min(want, max_splits);
    const int chunk      = std::max(dec_T, (int) GGML_PAD((n_rows + nsplit_req - 1)/nsplit_req, dec_T));
    const int nsplit     = (n_rows + chunk - 1)/chunk;

    ggml_cuda_pool_alloc<float4> parts(ctx.pool(), (size_t) n_q*n_head*nsplit*(dec_D/4));
    ggml_cuda_pool_alloc<float2> meta (ctx.pool(), (size_t) n_q*n_head*nsplit);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const dim3 grid(nsplit, n_hg, n_q);
    const dim3 block(WARP_SIZE, NW, 1);
    fattn_dec_sinks<NW><<<grid, block, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) mask->data,
        sel ? (const int *) sel->data : nullptr, parts.get(), meta.get(), scale,
        n_rows, chunk, n_head, Q->nb[1], Q->nb[2], K->nb[1], mask->nb[1], sel ? sel->nb[1]/sizeof(int) : 0);
    CUDA_CHECK(cudaGetLastError());

    fattn_dec_sinks_combine<<<dim3(n_head, n_q, 1), dim3(dec_D/4, 1, 1), 0, stream>>>(
        parts.get(), meta.get(), sinks ? (const float *) sinks->data : nullptr, (float4 *) dst->data, nsplit, n_head);
    CUDA_CHECK(cudaGetLastError());
}

static bool same_view(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == b) {
        return true;
    }
    const ggml_tensor * sa = a->view_src ? a->view_src : a;
    const ggml_tensor * sb = b->view_src ? b->view_src : b;
    if (sa != sb || a->view_offs != b->view_offs || a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_fattn_dec_sinks_wants(const ggml_tensor * dst) {
    const dec_settings & st = dec_get_settings();
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * sel   = dst->src[5];
    if (!st.on || Q->ne[1] > (sel ? st.max_q_sparse : st.max_q) || !mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f || dst->op_params[4] > 0 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || Q->ne[0] != dec_D || K->ne[0] != dec_D || V->ne[0] != dec_D) {
        return false;
    }
    if (K->ne[2] != 1 || K->ne[3] != 1 || Q->ne[3] != 1 || dec_nwarps(Q->ne[2]) == 0 || !same_view(K, V)) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 != 0 || Q->nb[2] % 16 != 0 || K->nb[0] != sizeof(half) || K->nb[1] % 16 != 0) {
        return false;
    }
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (sel && (sel->type != GGML_TYPE_I32 || sel->ne[1] < Q->ne[1] || sel->ne[2] != 1 || sel->ne[3] != 1)) {
        return false;
    }
    return mask->ne[1] >= Q->ne[1] && mask->ne[0] >= K->ne[1] && K->ne[1] > 0 && ggml_is_contiguous(dst);
}

void ggml_cuda_flash_attn_ext_dec_sinks(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dec_nwarps(dst->src[0]->ne[2]) == 8) {
        launch_fattn_dec_sinks<8>(ctx, dst);
    } else {
        launch_fattn_dec_sinks<4>(ctx, dst);
    }
}
