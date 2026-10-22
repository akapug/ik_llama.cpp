// test-fattn-dec: the Pascal split-K decode attention kernel (fattn-dec-sinks.cu, LLAMA_FA_DEC=1) against the CPU backend,
// plus a per-op timing. DeepSeek V4 shape: head size 512, K == V (one F16 KV head), 64 or 16 Q heads, F32 sinks, F16 mask,
// F32 precision, few Q rows; dense (every KV row) and sparse (src[5] = the rows whose mask is 0, padded with -1, as
// ggml_mask_to_index builds it for V4 decode). The CPU reference is the generic FA on the full mask (IQK disabled, as
// build_deepseek4 marks it), which equals the sparse result: a masked row adds nothing.
//   test-fattn-dec [device=0] [iters=200]
// The kernel is chosen by the environment (LLAMA_FA_DEC, LLAMA_FA_DEC_MAX_Q); with LLAMA_FA_DEC unset this test sets it
// to 1 and LLAMA_FA_DEC_MAX_Q and LLAMA_FA_DEC_MAX_Q_SPARSE to 8. LLAMA_FA_DEC=0 times the previous device path (tile_new, or the dsa cuBLAS path for
// sparse ops). Exit 0 when every case runs on the device, matches the CPU (NMSE < 1e-3) and a dropped sink is caught.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct shape { int n_head, n_kv, n_tokens, masked_tail; bool sinks; int n_valid; };   // n_valid > 0: sparse, that many rows

struct result { std::vector<float> out; bool supported = false; double us = 0.0; };

result run(ggml_backend_t be, const shape & s, uint32_t seed, int iters, bool use_sel) {
    const int D = 512;
    const bool sparse = s.n_valid > 0;
    const int n_sel = sparse ? GGML_PAD(s.n_valid, 256) : 0;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_pad = GGML_PAD(s.n_tokens, GGML_KQ_MASK_PAD);
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, s.n_tokens, s.n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, s.n_kv, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.n_kv, n_pad, 1, 1);
    ggml_tensor * snk  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.n_head);
    ggml_tensor * sel  = sparse ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_sel, n_pad) : nullptr;
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, q, k, k, mask, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
    out->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
    if (s.sinks) ggml_flash_attn_ext_add_sinks(out, snk);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    if (sparse && use_sel) {
        out->src[5] = sel;
        out->op_params[4] = 0;   // as build_deepseek4 with a selected-row list
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qv(ggml_nelements(q));
    for (auto & v : qv) v = nd(rng);
    std::vector<ggml_fp16_t> kv(ggml_nelements(k));
    for (auto & v : kv) v = ggml_fp32_to_fp16(nd(rng));
    // sparse: per token row, n_valid rows of the n_kv - masked_tail visible ones stay at 0 (a random pick), the rest -inf
    std::vector<ggml_fp16_t> mv(ggml_nelements(mask));
    std::vector<int32_t> iv(sparse ? (size_t) n_sel * n_pad : 0, -1);
    std::vector<int> perm(s.n_kv);
    for (int t = 0; t < n_pad; ++t) {
        std::vector<char> on(s.n_kv, 0);
        const int vis = s.n_kv - s.masked_tail;
        if (sparse) {
            for (int j = 0; j < s.n_kv; ++j) perm[j] = j;
            std::shuffle(perm.begin(), perm.begin() + vis, rng);
            for (int j = 0; j < std::min(s.n_valid, vis); ++j) on[perm[j]] = 1;
        } else {
            for (int j = 0; j < vis; ++j) on[j] = 1;
        }
        int n = 0;
        for (int j = 0; j < s.n_kv; ++j) {
            mv[(size_t) t * s.n_kv + j] = ggml_fp32_to_fp16(on[j] ? 0.0f : -INFINITY);
            if (sparse && on[j]) iv[(size_t) t * n_sel + n++] = j;
        }
    }
    std::vector<float> sv(s.n_head);
    for (auto & v : sv) v = 4.0f + 2.0f * nd(rng);   // sinks carry a large share of the softmax mass
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(snk, sv.data(), 0, ggml_nbytes(snk));
    if (sel) ggml_backend_tensor_set(sel, iv.data(), 0, ggml_nbytes(sel));

    result r;
    r.supported = ggml_backend_supports_op(be, out);
    r.out.resize(ggml_nelements(out));
    if (r.supported) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(out, r.out.data(), 0, ggml_nbytes(out));
        if (iters > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            const auto t1 = std::chrono::steady_clock::now();
            r.us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) { num += (a[i] - ref[i]) * (double) (a[i] - ref[i]); den += ref[i] * (double) ref[i]; }
    return den > 0 ? num / den : (num > 0 ? 1.0 : 0.0);
}

} // namespace

int main(int argc, char ** argv) {
    if (!getenv("LLAMA_FA_DEC")) setenv("LLAMA_FA_DEC", "1", 1);
    if (!getenv("LLAMA_FA_DEC_MAX_Q")) setenv("LLAMA_FA_DEC_MAX_Q", "8", 1);
    if (!getenv("LLAMA_FA_DEC_MAX_Q_SPARSE")) setenv("LLAMA_FA_DEC_MAX_Q_SPARSE", "8", 1);
    const int dev   = argc > 1 ? atoi(argv[1]) : 0;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    ggml_backend_t gpu = ggml_backend_cuda_init(dev, nullptr, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) { printf("FAIL: no backend\n"); return 1; }
    printf("LLAMA_FA_DEC=%s LLAMA_FA_DEC_MAX_Q=%s LLAMA_FA_DEC_SPLITS=%s iters %d\n", getenv("LLAMA_FA_DEC"),
           getenv("LLAMA_FA_DEC_MAX_Q"), getenv("LLAMA_FA_DEC_SPLITS") ? getenv("LLAMA_FA_DEC_SPLITS") : "auto", iters);
    const shape shapes[] = {
        { 64,  512, 1,   0, true, 0 },   // decode, short cache
        { 64, 1024, 1,   0, true, 0 },   // decode over the V4 raw window (512) + 512 compressed rows
        { 64, 2304, 1, 100, true, 0 },   // decode at ~2.2k depth, masked tail
        { 64, 4096, 1, 300, true, 0 },   // ~4k
        { 64, 8192, 1, 0,   true, 0 },   // the whole -c 8192 cache
        { 64, 1024, 1,   0, true, 640 }, // V4 decode as served: 640 = n_swa 128 + top_k 512 selected rows of 1024 (src[5], 768)
        { 64, 4608, 1,   0, true, 640 }, // the same pick from a longer list
        { 64, 2304, 2, 100, true, 0 },   // draft-verify batches
        { 64, 2304, 3, 100, true, 0 },
        { 64, 2304, 5, 100, true, 0 },
        { 64, 1024, 5,   0, true, 640 },
        { 64, 2304, 8, 100, true, 0 },
        { 64, 1024, 8,   0, true, 640 },
        { 16, 1024, 1,   0, true, 0 },   // 16 Q heads per KV head (4-warp blocks)
        { 64, 2304, 1, 100, false, 0 },  // control: no sinks
        { 64,  200, 1,   0, true, 0 },   // fewer rows than one split, a ragged last tile
    };
    int fails = 0;
    uint32_t seed = 1;
    for (const auto & s : shapes) {
        const result g = run(gpu, s, seed, iters, true);
        const result c = run(cpu, s, seed, 0, false);
        ++seed;
        const double e = nmse(g.out, c.out);
        const bool ok = g.supported && c.supported && e < 1e-3;
        fails += ok ? 0 : 1;
        printf("%s heads %d n_kv %d tokens %d masked %d sinks %d sparse %d: device %s, NMSE %.3g, %.1f us/op\n",
               ok ? "PASS" : "FAIL", s.n_head, s.n_kv, s.n_tokens, s.masked_tail, s.sinks, s.n_valid,
               g.supported ? "runs it" : "REFUSES it", e, g.us);
    }
    {   // discrimination: the device result with sinks against the CPU without them must differ well above the bar
        shape with = shapes[2], without = shapes[2];
        without.sinks = false;
        const result g = run(gpu, with, 99, 0, true);
        const result c = run(cpu, without, 99, 0, false);
        const double e = nmse(g.out, c.out);
        const bool ok = g.supported && c.supported && e > 1e-3;
        fails += ok ? 0 : 1;
        printf("%s a dropped sink is caught: NMSE %.3g (bar 1e-3)\n", ok ? "PASS" : "FAIL", e);
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("%s: %d failed\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
