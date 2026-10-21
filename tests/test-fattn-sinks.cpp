// test-fattn-sinks: GGML_OP_FLASH_ATTN_EXT with attention sinks at head size 512 (DeepSeek V4 shape: K == V, one KV head,
// F32 sinks per Q head, F16 mask, F32 precision) on a CUDA device against the CPU backend. On Pascal (sm_6x) this is the
// tile_new kernel; before its sink support the op was refused and ran on the CPU.
//   test-fattn-sinks [device=0]   exit 0 when every case is supported on the device and matches the CPU (NMSE < 1e-6)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

struct shape { int n_head, n_kv, n_tokens, masked_tail; bool sinks; };

std::vector<float> run(ggml_backend_t be, const shape & s, uint32_t seed, bool & supported) {
    const int D = 512;
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int n_pad = GGML_PAD(s.n_tokens, GGML_KQ_MASK_PAD);
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, s.n_tokens, s.n_head, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, s.n_kv, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.n_kv, n_pad, 1, 1);
    ggml_tensor * snk  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.n_head);
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, q, k, k, mask, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
    if (s.sinks) ggml_flash_attn_ext_add_sinks(out, snk);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qv(ggml_nelements(q));
    for (auto & v : qv) v = nd(rng);
    std::vector<ggml_fp16_t> kv(ggml_nelements(k));
    for (auto & v : kv) v = ggml_fp32_to_fp16(nd(rng));
    std::vector<ggml_fp16_t> mv(ggml_nelements(mask));
    for (int t = 0; t < n_pad; ++t)
        for (int j = 0; j < s.n_kv; ++j)
            mv[(size_t) t * s.n_kv + j] = ggml_fp32_to_fp16(j >= s.n_kv - s.masked_tail ? -INFINITY : 0.0f);
    std::vector<float> sv(s.n_head);
    for (auto & v : sv) v = 2.0f * nd(rng);
    ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(snk, sv.data(), 0, ggml_nbytes(snk));

    supported = ggml_backend_supports_op(be, out);
    std::vector<float> res(ggml_nelements(out));
    if (supported) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

} // namespace

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    ggml_backend_t gpu = ggml_backend_cuda_init(dev, nullptr, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) { printf("FAIL: no backend\n"); return 1; }
    const shape shapes[] = {
        { 64,  512, 1, 0,   true  },   // decode, short cache
        { 64, 2304, 1, 100, true  },   // decode at ~2.2k depth, masked tail (the std prompts)
        { 64, 4096, 1, 300, true  },   // decode at ~4k depth (the long prompt)
        { 64, 2304, 5, 100, true  },   // a 5-token verify batch
        { 16, 1024, 1, 0,   true  },   // 16 Q heads per KV head
        { 64, 2304, 1, 100, false },   // control: no sinks
    };
    int fails = 0;
    uint32_t seed = 1;
    for (const auto & s : shapes) {
        bool sup_g = false, sup_c = false;
        const std::vector<float> g = run(gpu, s, seed, sup_g);
        const std::vector<float> c = run(cpu, s, seed, sup_c);
        ++seed;
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < c.size(); ++i) { num += (g[i] - c[i]) * (double) (g[i] - c[i]); den += c[i] * (double) c[i]; }
        const double nmse = den > 0 ? num / den : 0.0;
        const bool ok = sup_g && sup_c && nmse < 1e-6;
        fails += ok ? 0 : 1;
        printf("%s heads %d n_kv %d tokens %d masked %d sinks %d: device %s, NMSE %.3g\n", ok ? "PASS" : "FAIL",
               s.n_head, s.n_kv, s.n_tokens, s.masked_tail, s.sinks, sup_g ? "runs it" : "REFUSES it", nmse);
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("%s: %d failed\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
