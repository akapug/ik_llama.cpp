// Grouped 3D mul_mat with a permuted src1, CPU backend vs CUDA backend.
//
// DeepSeek V4/V4.1 attention output: wo_a {D, R*G} is viewed as {D, R, G} and multiplied by the attention output
// reshaped to {D, G, T} and permuted to {D, T, G} (src1 non-contiguous, one matrix per group). On a GTX 1080 Ti (sm_61)
// this op's result departs from the CPU (V4.1 layer 0: sum 15.36 vs 49.22) while every op before it agrees, which turns
// all V4.1 text to garbage once attention runs on the GPU (2x EPYC 7601 + GTX 1080 Ti, 2026-10-20).
//
// Part 2 (sched): the V4.1 pattern through ggml_backend_sched. A producer runs on the CPU (on sm_61 the D=512 flash
// attention is CPU-only), an INPLACE op follows (the inverse rope), then the grouped mul_mat with a GPU weight. If the
// scheduler puts the inplace op on CUDA0 while its memory (view_src) lives in a CPU buffer, the mul_mat split copies that
// tensor in BEFORE the inplace op has written it and multiplies stale data.
//
// GGML_SCHED_INPLACE_FOLLOW=0 turns the scheduler fix off (expect the sched cases to FAIL on any CUDA device).
//
// Usage: test-cuda-grouped-mm [T...]   prints one line per case (type, layout, T, path hint, NMSE) and exits 1 on any
// case above the tolerance. Defaults: V4.1 shapes D=4096 R=1024 G=8, T in {1 2 7 8 9 16 17 32 64}.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct shape { int64_t d, r, g; };

static std::vector<float> run(ggml_backend_t be, ggml_type wt, const shape & s, int64_t t, bool permuted,
                              const std::vector<float> & w, const std::vector<float> & x) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, wt, s.d, s.r*s.g);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.d, s.g, t);
    ggml_tensor * a3 = ggml_reshape_3d(ctx, a, s.d, s.r, s.g);
    ggml_tensor * b3 = ggml_permute(ctx, b, 0, 2, 1, 3);             // {d, t, g}, non-contiguous
    if (!permuted) b3 = ggml_cont(ctx, b3);
    ggml_tensor * y = ggml_mul_mat(ctx, a3, b3);                       // {r, t, g}
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<uint8_t> q(ggml_row_size(wt, s.d) * s.r*s.g);
    if (wt == GGML_TYPE_F32) std::memcpy(q.data(), w.data(), q.size());
    else ggml_quantize_chunk(wt, w.data(), q.data(), 0, s.r*s.g, s.d, nullptr, nullptr);
    ggml_backend_tensor_set(a, q.data(), 0, q.size());
    ggml_backend_tensor_set(b, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// x {d*g, t} on CPU -> p = scale(x, 1) forced to CPU -> scale_inplace(p, 2) -> {d, g, t} -> permute -> mul_mat(W on GPU)
static std::vector<float> run_sched(ggml_backend_t gpu, ggml_backend_t cpu, const shape & s, int64_t t,
                                    const std::vector<uint8_t> & wq, const std::vector<float> & x, std::string & where) {
    ggml_init_params wp = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * a = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, s.d, s.r*s.g);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, gpu);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(a, wq.data(), 0, wq.size());

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.d*s.g, t);
    ggml_set_input(in);
    ggml_tensor * p = ggml_scale(ctx, in, 1.0f);
    ggml_tensor * q = ggml_scale_inplace(ctx, p, 2.0f);
    ggml_tensor * b3 = ggml_permute(ctx, ggml_reshape_3d(ctx, q, s.d, s.g, t), 0, 2, 1, 3);
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_3d(ctx, a, s.d, s.r, s.g), b3);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t bes[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, 2, 1024, false);
    ggml_backend_sched_set_tensor_backend(sched, p, cpu);   // the producer the GPU cannot run (FA D=512 on sm_61)
    ggml_backend_sched_set_tensor_backend(sched, q, gpu);   // the llama graph callback pins layer nodes to the layer device
    ggml_backend_sched_set_tensor_backend(sched, y, gpu);
    ggml_backend_sched_alloc_graph(sched, gf);
    where = std::string("scale_inplace on ") + ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, q));
    ggml_backend_tensor_set(in, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return out;
}

int main(int argc, char ** argv) {
    const shape s = { 4096, 1024, 8 };
    std::vector<int64_t> ts = { 1, 2, 7, 8, 9, 16, 17, 32, 64 };
    if (argc > 1) { ts.clear(); for (int i = 1; i < argc; ++i) ts.push_back(atoll(argv[i])); }
    const double tol = getenv("TOL") ? atof(getenv("TOL")) : 1e-3;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, getenv("NT") ? atoi(getenv("NT")) : 16);
    ggml_backend_t gpu = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!gpu) { fprintf(stderr, "no CUDA device\n"); return 2; }

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w(s.d*s.r*s.g);
    for (auto & v : w) v = 0.02f*nd(rng);

    int bad = 0;
    for (ggml_type wt : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16 }) {
        for (int64_t t : ts) {
            std::vector<float> x(s.d*s.g*t);
            for (auto & v : x) v = nd(rng);
            for (bool permuted : { true, false }) {
                auto rc = run(cpu, wt, s, t, permuted, w, x);
                auto rg = run(gpu, wt, s, t, permuted, w, x);
                double num = 0, den = 0;
                for (size_t i = 0; i < rc.size(); ++i) { num += (rc[i]-rg[i])*(rc[i]-rg[i]); den += rc[i]*rc[i]; }
                const double nmse = num/(den + 1e-30);
                const bool ok = nmse <= tol && std::isfinite(nmse);
                bad += !ok;
                printf("%-5s %-9s T=%-4lld nmse %.3e %s\n", ggml_type_name(wt), permuted ? "permuted" : "cont",
                       (long long) t, nmse, ok ? "OK" : "FAIL");
            }
        }
    }
    {
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, s.d) * s.r*s.g);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), wq.data(), 0, s.r*s.g, s.d, nullptr, nullptr);
        for (int64_t t : ts) {
            std::vector<float> x(s.d*s.g*t), x2(x.size());
            for (auto & v : x) v = nd(rng);
            for (size_t i = 0; i < x.size(); ++i) x2[i] = 2.0f*x[i];
            auto rc = run(cpu, GGML_TYPE_Q8_0, s, t, true, w, x2);
            std::string where;
            auto rg = run_sched(gpu, cpu, s, t, wq, x, where);
            double num = 0, den = 0;
            for (size_t i = 0; i < rc.size(); ++i) { num += (rc[i]-rg[i])*(rc[i]-rg[i]); den += rc[i]*rc[i]; }
            const double nmse = num/(den + 1e-30);
            const bool ok = nmse <= tol && std::isfinite(nmse);
            bad += !ok;
            printf("sched q8_0 cpu-producer+inplace T=%-4lld nmse %.3e %s (%s)\n", (long long) t, nmse, ok ? "OK" : "FAIL", where.c_str());
        }
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    printf("%s: %d failing case(s)\n", bad ? "FAIL" : "OK", bad);
    return bad ? 1 : 0;
}
