// Microbench + byte check of the CUDA backend's host -> device tensor upload (ggml_backend_tensor_set_async), the path
// offloaded MoE prefill streams CPU-resident experts through. Run it twice, GGML_CUDA_H2D_STAGE=0 (driver pageable
// path) and =1 (pinned staging, ggml-cuda/h2d-stage.h), under the server's numactl/env to compare copy rates.
//   test-cuda-h2d-bench [total_MiB=2048] [piece_MiB=0 (0 = one copy of the whole buffer)] [reps=5]
// Prints one line per rep and a summary; exits 1 when the device bytes differ from the source.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const size_t total = (size_t) (argc > 1 ? atoll(argv[1]) : 2048) << 20;
    const size_t piece = (size_t) (argc > 2 ? atoll(argv[2]) : 0) << 20;
    const int    reps  = argc > 3 ? atoi(argv[3]) : 5;
    const char * mode  = getenv("GGML_CUDA_H2D_STAGE");

    ggml_backend_t be = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (!be) { fprintf(stderr, "no CUDA device\n"); return 2; }
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) total);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "device alloc of %zu MiB failed\n", total >> 20); return 2; }

    // pageable source, first-touched here (numactl decides the nodes), like --no-mmap expert weights
    unsigned char * src = (unsigned char *) aligned_alloc(4096, total);
    uint64_t x = 0x243f6a8885a308d3ull;
    for (size_t i = 0; i < total; i += 8) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        memcpy(src + i, &x, std::min<size_t>(8, total - i));
    }

    auto upload = [&]() {
        if (piece == 0) {
            ggml_backend_tensor_set_async(be, t, src, 0, total);
        } else {
            for (size_t off = 0; off < total; off += piece) {
                ggml_backend_tensor_set_async(be, t, src + off, off, std::min(piece, total - off));
            }
        }
        ggml_backend_synchronize(be);
    };

    upload();   // warm: ring allocation, first DMA
    double best = 0, sum = 0;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        upload();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double gbs = total / 1e9 / s;
        best = std::max(best, gbs); sum += gbs;
        printf("rep %d: %.3f s %.2f GB/s\n", r, s, gbs);
    }

    // byte check: whole buffer, then an unaligned partial upload over a cleared region
    std::vector<unsigned char> back(total);
    ggml_backend_tensor_get(t, back.data(), 0, total);
    int bad = memcmp(back.data(), src, total) != 0;
    const size_t off = 12345, len = total / 3 + 77;
    std::vector<unsigned char> zero(len, 0);
    ggml_backend_tensor_set(t, zero.data(), off, len);
    ggml_backend_tensor_set_async(be, t, src + off, off, len);
    ggml_backend_synchronize(be);
    ggml_backend_tensor_get(t, back.data(), 0, total);
    bad |= memcmp(back.data(), src, total) != 0;

    printf("h2d-bench mode=%s total_MiB=%zu piece_MiB=%zu reps=%d mean %.2f GB/s best %.2f GB/s bytes %s\n",
           mode ? mode : "default", total >> 20, piece >> 20, reps, sum / reps, best, bad ? "MISMATCH" : "exact");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    free(src);
    return bad ? 1 : 0;
}
