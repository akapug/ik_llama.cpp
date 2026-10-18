// Tests ggml/src/ggml-cuda/h2d-stage.h (settings + the host copy pool behind pinned H2D staging) without a GPU.
// The staged copy must be byte-exact for every size, slice count and repeat; a wrong slice or a lost wake-up would
// corrupt expert weights on their way to the GPU (offloaded prefill) or hang the server.

#include "../ggml/src/ggml-cuda/h2d-stage.h"

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace hs = ggml_cuda_h2d_stage;

static int failures = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static hs::settings parse(const std::map<std::string, std::string> & env) {
    return hs::parse([&](const char * n) {
        auto it = env.find(n);
        return it == env.end() ? (const char *) nullptr : it->second.c_str();
    });
}

static void test_settings() {
    const auto d = parse({});
    CHECK(d.enabled == (hs::k_default_enabled != 0));
    CHECK(d.chunk == hs::k_default_chunk_mib << 20);
    CHECK(d.bufs == hs::k_default_bufs);
    CHECK(d.threads == hs::k_default_threads);
    CHECK(d.min_bytes == hs::k_default_min_kib << 10);

    const auto off = parse({{"GGML_CUDA_H2D_STAGE", "0"}});
    CHECK(!off.enabled);

    const auto s = parse({{"GGML_CUDA_H2D_STAGE_MIB", "8"}, {"GGML_CUDA_H2D_STAGE_BUFS", "3"},
                          {"GGML_CUDA_H2D_STAGE_THREADS", "5"}, {"GGML_CUDA_H2D_STAGE_MIN_KIB", "0"}});
    CHECK(s.chunk == (size_t) 8 << 20);
    CHECK(s.bufs == 3);
    CHECK(s.threads == 5);
    CHECK(s.min_bytes == 0);

    // clamps: a ring needs 2 chunks to overlap copy with DMA; chunk >= 1 MiB; junk keeps the default
    const auto c = parse({{"GGML_CUDA_H2D_STAGE_BUFS", "1"}, {"GGML_CUDA_H2D_STAGE_MIB", "0"},
                          {"GGML_CUDA_H2D_STAGE_THREADS", "x"}, {"GGML_CUDA_H2D_STAGE", "7"}});
    CHECK(c.bufs == 2);
    CHECK(c.chunk == (size_t) 1 << 20);
    CHECK(c.threads == hs::k_default_threads);
    CHECK(c.enabled);
}

static void test_slices() {
    for (size_t len : {(size_t) 0, (size_t) 1, (size_t) 63, (size_t) 64, (size_t) 4097, (size_t) 33 << 20}) {
        for (int n : {1, 2, 3, 7, 16, 64}) {
            size_t covered = 0, prev = 0;
            for (int i = 0; i < n; ++i) {
                size_t a, b;
                hs::slice(len, n, i, a, b);
                CHECK(a <= b && b <= len);
                CHECK(a == prev || a == b);   // contiguous, no overlap
                CHECK(a % hs::k_slice_align == 0 || a == len);
                covered += b - a;
                if (b > a) prev = b;
            }
            CHECK(covered == len);
        }
    }
}

static void test_pool() {
    std::vector<unsigned char> src((33u << 20) + 7), dst(src.size());
    uint64_t x = 0x9e3779b97f4a7c15ull;
    for (auto & v : src) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = (unsigned char) x;
    }
    for (int n : {1, 3, 16}) {
        hs::copy_pool pool(n);
        CHECK(pool.size() == n);
        for (size_t len : {(size_t) 0, (size_t) 1, (size_t) 4095, ((size_t) 1 << 20) - 1, (size_t) 1 << 20, src.size()}) {
            for (size_t off : {(size_t) 0, (size_t) 3}) {
                if (off + len > src.size()) continue;
                std::fill(dst.begin(), dst.end(), 0xa5);
                pool.copy(dst.data() + off, src.data() + off, len);
                CHECK(memcmp(dst.data() + off, src.data() + off, len) == 0);
                CHECK(off == 0 || dst[off - 1] == 0xa5);                     // nothing before
                CHECK(off + len == dst.size() || dst[off + len] == 0xa5);    // nothing after
            }
        }
        // many back-to-back parallel copies: every wake-up must be seen exactly once
        for (int r = 0; r < 2000; ++r) {
            const size_t len = ((size_t) 1 << 20) + (size_t) r * 97;
            pool.copy(dst.data(), src.data() + r, len, 0);
            if (memcmp(dst.data(), src.data() + r, len) != 0) { CHECK(false); break; }
        }
    }
}

#ifdef __linux__
// copy threads must run on the process mask taken at static init, not on the one CPU the creating thread was pinned
// to (ggml pins the main thread with -cr; a pool started from it would copy on a single core)
static void test_affinity() {
    const int all = CPU_COUNT(&hs::g_initial_affinity);
    CHECK(all >= 1);
    int first = -1;
    for (int i = 0; i < CPU_SETSIZE && first < 0; ++i) if (CPU_ISSET(i, &hs::g_initial_affinity)) first = i;
    cpu_set_t one, saved;
    CPU_ZERO(&one);
    CPU_SET(first, &one);
    pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved);
    pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    {
        hs::copy_pool pool(4);
        for (int i = 1; i < 4; ++i) {
            int c = 0;
            for (int k = 0; k < 2000 && (c = pool.cpus_of(i)) == 0; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(c == all);
        }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved);
}
#endif

int main() {
    test_settings();
    test_slices();
    test_pool();
#ifdef __linux__
    test_affinity();
#endif
    if (failures) {
        fprintf(stderr, "test-cuda-h2d-stage: %d failure(s)\n", failures);
        return 1;
    }
    printf("test-cuda-h2d-stage: OK\n");
    return 0;
}
