#pragma once

// Pageable host -> device copies through pinned staging (homelab droopy prefill lane, 2026-10-18).
//
// cudaMemcpyAsync from pageable memory is staged by the driver: one host thread copies the source into a small
// driver-owned pinned buffer, then DMA runs. Offloaded MoE prefill (-cuda offload-batch-size=N) streams the active
// experts of every CPU-resident expert tensor this way, and with GGML_CUDA_NO_PINNED_WEIGHTS (200+ GiB of experts
// cannot be pinned) the stream ran at ~1.7 GB/s on PCIe 3 x16 (~12 GB/s possible): droopy P17, 2026-10-16.
//
// The CUDA side (ggml-cuda.cu) keeps a ring of pinned chunks per device: many threads copy chunk i into a free
// buffer while the DMA of chunk i-1 runs. This header holds the CUDA-free parts (settings, the copy thread pool)
// so tests/test-cuda-h2d-stage.cpp checks them in any build.
//
// Env (named defaults below): GGML_CUDA_H2D_STAGE (1 = on, 0 = the driver's pageable path), GGML_CUDA_H2D_STAGE_MIB
// (chunk size), GGML_CUDA_H2D_STAGE_BUFS (chunks in the ring), GGML_CUDA_H2D_STAGE_THREADS (host copy threads,
// the calling thread included), GGML_CUDA_H2D_STAGE_MIN_KIB (smaller copies keep the driver path).
// The ring is pinned through the CUDA backend's host allocator, so GGML_CUDA_NO_PINNED without
// GGML_CUDA_PIN_STAGING turns staging off and GGML_CUDA_PIN_MAX_MIB counts it.

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace ggml_cuda_h2d_stage {

constexpr int    k_default_enabled = 1;
constexpr size_t k_default_chunk_mib = 32;
constexpr int    k_default_bufs = 4;
constexpr int    k_default_threads = 16;
constexpr size_t k_default_min_kib = 1024;
constexpr size_t k_slice_align = 64;   // copy-thread slices start on cache-line boundaries

struct settings {
    bool   enabled = k_default_enabled != 0;
    size_t chunk = k_default_chunk_mib << 20;
    int    bufs = k_default_bufs;
    int    threads = k_default_threads;
    size_t min_bytes = k_default_min_kib << 10;
};

// get(name) returns the value or nullptr (getenv in production, a table in the test).
template <typename Get>
settings parse(Get get) {
    settings s;
    auto num = [&](const char * name, long long def, long long lo, long long hi) {
        const char * v = get(name);
        if (v == nullptr || *v == 0) {
            return def;
        }
        char * end = nullptr;
        const long long x = strtoll(v, &end, 10);
        if (end == v) {
            return def;
        }
        return std::min(hi, std::max(lo, x));
    };
    s.enabled   = num("GGML_CUDA_H2D_STAGE", k_default_enabled, 0, 1) != 0;
    s.chunk     = (size_t) num("GGML_CUDA_H2D_STAGE_MIB", (long long) k_default_chunk_mib, 1, 1024) << 20;
    s.bufs      = (int) num("GGML_CUDA_H2D_STAGE_BUFS", k_default_bufs, 2, 64);
    s.threads   = (int) num("GGML_CUDA_H2D_STAGE_THREADS", k_default_threads, 1, 256);
    s.min_bytes = (size_t) num("GGML_CUDA_H2D_STAGE_MIN_KIB", (long long) k_default_min_kib, 0, 1ll << 30) << 10;
    return s;
}

inline settings from_env() {
    return parse([](const char * n) { return (const char *) getenv(n); });
}

// [first, last) byte range of slice i of n over len bytes; slices are k_slice_align aligned and cover len exactly.
inline void slice(size_t len, int n, int i, size_t & first, size_t & last) {
    size_t per = (len + n - 1) / n;
    per = (per + k_slice_align - 1) / k_slice_align * k_slice_align;
    first = std::min(len, per * (size_t) i);
    last  = std::min(len, first + per);
}

// Persistent threads that split one memcpy; the caller copies slice 0 and returns when every slice is done.
class copy_pool {
public:
    explicit copy_pool(int n_threads) : n(std::max(1, n_threads)) {
        for (int i = 1; i < n; ++i) {
            workers.emplace_back([this, i] { work(i); });
        }
    }
    ~copy_pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }
    copy_pool(const copy_pool &) = delete;
    copy_pool & operator=(const copy_pool &) = delete;

    int size() const { return n; }

    // Copies below min_parallel bytes run on the calling thread alone.
    void copy(void * dst, const void * src, size_t len, size_t min_parallel = (size_t) 1 << 20) {
        if (n == 1 || len < min_parallel) {
            memcpy(dst, src, len);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            d = (char *) dst;
            s = (const char *) src;
            bytes = len;
            remaining = n - 1;
            ++gen;
        }
        cv.notify_all();
        run(0);
        std::unique_lock<std::mutex> lk(m);
        done.wait(lk, [this] { return remaining == 0; });
    }

private:
    void run(int i) {
        size_t a, b;
        slice(bytes, n, i, a, b);
        if (b > a) {
            memcpy(d + a, s + a, b - a);
        }
    }
    void work(int i) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
            }
            run(i);
            {
                std::lock_guard<std::mutex> lk(m);
                if (--remaining == 0) {
                    done.notify_one();
                }
            }
        }
    }

    const int n;
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv, done;
    uint64_t gen = 0;
    int remaining = 0;
    bool stop = false;
    char * d = nullptr;
    const char * s = nullptr;
    size_t bytes = 0;
};

} // namespace ggml_cuda_h2d_stage
