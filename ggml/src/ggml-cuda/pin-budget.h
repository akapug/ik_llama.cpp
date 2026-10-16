#pragma once

// Total pinned-bytes budget for the GGML_CUDA_PIN_STAGING path (research, hostcut lane).
//
// Pinned (cudaHostRegister'd) pages cannot be reclaimed by the kernel, not even by a memcg MemoryMax.
// A per-allocation limit alone is not enough: with --no-mmap the weights arrive as many chunks that are
// each below that limit (2026-10-16: 77 x 2295 MiB pinned, a 188 GiB box stalled). This budget caps the
// running total; a request that does not fit stays pageable.
//
// Dependency-free (no CUDA, no ggml) so tests/test-cuda-pin-budget.cpp can test it in a CPU-only build.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_cuda_pin_budget {

constexpr size_t k_fallback_cap = (size_t) 4096 << 20;   // when /proc/meminfo has no MemAvailable
constexpr size_t k_avail_divisor = 4;                     // default cap = 25 % of MemAvailable

// MemAvailable in bytes from the text of /proc/meminfo; 0 when the line is missing or unreadable.
inline size_t mem_available(const char * meminfo) {
    if (meminfo == nullptr) {
        return 0;
    }
    static const char key[] = "MemAvailable:";
    for (const char * p = meminfo; p != nullptr; ) {
        if (strncmp(p, key, sizeof(key) - 1) == 0) {
            const char * v = p + sizeof(key) - 1;
            char * end = nullptr;
            const unsigned long long kib = strtoull(v, &end, 10);
            if (end == v) {
                return 0;
            }
            return kib > (SIZE_MAX >> 10) ? SIZE_MAX : (size_t) kib << 10;
        }
        p = strchr(p, '\n');
        p = p ? p + 1 : nullptr;
    }
    return 0;
}

// The cap in bytes: env_mib (GGML_CUDA_PIN_MAX_MIB) when set and non-empty, else 25 % of MemAvailable,
// else k_fallback_cap. A negative or unparsable env value counts as 0 (pin nothing).
inline size_t cap_bytes(const char * env_mib, const char * meminfo) {
    if (env_mib != nullptr && *env_mib) {
        const long long v = atoll(env_mib);
        if (v <= 0) {
            return 0;
        }
        return (unsigned long long) v > (SIZE_MAX >> 20) ? SIZE_MAX : (size_t) v << 20;
    }
    const size_t avail = mem_available(meminfo);
    return avail ? avail / k_avail_divisor : k_fallback_cap;
}

// Reads /proc/meminfo; nullptr text (unreadable) gives the fallback.
inline size_t cap_bytes_from_system(const char * env_mib) {
    char buf[8192];
    size_t n = 0;
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
    }
    buf[n] = 0;
    return cap_bytes(env_mib, n ? buf : nullptr);
}

struct budget {
    explicit budget(size_t cap) : cap(cap) {}

    const size_t        cap;
    std::atomic<size_t> used{0};
    std::atomic<size_t> count{0};

    // Adds size to the total if the total stays within cap; false (nothing changed) otherwise.
    bool try_reserve(size_t size) {
        size_t cur = used.load();
        do {
            if (size > cap || cur > cap - size) {
                return false;
            }
        } while (!used.compare_exchange_weak(cur, cur + size));
        count.fetch_add(1);
        return true;
    }

    // Returns size to the budget (clamped at 0, so a double release cannot wrap).
    void release(size_t size) {
        size_t cur = used.load();
        while (!used.compare_exchange_weak(cur, cur - (size < cur ? size : cur))) {}
        size_t c = count.load();
        while (c > 0 && !count.compare_exchange_weak(c, c - 1)) {}
    }

    size_t total() const { return used.load(); }
};

} // namespace ggml_cuda_pin_budget
