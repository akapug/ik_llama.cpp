// Tests ggml/src/ggml-cuda/pin-budget.h (the GGML_CUDA_PIN_MAX_MIB total pinned-bytes cap) without a GPU.
// Incident it guards (2026-10-16): with --no-mmap, 77 weight chunks of 2295 MiB each passed the per-allocation
// limit and were all pinned (~172 GiB), stalling a 188 GiB box.

#include "../ggml/src/ggml-cuda/pin-budget.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using ggml_cuda_pin_budget::budget;

static int failures = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

constexpr size_t MiB = (size_t) 1 << 20;
constexpr size_t GiB = (size_t) 1 << 30;

static void test_incident() {
    budget b(10 * GiB);
    const size_t chunk = 2295 * MiB;
    int admitted = 0;
    for (int i = 0; i < 77; ++i) {
        admitted += b.try_reserve(chunk);
        CHECK(b.total() <= b.cap);
    }
    // 4 x 2295 = 9180 MiB fits in 10240 MiB, a 5th (11475 MiB) does not
    CHECK(admitted == 4);
    CHECK(b.total() == 4 * chunk);
    CHECK(b.count.load() == 4);
    printf("incident: 77 x 2295 MiB under a 10 GiB cap -> %d admitted, %zu MiB pinned\n", admitted, b.total() / MiB);
}

static void test_release() {
    budget b(10 * GiB);
    const size_t chunk = 2295 * MiB;
    for (int i = 0; i < 77; ++i) {
        b.try_reserve(chunk);
    }
    CHECK(!b.try_reserve(chunk));
    b.release(chunk);
    CHECK(b.total() == 3 * chunk);
    CHECK(b.count.load() == 3);
    CHECK(b.try_reserve(chunk));
    CHECK(!b.try_reserve(chunk));
    for (int i = 0; i < 4; ++i) {
        b.release(chunk);
    }
    CHECK(b.total() == 0);
    CHECK(b.count.load() == 0);
    b.release(chunk); // a stray release clamps at 0 instead of wrapping
    CHECK(b.total() == 0);
    CHECK(b.count.load() == 0);

    // exact fit, a request larger than the cap, a zero cap
    budget e(100 * MiB);
    CHECK(e.try_reserve(60 * MiB));
    CHECK(e.try_reserve(40 * MiB));
    CHECK(!e.try_reserve(1));
    CHECK(!budget(100 * MiB).try_reserve(101 * MiB));
    CHECK(!budget(0).try_reserve(1));
    // near SIZE_MAX the check must not wrap
    budget m(SIZE_MAX);
    CHECK(m.try_reserve(SIZE_MAX - 1));
    CHECK(!m.try_reserve(2));
}

static void test_threads() {
    const int n_threads = 8;

    // fill: every thread races 200 reservations of 7 MiB against a 1000 MiB cap -> exactly 142 fit
    {
        budget b(1000 * MiB);
        std::atomic<int> admitted{0};
        std::atomic<size_t> max_seen{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < n_threads; ++t) {
            ts.emplace_back([&] {
                for (int i = 0; i < 200; ++i) {
                    if (b.try_reserve(7 * MiB)) {
                        admitted.fetch_add(1);
                    }
                    const size_t cur = b.total();
                    size_t m = max_seen.load();
                    while (cur > m && !max_seen.compare_exchange_weak(m, cur)) {}
                }
            });
        }
        for (auto & t : ts) {
            t.join();
        }
        CHECK(admitted.load() == 142);
        CHECK(b.total() == 142 * 7 * MiB);
        CHECK(max_seen.load() <= b.cap);
        printf("threads fill: %d threads x 200 x 7 MiB under 1000 MiB -> %d admitted, max seen %zu MiB\n",
               n_threads, admitted.load(), max_seen.load() / MiB);
    }

    // churn: reserve and release mixed sizes; the total never exceeds the cap and ends equal to what is held
    {
        budget b(256 * MiB);
        std::atomic<size_t> max_seen{0};
        std::atomic<size_t> held_sum{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < n_threads; ++t) {
            ts.emplace_back([&, t] {
                std::vector<size_t> held;
                unsigned x = 12345u + 77u * t;
                for (int i = 0; i < 20000; ++i) {
                    x = x * 1664525u + 1013904223u;
                    const size_t size = (1 + (x >> 24) % 48) * MiB;
                    if (!held.empty() && (x & 0x100)) {
                        b.release(held.back());
                        held.pop_back();
                    } else if (b.try_reserve(size)) {
                        held.push_back(size);
                    }
                    const size_t cur = b.total();
                    size_t m = max_seen.load();
                    while (cur > m && !max_seen.compare_exchange_weak(m, cur)) {}
                }
                size_t sum = 0;
                for (size_t h : held) {
                    sum += h;
                }
                held_sum.fetch_add(sum);
            });
        }
        for (auto & t : ts) {
            t.join();
        }
        CHECK(max_seen.load() <= b.cap);
        CHECK(b.total() == held_sum.load());
        printf("threads churn: max seen %zu of %zu MiB, final %zu MiB == held %zu MiB\n",
               max_seen.load() / MiB, b.cap / MiB, b.total() / MiB, held_sum.load() / MiB);
    }
}

static void test_default_cap() {
    using namespace ggml_cuda_pin_budget;
    const char * meminfo =
        "MemTotal:       197102236 kB\n"
        "MemFree:         3215440 kB\n"
        "MemAvailable:   180000000 kB\n"
        "Buffers:          123456 kB\n";
    CHECK(mem_available(meminfo) == (size_t) 180000000 * 1024);
    CHECK(cap_bytes(nullptr, meminfo) == (size_t) 180000000 * 1024 / 4);
    CHECK(cap_bytes("", meminfo) == (size_t) 180000000 * 1024 / 4);
    // the explicit setting wins, in MiB; 0 or negative pins nothing
    CHECK(cap_bytes("10240", meminfo) == 10 * GiB);
    CHECK(cap_bytes("0", meminfo) == 0);
    CHECK(cap_bytes("-5", meminfo) == 0);
    // missing or unreadable meminfo -> 4096 MiB
    CHECK(cap_bytes(nullptr, "MemTotal: 1000 kB\nMemFree: 10 kB\n") == 4096 * MiB);
    CHECK(cap_bytes(nullptr, nullptr) == 4096 * MiB);
    CHECK(cap_bytes(nullptr, "MemAvailable: garbage\n") == 4096 * MiB);
    // the key only counts at a line start; first line works
    CHECK(mem_available("XMemAvailable: 5 kB\nMemAvailable: 8 kB\n") == 8 * 1024);
    CHECK(mem_available("MemAvailable: 12 kB") == 12 * 1024);
    const size_t live = cap_bytes_from_system(nullptr);
    CHECK(live > 0);
    printf("default cap: fake meminfo -> %zu MiB; this box -> %zu MiB\n",
           cap_bytes(nullptr, meminfo) / MiB, live / MiB);
}

int main() {
    test_incident();
    test_release();
    test_threads();
    test_default_cap();
    if (failures) {
        fprintf(stderr, "test-cuda-pin-budget: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("test-cuda-pin-budget: all checks passed\n");
    return 0;
}
