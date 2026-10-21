// Tier-concurrent routed-expert reads: the tier map, the split knobs and the DRAM ring prefetch (see ggml-tier.h).

#include "ggml-tier.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

std::vector<int> parse_list(const char * s) {
    std::vector<int> out;
    if (!s) return out;
    std::string str(s);
    size_t i = 0;
    while (i < str.size()) {
        const size_t j = std::min(str.find(',', i), str.size());
        const std::string part = str.substr(i, j - i);
        i = j + 1;
        if (part.empty()) continue;
        const size_t dash = part.find('-', 1);
        const int a = atoi(part.c_str());
        const int b = dash == std::string::npos ? a : atoi(part.c_str() + dash + 1);
        for (int v = a; v <= b; ++v) out.push_back(v);
    }
    return out;
}

const char * env(const char * name) {
    const char * v = getenv(name);
    return v && *v ? v : nullptr;
}

struct config {
    bool on = false;
    std::vector<int> split;         // per group; the last value repeats
    bool steal = true;
    std::vector<int> slow_nodes;
    int samples = 4;
    std::vector<int> fake_experts;
    uintptr_t fake_a = 0, fake_b = 0;
    bool prefetch = false;
    size_t ring_mib = 1024;
    std::vector<int> ring_nodes;
    bool ring_bind = false;   // GGML_TIER_RING_POLICY: bind (OOM when the node is full) or preferred (falls back)
    int pf_threads = 1;
    std::vector<int> pf_cpus;
    bool release_at_end = true;
    long stats = 0;
    std::string ctl;      // GGML_TIER_CTL file: run-time split/steal/prefetch
    int ctl_ms = 250;
};

// the run-time values (the environment's at start; GGML_TIER_CTL replaces them while the process runs)
constexpr int k_max_groups = 8;   // = GGML_MOE_MAX_GROUPS
struct runtime {
    std::atomic<int> split[k_max_groups];
    std::atomic<bool> split_any{false}, steal{true}, prefetch{false};
};

runtime & rt() {
    static runtime r;
    return r;
}

void set_split(const std::vector<int> & v) {
    runtime & r = rt();
    bool any = false;
    for (int g = 0; g < k_max_groups; ++g) {
        const int x = v.empty() ? 0 : std::max(0, v[std::min<size_t>((size_t) g, v.size() - 1)]);
        r.split[g].store(x);
        any = any || x > 0;
    }
    r.split_any.store(any);
}

// nodes with memory, split into CPU-less (slow default) and with CPUs (ring default)
void scan_nodes(std::vector<int> & cpuless, std::vector<int> & with_cpus) {
#if defined(__linux__)
    DIR * d = opendir("/sys/devices/system/node");
    if (!d) return;
    while (const dirent * de = readdir(d)) {
        if (strncmp(de->d_name, "node", 4) != 0 || de->d_name[4] < '0' || de->d_name[4] > '9') continue;
        const int n = atoi(de->d_name + 4);
        const std::string p = std::string("/sys/devices/system/node/") + de->d_name + "/cpulist";
        FILE * f = fopen(p.c_str(), "r");
        char buf[256] = {0};
        const bool has = f && fgets(buf, sizeof(buf), f) && buf[0] >= '0' && buf[0] <= '9';
        if (f) fclose(f);
        (has ? with_cpus : cpuless).push_back(n);
    }
    closedir(d);
    std::sort(cpuless.begin(), cpuless.end());
    std::sort(with_cpus.begin(), with_cpus.end());
#else
    (void) cpuless; (void) with_cpus;
#endif
}

#if defined(__linux__)
// the process CPU mask at static init, before any compute thread pins itself: the default mask of the copy threads
// (they are started from a compute thread, whose pinned one-CPU mask they would inherit)
struct initial_mask {
    cpu_set_t set;
    bool ok;
    initial_mask() { CPU_ZERO(&set); ok = sched_getaffinity(0, sizeof(set), &set) == 0; }
};
const initial_mask g_initial_mask;
#endif

std::string join(const std::vector<int> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s.empty() ? "-" : s;
}

config read_config() {
    config c;
    const char * e;
    c.split = parse_list(env("GGML_TIER_SPLIT"));
    if (std::all_of(c.split.begin(), c.split.end(), [](int s) { return s <= 0; })) c.split.clear();
    c.steal = (e = env("GGML_TIER_STEAL")) ? atoi(e) != 0 : true;
    c.samples = (e = env("GGML_TIER_SAMPLES")) ? std::max(1, atoi(e)) : 4;
    if ((e = env("GGML_TIER_FAKE_SLOW"))) {
        if (strncmp(e, "experts:", 8) == 0) {
            c.fake_experts = parse_list(e + 8);
        } else if (strncmp(e, "addr:", 5) == 0) {
            char * end = nullptr;
            c.fake_a = (uintptr_t) strtoull(e + 5, &end, 0);
            c.fake_b = end && *end == '-' ? (uintptr_t) strtoull(end + 1, nullptr, 0) : 0;
        } else {
            fprintf(stderr, "ggml tier: GGML_TIER_FAKE_SLOW=%s is neither experts:<ids> nor addr:<A>-<B>; ignored\n", e);
        }
    }
    c.prefetch = (e = env("GGML_TIER_PREFETCH")) ? atoi(e) != 0 : false;
    c.ring_mib = (e = env("GGML_TIER_RING_MIB")) ? (size_t) std::max(0ll, atoll(e)) : 1024;
    c.ring_bind = (e = env("GGML_TIER_RING_POLICY")) && strcmp(e, "bind") == 0;
    c.pf_threads = (e = env("GGML_TIER_PREFETCH_THREADS")) ? std::max(1, atoi(e)) : 1;
    c.pf_cpus = parse_list(env("GGML_TIER_PREFETCH_CPUS"));
    c.release_at_end = !((e = env("GGML_TIER_PREFETCH_WHEN")) && strcmp(e, "now") == 0);
    c.stats = (e = env("GGML_TIER_STATS")) ? atol(e) : 0;
    c.ctl = (e = env("GGML_TIER_CTL")) ? e : "";
    c.ctl_ms = (e = env("GGML_TIER_CTL_MS")) ? std::max(0, atoi(e)) : 250;
    c.on = !c.split.empty() || c.prefetch || !c.ctl.empty();
    set_split(c.split);
    rt().steal.store(c.steal);
    rt().prefetch.store(c.prefetch);
    if (!c.on) return c;

    std::vector<int> cpuless, with_cpus;
    scan_nodes(cpuless, with_cpus);
    c.slow_nodes = (e = env("GGML_TIER_SLOW_NODES")) ? parse_list(e) : cpuless;
    e = env("GGML_TIER_RING_NODES");
    c.ring_nodes = e ? (strcmp(e, "-") == 0 ? std::vector<int>() : parse_list(e)) : with_cpus;
    if (c.ring_nodes.size() < 2 && !e) c.ring_nodes.clear();
    fprintf(stderr, "ggml tier: GGML_TIER_SPLIT=%s steal=%d slow nodes=%s samples=%d fake=%s prefetch=%d ring=%zu MiB nodes=%s %s "
            "threads=%d cpus=%s when=%s stats=%ld ctl=%s\n", join(c.split).c_str(), c.steal ? 1 : 0, join(c.slow_nodes).c_str(), c.samples,
            !c.fake_experts.empty() ? "experts" : c.fake_b > c.fake_a ? "addr" : "-", c.prefetch ? 1 : 0, c.ring_mib,
            join(c.ring_nodes).c_str(), c.ring_bind ? "bind" : "preferred", c.pf_threads, join(c.pf_cpus).c_str(), c.release_at_end ? "end" : "now", c.stats, c.ctl.empty() ? "-" : c.ctl.c_str());
    return c;
}

const config & cfg() {
    static const config c = read_config();
    return c;
}

// one routed-expert weight tensor
struct weight {
    const char * base = nullptr;
    size_t nb2 = 0;
    int n = 0;
    int n_slow = 0;
    std::vector<uint8_t> slow;         // [n] this weight's tier
    std::vector<uint8_t> slow_pair;    // [n] slow in this weight or in `pair` (a fused up/gate op)
    const weight * pair = nullptr;
    std::vector<const char *> src;     // [n] where the current op reads expert e
    std::vector<int> slot;             // [n] ring slot holding (or copying) expert e, -1
    std::vector<uint8_t> want;         // [n] a copy request is pending
};

enum { SLOT_FREE, SLOT_COPYING, SLOT_READY };

struct slot {
    weight * w = nullptr;
    int e = -1;
    int state = SLOT_FREE;
    int pins = 0;
    uint64_t used = 0;
    bool hit = false;   // read by an op since its copy
};

struct counters {
    uint64_t computes = 0, ops = 0, used_fast = 0, used_slow = 0, hits = 0, misses = 0, copies = 0, copied = 0,
             evicted = 0, wasted = 0, dropped = 0;
};

struct tier_state {
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable cv_done;   // a copy finished (ggml_tier_forget waits for copies out of a freed range)
    std::unordered_map<const void *, std::unique_ptr<weight>> ws;
    size_t n_slow_experts = 0, n_experts = 0, slow_bytes = 0, unknown_pages = 0;
    char * ring = nullptr;
    size_t slot_bytes = 0;
    bool ring_failed = false;
    std::vector<slot> slots;
    uint64_t clock = 0;
    std::deque<std::pair<weight *, int>> pending, queue;
    std::vector<std::thread> threads;
    bool stop = false;
    counters n;

    ~tier_state() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) if (t.joinable()) t.join();
        if (cfg().stats > 0) print("exit");
    }

    void print(const char * when) {
        fprintf(stderr, "ggml tier %s: computes %llu ops %llu used fast %llu slow %llu | ring hits %llu misses %llu copies %llu "
                "(%.1f MiB) evicted %llu unused %llu dropped %llu | map %zu of %zu experts slow (%.2f GiB), %zu unknown pages\n",
                when, (unsigned long long) n.computes, (unsigned long long) n.ops, (unsigned long long) n.used_fast,
                (unsigned long long) n.used_slow, (unsigned long long) n.hits, (unsigned long long) n.misses,
                (unsigned long long) n.copies, n.copied / 1048576.0, (unsigned long long) n.evicted,
                (unsigned long long) n.wasted, (unsigned long long) n.dropped, n_slow_experts, n_experts,
                slow_bytes / 1073741824.0, unknown_pages);
    }

    // the tier map of a weight from the NUMA node of sampled pages of every expert (under mu)
    weight * get(const ggml_tensor * t) {
        auto it = ws.find(t->data);
        if (it != ws.end()) return it->second.get();
        auto w = std::make_unique<weight>();
        const config & c = cfg();
        const bool ok = t->data && t->ne[3] == 1 && t->ne[2] >= 1 && (size_t) t->nb[2] * (size_t) t->ne[2] == ggml_nbytes(t);
        if (ok) {
            w->base = (const char *) t->data;
            w->nb2 = t->nb[2];
            w->n = (int) t->ne[2];
            w->slow.assign(w->n, 0);
            w->src.assign(w->n, nullptr);
            w->slot.assign(w->n, -1);
            w->want.assign(w->n, 0);
            const int P = c.samples;
            const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
            std::vector<void *> pages((size_t) w->n * P);
            std::vector<int> status(pages.size(), -ENOENT);
            for (int e = 0; e < w->n; ++e) {
                for (int k = 0; k < P; ++k) {
                    const uintptr_t a = (uintptr_t) w->base + (uintptr_t) e * w->nb2 + (uintptr_t) ((2*k + 1) * w->nb2 / (2*P));
                    pages[(size_t) e * P + k] = (void *) (a & ~(page - 1));
                }
            }
#if defined(__linux__)
            if (!c.slow_nodes.empty() && syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) != 0) {
                std::fill(status.begin(), status.end(), -errno);
            }
#endif
            for (int e = 0; e < w->n; ++e) {
                for (int k = 0; k < P; ++k) {
                    const int s = status[(size_t) e * P + k];
                    const uintptr_t a = (uintptr_t) pages[(size_t) e * P + k];
                    if (s < 0 && !c.slow_nodes.empty()) unknown_pages++;
                    if ((s >= 0 && std::find(c.slow_nodes.begin(), c.slow_nodes.end(), s) != c.slow_nodes.end()) ||
                        (c.fake_b > c.fake_a && a >= c.fake_a && a < c.fake_b)) {
                        w->slow[e] = 1;
                    }
                }
            }
            for (int e : c.fake_experts) if (e >= 0 && e < w->n) w->slow[e] = 1;
            for (int e = 0; e < w->n; ++e) w->n_slow += w->slow[e];
            n_slow_experts += w->n_slow;
            n_experts += w->n;
            slow_bytes += (size_t) w->n_slow * w->nb2;
        }
        weight * r = w.get();
        ws.emplace(t->data, std::move(w));
        return r;
    }

    // under mu
    bool alloc_ring() {
        if (ring || ring_failed) return ring != nullptr;
        const config & c = cfg();
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        size_t mx = 0;
        for (const auto & kv : ws) if (kv.second->n_slow) mx = std::max(mx, kv.second->nb2);
        const size_t parts = std::max<size_t>(1, c.ring_nodes.size());
        slot_bytes = (mx + parts*page - 1) / (parts*page) * (parts*page);
        const size_t n_slots = slot_bytes ? c.ring_mib * 1048576 / slot_bytes : 0;
        if (!n_slots) {
            fprintf(stderr, "ggml tier: ring of %zu MiB holds no slot of %zu bytes; prefetch off\n", c.ring_mib, slot_bytes);
            ring_failed = true;
            return false;
        }
#if defined(__linux__)
        void * p = mmap(nullptr, n_slots * slot_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "ggml tier: ring mmap of %zu MiB failed (%s); prefetch off\n", n_slots * slot_bytes >> 20, strerror(errno));
            ring_failed = true;
            return false;
        }
        size_t bound_fail = 0;
        if (parts > 1) {
            const size_t part = slot_bytes / parts;
            for (size_t s = 0; s < n_slots; ++s) {
                for (size_t k = 0; k < parts; ++k) {
                    unsigned long mask[16] = {0};
                    const int node = c.ring_nodes[k];
                    if (node < 0 || node >= 1024) continue;
                    mask[node / (8*sizeof(unsigned long))] |= 1ul << (node % (8*sizeof(unsigned long)));
                    const int mode = c.ring_bind ? 2 /* MPOL_BIND */ : 1 /* MPOL_PREFERRED */;
                    if (syscall(SYS_mbind, (char *) p + s*slot_bytes + k*part, part, mode, mask, 1024, 0) != 0) bound_fail++;
                }
            }
        }
        // fault the ring in now (one page write each) and read where its pages landed
        std::vector<void *> pages;
        for (size_t off = 0; off < n_slots * slot_bytes; off += page) {
            ((volatile char *) p)[off] = 0;
            if (off % (64*page) == 0) pages.push_back((char *) p + off);
        }
        std::vector<int> status(pages.size(), -1);
        syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0);
        std::string where;
        for (int node = 0; node < 64; ++node) {
            const long cnt = std::count(status.begin(), status.end(), node);
            if (cnt) where += " n" + std::to_string(node) + " " + std::to_string(cnt * 100 / (long) status.size()) + "%";
        }
        ring = (char *) p;
        slots.assign(n_slots, slot());
        fprintf(stderr, "ggml tier: ring %zu slots x %.2f MiB = %.1f MiB, parts on nodes %s (%s, %zu mbind failures); pages:%s\n",
                n_slots, slot_bytes / 1048576.0, n_slots * slot_bytes / 1048576.0, join(c.ring_nodes).c_str(),
                c.ring_bind ? "bind" : "preferred", bound_fail, where.c_str());
        return true;
#else
        ring_failed = true;
        return false;
#endif
    }

    void copier() {
        const config & c = cfg();
#if defined(__linux__)
        cpu_set_t set = g_initial_mask.set;
        if (!c.pf_cpus.empty()) {
            CPU_ZERO(&set);
            for (int cpu : c.pf_cpus) if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
        }
        if (!c.pf_cpus.empty() || g_initial_mask.ok) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
        std::unique_lock<std::mutex> lock(mu);
        while (true) {
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) return;
            auto [w, e] = queue.front();
            queue.pop_front();
            w->want[e] = 0;
            if (w->slot[e] >= 0) continue;
            if (!alloc_ring() || w->nb2 > slot_bytes) {
                n.dropped++;
                continue;
            }
            // a free slot, else the least recently used unpinned ready one
            int v = -1;
            for (int s = 0; s < (int) slots.size(); ++s) {
                const slot & x = slots[s];
                if (x.state == SLOT_FREE) { v = s; break; }
                if (x.state == SLOT_READY && x.pins == 0 && (v < 0 || x.used < slots[v].used)) v = s;
            }
            if (v < 0) {
                n.dropped++;
                continue;
            }
            slot & x = slots[v];
            if (x.state == SLOT_READY) {
                x.w->slot[x.e] = -1;
                n.evicted++;
                n.wasted += x.hit ? 0 : 1;
            }
            x.w = w;
            x.e = e;
            x.state = SLOT_COPYING;
            x.hit = false;
            w->slot[e] = v;
            char * dst = ring + (size_t) v * slot_bytes;
            const char * src = w->base + (size_t) e * w->nb2;
            const size_t len = w->nb2;
            lock.unlock();
            memcpy(dst, src, len);
            lock.lock();
            x.state = SLOT_READY;
            x.used = ++clock;
            n.copies++;
            n.copied += len;
            cv_done.notify_all();
        }
    }

    // GGML_TIER_CTL: re-read the file when it changed (checked at most every ctl_ms); "split=S[,S1..] steal=0|1
    // prefetch=0|1" (missing keys keep their value). The counters so far are printed first, so each setting's share
    // of the run reads as the difference of two lines.
    std::chrono::steady_clock::time_point ctl_last{};
    struct timespec ctl_mtime{};
    off_t ctl_size = -1;
    void poll_ctl() {
        const config & c = cfg();
        const auto now = std::chrono::steady_clock::now();
        if (now - ctl_last < std::chrono::milliseconds(c.ctl_ms)) return;
        ctl_last = now;
#if defined(__linux__)
        struct stat sb;
        if (stat(c.ctl.c_str(), &sb) != 0) return;
        if (sb.st_mtim.tv_sec == ctl_mtime.tv_sec && sb.st_mtim.tv_nsec == ctl_mtime.tv_nsec && sb.st_size == ctl_size) return;
        ctl_mtime = sb.st_mtim;
        ctl_size = sb.st_size;
        FILE * f = fopen(c.ctl.c_str(), "r");
        if (!f) return;
        char buf[512] = {0};
        const size_t len = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[len] = 0;
        print("ctl-before");
        std::string text(buf);
        for (char & ch : text) if (ch == '\n' || ch == '\t') ch = ' ';
        size_t i = 0;
        while (i < text.size()) {
            const size_t j = std::min(text.find(' ', i), text.size());
            const std::string kv = text.substr(i, j - i);
            i = j + 1;
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
            if (k == "split") set_split(parse_list(v.c_str()));
            else if (k == "steal") rt().steal.store(atoi(v.c_str()) != 0);
            else if (k == "prefetch") rt().prefetch.store(atoi(v.c_str()) != 0);
            else fprintf(stderr, "ggml tier ctl: unknown key %s\n", k.c_str());
        }
        std::vector<int> sp;
        for (int g = 0; g < k_max_groups; ++g) sp.push_back(rt().split[g].load());
        fprintf(stderr, "ggml tier ctl: split=%s steal=%d prefetch=%d (from %s: %s)\n", join(sp).c_str(), rt().steal.load() ? 1 : 0,
                rt().prefetch.load() ? 1 : 0, c.ctl.c_str(), text.c_str());
#endif
    }

    void start_threads() {
        if (!threads.empty()) return;
        for (int i = 0; i < cfg().pf_threads; ++i) threads.emplace_back([this] { copier(); });
    }

    // under mu: hand pending requests to the copy threads, newest kept when they exceed the ring
    void release() {
        if (pending.empty()) return;
        start_threads();
        for (auto & r : pending) queue.push_back(r);
        pending.clear();
        const size_t cap = slots.empty() ? SIZE_MAX : slots.size();
        while (queue.size() > cap) {
            queue.front().first->want[queue.front().second] = 0;
            queue.pop_front();
            n.dropped++;
        }
        cv.notify_all();
    }
};

tier_state & st() {
    static tier_state s;
    return s;
}

thread_local std::vector<int> t_pins; // ring slots pinned by the CPU graph compute this thread runs

} // namespace

bool ggml_tier_on(void) {
    return cfg().on;
}

int ggml_tier_split(int group) {
    return rt().split[std::min(std::max(group, 0), k_max_groups - 1)].load(std::memory_order_relaxed);
}

bool ggml_tier_steal(void) {
    return rt().steal.load(std::memory_order_relaxed);
}

void ggml_tier_op_begin(const struct ggml_tensor * w0, const struct ggml_tensor * w1, const int64_t * counts, int n_as,
        struct ggml_tier_op * op) {
    op->slow = nullptr;
    op->src[0] = op->src[1] = nullptr;
    const config & c = cfg();
    tier_state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    weight * a = s.get(w0);
    weight * b = w1 ? s.get(w1) : nullptr;
    if (!a->n || a->n != n_as || (b && b->n != n_as) || (a->n_slow == 0 && (!b || b->n_slow == 0))) return;
    const uint8_t * slow = a->slow.data();
    if (b) {
        if (a->pair != b) {
            a->slow_pair.resize(a->n);
            for (int e = 0; e < a->n; ++e) a->slow_pair[e] = a->slow[e] | b->slow[e];
            a->pair = b;
        }
        slow = a->slow_pair.data();
    }
    s.n.ops++;
    for (int e = 0; e < n_as; ++e) {
        if (!counts[e]) continue;
        (slow[e] ? s.n.used_slow : s.n.used_fast)++;
    }
    if (rt().split_any.load(std::memory_order_relaxed)) op->slow = slow;
    if (!rt().prefetch.load(std::memory_order_relaxed)) return;
    weight * ws[2] = { a, b };
    for (int i = 0; i < 2; ++i) {
        weight * w = ws[i];
        if (!w) continue;
        for (int e = 0; e < n_as; ++e) {
            if (!counts[e]) continue;
            w->src[e] = w->base + (size_t) e * w->nb2;
            if (!w->slow[e]) continue;
            const int v = w->slot[e];
            if (v >= 0 && s.slots[v].state == SLOT_READY) {
                slot & x = s.slots[v];
                x.pins++;
                x.used = ++s.clock;
                x.hit = true;
                t_pins.push_back(v);
                w->src[e] = s.ring + (size_t) v * s.slot_bytes;
                s.n.hits++;
            } else {
                s.n.misses++;
                if (v < 0 && !w->want[e]) {
                    w->want[e] = 1;
                    s.pending.emplace_back(w, e);
                }
            }
        }
        op->src[i] = w->src.data();
    }
    if (!c.release_at_end) s.release();
}

void ggml_tier_graph_end(void) {
    const config & c = cfg();
    if (!c.on) return;
    tier_state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    for (int v : t_pins) s.slots[v].pins--;
    t_pins.clear();
    s.release();
    s.n.computes++;
    if (c.stats > 0 && s.n.computes % (uint64_t) c.stats == 0) s.print("stats");
    if (!c.ctl.empty()) s.poll_ctl();
}

void ggml_tier_counts(uint64_t out[7]) {
    tier_state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    const uint64_t v[7] = { s.n.ops, s.n.used_fast, s.n.used_slow, s.n.hits, s.n.misses, s.n.copies, s.n.dropped };
    memcpy(out, v, sizeof(v));
}

void ggml_tier_forget(const void * base, size_t size) {
    if (!cfg().on || !base || !size) return;
    tier_state & s = st();
    std::unique_lock<std::mutex> lock(s.mu);
    const uintptr_t a = (uintptr_t) base, b = a + size;
    auto in = [&](const weight * w) { return w && (uintptr_t) w->base >= a && (uintptr_t) w->base < b; };
    s.cv_done.wait(lock, [&] {
        for (const auto & x : s.slots) if (x.state == SLOT_COPYING && in(x.w)) return false;
        return true;
    });
    auto drop = [&](std::deque<std::pair<weight *, int>> & q) {
        q.erase(std::remove_if(q.begin(), q.end(), [&](const std::pair<weight *, int> & r) { return in(r.first); }), q.end());
    };
    drop(s.pending);
    drop(s.queue);
    for (auto & x : s.slots) if (x.state != SLOT_FREE && in(x.w)) x = slot();
    for (auto it = s.ws.begin(); it != s.ws.end();) {
        if (in(it->second.get())) {
            s.n_slow_experts -= it->second->n_slow;
            s.n_experts -= it->second->n;
            s.slow_bytes -= (size_t) it->second->n_slow * it->second->nb2;
            it = s.ws.erase(it);
        } else {
            ++it;
        }
    }
    for (auto & kv : s.ws) if (kv.second->pair && in(kv.second->pair)) kv.second->pair = nullptr;
}
