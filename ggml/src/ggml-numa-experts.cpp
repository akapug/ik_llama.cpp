// NUMA-local expert sharding: the config, the id partition and the per-weight owner maps (see ggml-numa-experts.h).

#include "ggml-numa-experts.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

constexpr int k_max = GGML_NUMA_EXPERTS_MAX_GROUPS;

const char * env(const char * name) {
    const char * v = getenv(name);
    return v && *v ? v : nullptr;
}

// comma-separated fields, spaces dropped
std::vector<std::string> fields(const char * s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char * p = s; p && *p; ++p) {
        if (*p == ',') { out.push_back(cur); cur.clear(); } else if (*p != ' ') cur += *p;
    }
    if (s && *s) out.push_back(cur);
    return out;
}

std::vector<int> parse_list(const char * s) {
    std::vector<int> out;
    for (const auto & f : fields(s)) {
        if (f.empty()) continue;
        const size_t dash = f.find('-', 1);
        const int a = atoi(f.c_str());
        const int b = dash == std::string::npos ? a : atoi(f.c_str() + dash + 1);
        for (int v = a; v <= b; ++v) out.push_back(v);
    }
    return out;
}

// MemTotal (kB) of a node, or the CPU count of its cpulist
double node_value(int node, bool cpus) {
#if defined(__linux__)
    const std::string p = "/sys/devices/system/node/node" + std::to_string(node) + (cpus ? "/cpulist" : "/meminfo");
    FILE * f = fopen(p.c_str(), "r");
    if (!f) return 0;
    char buf[4096] = {0};
    const size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[len] = 0;
    if (cpus) {
        for (char & c : buf) if (c == '\n') c = 0;
        return (double) parse_list(buf).size();
    }
    const char * m = strstr(buf, "MemTotal:");
    return m ? atof(m + 9) : 0;
#else
    (void) node; (void) cpus;
    return 0;
#endif
}

struct config {
    std::vector<int> nodes;   // per group
    bool steal = true;
    int samples = 4;
    std::vector<double> w;    // per group, sum 1
    bool fake = false;
    long stats = 0;
};

config read_config() {
    config c;
    const char * e = env("GGML_NUMA_EXPERTS");
    if (!e) return c;
    c.nodes = parse_list(e);
    if ((int) c.nodes.size() > k_max) {
        fprintf(stderr, "ggml numa experts: GGML_NUMA_EXPERTS=%s names %zu groups, at most %d; mode off\n", e, c.nodes.size(), k_max);
        c.nodes.clear();
        return c;
    }
    if (c.nodes.empty() || std::any_of(c.nodes.begin(), c.nodes.end(), [](int n) { return n < 0; })) {
        fprintf(stderr, "ggml numa experts: GGML_NUMA_EXPERTS=%s is not a node list; mode off\n", e);
        c.nodes.clear();
        return c;
    }
    const int G = (int) c.nodes.size();
    c.steal   = (e = env("GGML_NUMA_EXPERTS_STEAL"))   ? atoi(e) != 0 : true;
    c.samples = (e = env("GGML_NUMA_EXPERTS_SAMPLES")) ? std::max(1, atoi(e)) : 4;
    c.fake    = (e = env("GGML_NUMA_EXPERTS_FAKE"))    ? atoi(e) != 0 : false;
    c.stats   = (e = env("GGML_NUMA_EXPERTS_STATS"))   ? atol(e) : 0;
    if ((e = env("GGML_NUMA_EXPERTS_WEIGHTS"))) {
        for (const auto & f : fields(e)) c.w.push_back(std::max(0.0, atof(f.c_str())));
        if ((int) c.w.size() != G) {
            fprintf(stderr, "ggml numa experts: GGML_NUMA_EXPERTS_WEIGHTS has %zu values for %d groups; using the default\n", c.w.size(), G);
            c.w.clear();
        }
    }
    if (c.w.empty()) {
        // per listed node: min(share of memory, share of CPUs); a node served by k groups gives each 1/k of it
        std::vector<int> uniq = c.nodes;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        double mem = 0, cpu = 0;
        std::map<int, std::pair<double, double>> v;
        for (int n : uniq) {
            v[n] = { node_value(n, false), node_value(n, true) };
            mem += v[n].first;
            cpu += v[n].second;
        }
        for (int n : c.nodes) {
            const double ms = mem > 0 ? v[n].first / mem : 1.0 / uniq.size();
            const double cs = cpu > 0 ? v[n].second / cpu : 1.0 / uniq.size();
            c.w.push_back(std::min(ms, cs) / (double) std::count(c.nodes.begin(), c.nodes.end(), n));
        }
    }
    double sum = 0;
    for (double x : c.w) sum += x;
    for (double & x : c.w) x = sum > 0 ? x / sum : 1.0 / G;
    std::string ns, ws;
    for (int g = 0; g < G; ++g) {
        ns += (g ? "," : "") + std::to_string(c.nodes[g]);
        char b[32];
        snprintf(b, sizeof(b), "%s%.3f", g ? "," : "", c.w[g]);
        ws += b;
    }
    fprintf(stderr, "ggml numa experts: groups %d nodes %s weights %s steal %d samples %d fake %d stats %ld\n",
            G, ns.c_str(), ws.c_str(), c.steal ? 1 : 0, c.samples, c.fake ? 1 : 0, c.stats);
    return c;
}

const config & cfg() {
    static const config c = read_config();
    return c;
}

struct counters { uint64_t ops = 0, owned = 0, shared = 0, max_load = 0, weights = 0; };

// the owner map of a weight (or of a fused op's pair of weights)
struct owners {
    std::vector<int8_t> o;
    bool unknown = false;   // some sampled page was not present yet (an mmap not faulted in): rebuilt every 512 uses
    uint64_t uses = 0;
};

struct state {
    std::mutex mu;
    std::map<std::pair<const void *, const void *>, owners> maps;
    counters n;
    std::atomic<uint64_t> stolen{0};

    ~state() { if (cfg().stats > 0) print("exit"); }

    void print(const char * when) {
        fprintf(stderr, "ggml numa experts %s: ops %llu selected owned %llu shared %llu | largest group %.2f experts/op | stolen chunks %llu | %llu weights\n",
                when, (unsigned long long) n.ops, (unsigned long long) n.owned, (unsigned long long) n.shared,
                n.ops ? (double) n.max_load / n.ops : 0.0, (unsigned long long) stolen.load(), (unsigned long long) n.weights);
    }

    // owner group of every expert of one weight, from the nodes of P sampled pages per expert
    std::vector<int8_t> map_one(const ggml_tensor * t, bool & unknown) {
        const config & c = cfg();
        const int n = (int) t->ne[2], G = (int) c.nodes.size();
        std::vector<int8_t> o(n, -1);
        if (c.fake) {
            std::vector<int32_t> p(n);
            ggml_numa_experts_partition(n, nullptr, c.w.data(), G, 0, p.data());
            for (int e = 0; e < n; ++e) o[e] = (int8_t) p[e];
            return o;
        }
#if defined(__linux__)
        const int P = c.samples;
        const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
        const size_t nb2 = t->nb[2];
        std::vector<void *> pages((size_t) n * P);
        std::vector<int> status(pages.size(), -ENOENT);
        for (int e = 0; e < n; ++e) {
            for (int k = 0; k < P; ++k) {
                const uintptr_t a = (uintptr_t) t->data + (uintptr_t) e * nb2 + (uintptr_t) ((2*k + 1) * nb2 / (2*P));
                pages[(size_t) e * P + k] = (void *) (a & ~(page - 1));
            }
        }
        if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) != 0) {
            std::fill(status.begin(), status.end(), -errno);
        }
        for (int e = 0; e < n; ++e) {
            int best = -1, cnt = 0;
            for (int k = 0; k < P; ++k) {
                const int s = status[(size_t) e * P + k];
                if (s < 0) { unknown = true; continue; }
                const int m = (int) std::count_if(status.begin() + (size_t) e * P, status.begin() + (size_t) (e + 1) * P, [s](int x) { return x == s; });
                if (m > cnt) { cnt = m; best = s; }
            }
            if (2 * cnt <= P) continue;   // no majority node: shared
            std::vector<int> gs;
            for (int g = 0; g < G; ++g) if (c.nodes[g] == best) gs.push_back(g);
            if (!gs.empty()) o[e] = (int8_t) gs[e % gs.size()];
        }
#else
        (void) t; (void) unknown;
#endif
        return o;
    }
};

state & st() {
    static state s;
    return s;
}

bool expert_weight(const ggml_tensor * t, int n_as) {
    return t && t->data && t->ne[3] == 1 && t->ne[2] == n_as && (size_t) t->nb[2] * (size_t) t->ne[2] == ggml_nbytes(t);
}

} // namespace

bool ggml_numa_experts_on(void) {
    return !cfg().nodes.empty();
}

int ggml_numa_experts_groups(int * out) {
    const config & c = cfg();
    if (out) for (size_t g = 0; g < c.nodes.size(); ++g) out[g] = c.nodes[g];
    return (int) c.nodes.size();
}

bool ggml_numa_experts_steal(void) {
    return cfg().steal;
}

void ggml_numa_experts_weights(double * out) {
    const config & c = cfg();
    for (size_t g = 0; g < c.w.size(); ++g) out[g] = c.w[g];
}

void ggml_numa_experts_partition(int n, const uint8_t * skip, const double * w, int G, int salt, int32_t * owner) {
    if (G < 1 || n < 1) return;
    double sum = 0;
    for (int g = 0; g < G; ++g) sum += std::max(0.0, w[g]);
    std::vector<double> ws(G);
    for (int g = 0; g < G; ++g) ws[g] = sum > 0 ? std::max(0.0, w[g]) / sum : 1.0 / G;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int> ids;
        for (int e = 0; e < n; ++e) if ((skip && skip[e]) == (pass == 1)) ids.push_back(e);
        const int m = (int) ids.size();
        if (!m) continue;
        // largest-remainder quotas, then smooth weighted round-robin with those quotas: exactly quota[g] picks each,
        // interleaved over the order (the lowest group index wins ties)
        std::vector<int> quota(G);
        std::vector<std::pair<double, int>> rem;
        int given = 0;
        for (int g = 0; g < G; ++g) {
            const double x = m * ws[g];
            quota[g] = (int) x;
            given += quota[g];
            rem.push_back({ x - quota[g], g });
        }
        std::stable_sort(rem.begin(), rem.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int i = 0; given < m; ++i, ++given) quota[rem[i % G].second]++;
        std::vector<long> cur(G, 0);
        const int start = ((salt % m) + m) % m;
        for (int i = 0; i < m; ++i) {
            int best = -1;
            for (int g = 0; g < G; ++g) {
                if (!quota[g]) continue;
                cur[g] += quota[g];
                if (best < 0 || cur[g] > cur[best]) best = g;
            }
            cur[best] -= m;
            owner[ids[(start + i) % m]] = best;
        }
    }
}

const int8_t * ggml_numa_experts_op_begin(const struct ggml_tensor * w0, const struct ggml_tensor * w1,
        const int64_t * counts, int n_as) {
    const config & c = cfg();
    if (c.nodes.empty() || !expert_weight(w0, n_as) || (w1 && !expert_weight(w1, n_as))) return nullptr;
    state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    owners & m = s.maps[{ w0->data, w1 ? w1->data : nullptr }];
    if (m.o.empty() || (m.unknown && ++m.uses % 512 == 0)) {
        if (m.o.empty()) s.n.weights++;
        bool unknown = false;
        m.o = s.map_one(w0, unknown);
        if (w1) {
            const std::vector<int8_t> b = s.map_one(w1, unknown);
            for (int e = 0; e < n_as; ++e) if (m.o[e] != b[e]) m.o[e] = -1;   // up and gate on different nodes: shared
        }
        m.unknown = unknown;
    }
    s.n.ops++;
    int load[k_max] = {0};
    for (int e = 0; e < n_as; ++e) {
        if (!counts[e]) continue;
        if (m.o[e] >= 0) { s.n.owned++; load[m.o[e]]++; } else { s.n.shared++; }
    }
    s.n.max_load += *std::max_element(load, load + k_max);
    if (c.stats > 0 && s.n.ops % (uint64_t) c.stats == 0) s.print("stats");
    return m.o.data();
}

void ggml_numa_experts_count_steal(int n) {
    st().stolen.fetch_add((uint64_t) n, std::memory_order_relaxed);
}

void ggml_numa_experts_forget(const void * base, size_t size) {
    if (!ggml_numa_experts_on() || !base || !size) return;
    state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    const uintptr_t a = (uintptr_t) base, b = a + size;
    auto in = [&](const void * p) { return p && (uintptr_t) p >= a && (uintptr_t) p < b; };
    for (auto it = s.maps.begin(); it != s.maps.end();) {
        it = in(it->first.first) || in(it->first.second) ? s.maps.erase(it) : std::next(it);
    }
}

void ggml_numa_experts_counts(uint64_t out[6]) {
    state & s = st();
    std::lock_guard<std::mutex> lock(s.mu);
    const uint64_t v[6] = { s.n.ops, s.n.owned, s.n.shared, s.stolen.load(), s.n.max_load, s.n.weights };
    memcpy(out, v, sizeof(v));
}
