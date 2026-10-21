// GGML_NUMA_EXPERTS (NUMA-local expert sharding, ggml/src/ggml-numa-experts.h):
//  1. the id partition (in process): quotas, spread, salt, skipped experts, zero weights, fewer experts than groups;
//  2. kernel arms: every arm must give the bit-identical MoE output of its reference (the plain or GGML_MOE_GROUPS run
//     at the same thread count). The parent re-runs this binary with one environment per arm (the knobs are read once
//     per process) and compares per-step hashes of a fused up/gate + down chain (Q2_K, MXFP4, Q8_0; 1, 5 and 32 tokens)
//     and the sharding counters (owned experts must be > 0 when the mode is on);
//  3. loader placement arms (llama_expert_numa_apply on a fake two-layer model): the auto plan binds every whole expert
//     to the node of its partition group (anonymous --no-mmap buffers, and an mmap of a tmpfs file, whose pages it must
//     also fault in), with LLAMA_HOT_EXPERTS experts dealt separately.
// Not in CMake; link like test-expert-place (needs libllama + libggml):
//   ./test-numa-experts [threads] [nodes]     all parts (default 8 threads; nodes = a GGML_NUMA_EXPERTS list for the
//                                             placement arms, default "<heap node>,<heap node>"; on a multi-node box
//                                             e.g. 0-7)
//   ./test-numa-experts child <threads>       one kernel arm from the environment
//   ./test-numa-experts place <nodes> [mmap]  one placement arm
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-numa-experts.h"
#include "ggml-tier.h"
#include "llama-expert-place.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

int fails = 0;

void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    fails += ok ? 0 : 1;
}

int node_of(const void * p) {
    void * pages[1] = { (void *) ((uintptr_t) p & ~(uintptr_t) (sysconf(_SC_PAGESIZE) - 1)) };
    int status[1] = { -1000 };
    if (syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0) != 0) return -1000;
    return status[0];
}

int node_of_heap() {
    std::vector<char> b(1 << 20, 1);
    const int n = node_of(b.data() + 4096);
    return n < 0 ? 0 : n;
}

//
// 1. partition
//

std::vector<int> counts_of(const std::vector<int32_t> & o, int G, const std::vector<uint8_t> * skip = nullptr, bool skipped = false) {
    std::vector<int> c(G, 0);
    for (size_t e = 0; e < o.size(); ++e) {
        if (skip && (bool) (*skip)[e] != skipped) continue;
        if (o[e] >= 0 && o[e] < G) c[o[e]]++;
    }
    return c;
}

void test_partition() {
    auto part = [](int n, const std::vector<double> & w, int salt, const std::vector<uint8_t> * skip = nullptr) {
        std::vector<int32_t> o(n, -1);
        ggml_numa_experts_partition(n, skip ? skip->data() : nullptr, w.data(), (int) w.size(), salt, o.data());
        return o;
    };
    {
        const auto o = part(384, std::vector<double>(8, 1.0), 0);
        const auto c = counts_of(o, 8);
        check(std::all_of(c.begin(), c.end(), [](int x) { return x == 48; }), "partition 384 experts / 8 equal groups = 48 each");
        int run = 1, max_run = 1;
        for (size_t e = 1; e < o.size(); ++e) { run = o[e] == o[e - 1] ? run + 1 : 1; max_run = std::max(max_run, run); }
        check(max_run == 1, "partition equal weights: no two consecutive ids in one group");
        check(part(384, std::vector<double>(8, 1.0), 0) == o, "partition is deterministic");
        const auto s = part(384, std::vector<double>(8, 1.0), 5);
        check(s != o && counts_of(s, 8) == c, "partition salt 5 changes the deal, not the counts");
        check(s[5] == o[0], "partition salt 5 starts the deal at id 5");
    }
    {
        const auto c = counts_of(part(12, { 2, 1, 1 }, 3), 3);
        check(c == std::vector<int>({ 6, 3, 3 }), "partition weights 2:1:1 of 12 = 6/3/3");
    }
    {
        const auto c = counts_of(part(10, { 1, 1, 1 }, 0), 3);
        check(c == std::vector<int>({ 4, 3, 3 }), "partition 10 over 3 equal: largest remainder ties go to group 0");
    }
    {
        std::vector<uint8_t> skip(16, 0);
        for (int e : { 0, 1, 2, 3 }) skip[e] = 1;
        const auto o = part(16, { 1, 1 }, 0, &skip);
        check(counts_of(o, 2, &skip, false) == std::vector<int>({ 6, 6 }) && counts_of(o, 2, &skip, true) == std::vector<int>({ 2, 2 }),
              "partition with 4 banked experts: host 6/6, banked 2/2");
    }
    {
        const auto o = part(9, { 1, 0 }, 4);
        check(counts_of(o, 2) == std::vector<int>({ 9, 0 }), "partition weight 0 group gets no expert");
    }
    {
        const auto c = counts_of(part(3, std::vector<double>(8, 1.0), 0), 8);
        check(*std::max_element(c.begin(), c.end()) == 1 && std::count(c.begin(), c.end(), 1) == 3, "partition 3 experts / 8 groups: one each");
    }
}

//
// 2. kernel arms
//

constexpr int K = 256, N = 256, E = 32, USED = 6, STEPS = 3;

uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

struct rng {
    uint64_t s;
    float next() { s = s*6364136223846793005ull + 1442695040888963407ull; return ((s >> 40) & 0xffffff) / 8388608.0f - 1.0f; }
};

void fill_q(ggml_tensor * t, uint64_t seed) {
    rng r{seed};
    const int64_t rows = t->ne[1]*t->ne[2];
    std::vector<float> f(t->ne[0]*rows);
    for (auto & x : f) x = r.next();
    ggml_quantize_chunk(t->type, f.data(), t->data, 0, rows, t->ne[0], nullptr, nullptr);
}

int child(int nth) {
    ggml_init_params ip = { (size_t) 512 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    for (ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_MXFP4, GGML_TYPE_Q8_0 }) {
        ggml_tensor * up   = ggml_new_tensor_3d(ctx, type, K, N, E);
        ggml_tensor * gate = ggml_new_tensor_3d(ctx, type, K, N, E);
        ggml_tensor * down = ggml_new_tensor_3d(ctx, type, N, K, E);
        fill_q(up, 1); fill_q(gate, 2); fill_q(down, 3);
        for (int T : { 1, 5, 32 }) {
            for (int step = 0; step < STEPS; ++step) {
                ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, USED, T);
                rng r{(uint64_t) (100 + step + 10*T)};
                for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = r.next();
                for (int t = 0; t < T; ++t) for (int j = 0; j < USED; ++j) ((int32_t *) ids->data)[t*USED + j] = (t*11 + j*5 + step*3) % E;
                ggml_tensor * par = ggml_moe_up_gate(ctx, up, gate, x, ids, GGML_UNARY_OP_SILU);
                ggml_tensor * out = ggml_mul_mat_id(ctx, down, par, ids);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
                printf("%s T%d step%d par %016" PRIx64 " out %016" PRIx64 "\n", ggml_type_name(type), T, step,
                       fnv(par->data, ggml_nbytes(par)), fnv(out->data, ggml_nbytes(out)));
                if (getenv("TEST_FORGET")) ggml_numa_experts_forget(up->data, ggml_nbytes(up));   // the next op maps it again
            }
        }
    }
    uint64_t c[6] = {0};
    ggml_numa_experts_counts(c);
    printf("COUNTS ops %" PRIu64 " owned %" PRIu64 " shared %" PRIu64 " stolen %" PRIu64 " maxload %" PRIu64 " weights %" PRIu64 "\n",
           c[0], c[1], c[2], c[3], c[4], c[5]);
    ggml_free(ctx);
    return 0;
}

struct result { std::string hashes; uint64_t c[6] = {0}; bool ok = false; };

result run(const std::string & self, const std::string & envs, int nth) {
    result r;
    const std::string cmd = "env " + envs + " " + self + " child " + std::to_string(nth) + " 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return r;
    char line[512];
    while (fgets(line, sizeof(line), p)) {
        unsigned long long v[6];
        if (sscanf(line, "COUNTS ops %llu owned %llu shared %llu stolen %llu maxload %llu weights %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
            for (int i = 0; i < 6; ++i) r.c[i] = v[i];
            r.ok = true;
        } else {
            r.hashes += line;
        }
    }
    r.ok = pclose(p) == 0 && r.ok;
    return r;
}

void test_kernel(const std::string & self, int nth) {
    const std::string h = std::to_string(node_of_heap());
    const std::string other = "99";   // a node no page sits on: its group owns nothing and must steal (or idle)
    enum { NONE = 0, OWNED = 1, SHARED = 2, STOLEN = 4 };
    struct arm { std::string name, env; int ref; int check; };   // ref 0 = plain, 1 = groups G=2, 2 = groups G=4
    const std::vector<arm> arms = {
        { "plain",                          "GGML_X=0",                                                       0, NONE },
        { "groups G=2",                     "GGML_MOE_GROUPS=2",                                              1, NONE },
        { "groups G=4",                     "GGML_MOE_GROUPS=4",                                              2, NONE },
        { "numa fake 0,0",                  "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1",                 1, OWNED },
        { "numa fake 0,0 nosteal",          "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_NUMA_EXPERTS_STEAL=0", 1, OWNED },
        { "numa fake 4 groups",             "GGML_NUMA_EXPERTS=0,0,0,0 GGML_NUMA_EXPERTS_FAKE=1",             2, OWNED },
        { "numa fake weights 3:1",          "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_NUMA_EXPERTS_WEIGHTS=3,1", 1, OWNED },
        { "numa pages heap,heap",           "GGML_NUMA_EXPERTS=" + h + "," + h,                               1, OWNED },
        { "numa pages heap,99 (steal)",     "GGML_NUMA_EXPERTS=" + h + "," + other,                           1, OWNED | STOLEN },
        { "numa pages heap,99 nosteal",     "GGML_NUMA_EXPERTS=" + h + "," + other + " GGML_NUMA_EXPERTS_STEAL=0", 1, OWNED },
        { "numa pages 99,98 (all shared)",  "GGML_NUMA_EXPERTS=99,98",                                        1, SHARED },
        { "numa + MAIN_NO_MOE",             "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_MAIN_NO_MOE=1", 1, OWNED },
        { "numa + tier split",              "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_TIER_SPLIT=2 GGML_TIER_FAKE_SLOW=experts:0-9", 1, OWNED },
        { "numa + tier prefetch",           "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_TIER_PREFETCH=1 GGML_TIER_RING_MIB=1 GGML_TIER_FAKE_SLOW=experts:0-9", 1, OWNED },
        { "numa + GROUPS=4 ignored",        "GGML_NUMA_EXPERTS=0,0 GGML_NUMA_EXPERTS_FAKE=1 GGML_MOE_GROUPS=4", 1, OWNED },
        { "numa forget every step",         "GGML_NUMA_EXPERTS=" + h + "," + h + " TEST_FORGET=1",            1, OWNED },
    };
    std::vector<result> ref(3);
    for (const auto & a : arms) {
        const result r = run(self, a.env, nth);
        if (a.name == "plain") ref[0] = r;
        if (a.name == "groups G=2") ref[1] = r;
        if (a.name == "groups G=4") ref[2] = r;
        const bool same = r.ok && !r.hashes.empty() && r.hashes == ref[a.ref].hashes;
        const bool owned = !(a.check & OWNED) || r.c[1] > 0;
        const bool shared = !(a.check & SHARED) || (r.c[2] > 0 && r.c[1] == 0);
        const bool stolen = !(a.check & STOLEN) || r.c[3] > 0;
        char buf[512];
        snprintf(buf, sizeof(buf), "kernel T%d %-30s ops %4" PRIu64 " owned %5" PRIu64 " shared %5" PRIu64 " stolen %5" PRIu64 " largest %.2f%s%s%s%s",
                 nth, a.name.c_str(), r.c[0], r.c[1], r.c[2], r.c[3], r.c[0] ? (double) r.c[4] / r.c[0] : 0.0,
                 same ? "" : " [output differs from reference]", owned ? "" : " [no owned expert]",
                 shared ? "" : " [owned experts where all should be shared]", stolen ? "" : " [no chunk stolen]");
        check(same && owned && shared && stolen, buf);
    }
    check(ref[0].hashes == ref[1].hashes || nth % 2, "kernel plain == groups G=2 when the threads divide evenly");
}

//
// 3. loader placement arms
//

int place(const std::string & nodes, bool use_mmap) {
    setenv("GGML_NUMA_EXPERTS", nodes.c_str(), 1);
    const std::string hot = "/tmp/test-numa-experts-hot-" + std::to_string(getpid()) + ".tsv";
    { std::ofstream f(hot); f << "blk.0\t0,5,9\n"; }
    setenv("LLAMA_HOT_EXPERTS", hot.c_str(), 1);
    setenv("LLAMA_EXPERT_PLACEMENT_VERIFY", "1", 1);
    unsetenv("LLAMA_EXPERT_PLACEMENT");
    const size_t page = sysconf(_SC_PAGESIZE);
    constexpr int L = 2, NE = 16;

    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ts;
    llama_model model;
    model.layers.resize(L);
    for (int il = 0; il < L; ++il) {
        ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 8, NE);   // 32 KiB per expert (page multiple)
        ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1000, 3, NE);   // 12000 B per expert (straddles pages)
        ggml_format_name(up, "blk.%d.ffn_up_exps.weight", il);
        ggml_format_name(down, "blk.%d.ffn_down_exps.weight", il);
        model.layers[il].ffn_up_exps = up;
        model.layers[il].ffn_down_exps = down;
        ts.push_back(up);
        ts.push_back(down);
    }
    for (ggml_tensor * t : ts) model.tensors_by_name.emplace_back(ggml_get_name(t), t);

    ggml_backend_buffer_t buf = nullptr;
    void * map = MAP_FAILED;
    size_t total = 0;
    std::string shm;
    if (use_mmap) {
        // the tensors live in a MAP_SHARED mapping of a tmpfs file written with write(2), so no page is mapped yet
        for (ggml_tensor * t : ts) total += GGML_PAD(ggml_nbytes(t), page);
        shm = "/dev/shm/test-numa-experts-" + std::to_string(getpid());
        const int fd = open(shm.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
        std::vector<char> ones(total, 1);
        if (fd < 0 || write(fd, ones.data(), total) != (ssize_t) total) { printf("FAIL: cannot write %s\n", shm.c_str()); return 1; }
        map = mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { printf("FAIL: mmap %s\n", shm.c_str()); return 1; }
        buf = ggml_backend_cpu_buffer_from_ptr(map, total);
        size_t off = 0;
        for (ggml_tensor * t : ts) { ggml_backend_tensor_alloc(buf, t, (char *) map + off); off += GGML_PAD(ggml_nbytes(t), page); }
        int present = 0;
        for (size_t o = 0; o < total; o += page) present += node_of((char *) map + o) >= 0;
        printf("mmap before apply: %d of %zu pages present\n", present, total / page);
    } else {
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    llama_expert_numa_apply(model, use_mmap);
    if (!use_mmap) for (ggml_tensor * t : ts) memset(t->data, 1, ggml_nbytes(t));   // first touch after the policy
    llama_expert_numa_report(model);

    int G = ggml_numa_experts_groups(nullptr);
    std::vector<int> gn(G);
    ggml_numa_experts_groups(gn.data());
    std::vector<double> w(G);
    ggml_numa_experts_weights(w.data());
    int bad = 0, checked = 0, absent = 0;
    for (int il = 0; il < L; ++il) {
        std::vector<uint8_t> skip(NE, 0);
        if (il == 0) for (int e : { 0, 5, 9 }) skip[e] = 1;
        std::vector<int32_t> owner(NE);
        ggml_numa_experts_partition(NE, skip.data(), w.data(), G, il, owner.data());
        for (ggml_tensor * t : { model.layers[il].ffn_up_exps, model.layers[il].ffn_down_exps }) {
            const uintptr_t base = (uintptr_t) t->data, end = base + ggml_nbytes(t);
            for (int e = 0; e < NE; ++e) {
                const uintptr_t a = (base + e * t->nb[2] + page - 1) / page * page;
                const uintptr_t b = std::min<uintptr_t>(base + (e + 1) * t->nb[2], end) / page * page;
                for (uintptr_t p = a; p < b; p += page) {
                    const int got = node_of((void *) p);
                    ++checked;
                    absent += got < 0;
                    if (got != gn[owner[e]]) {
                        if (bad < 10) fprintf(stderr, "%s expert %d page %#lx: node %d, want %d (group %d)\n", t->name, e, (unsigned long) p, got, gn[owner[e]], owner[e]);
                        ++bad;
                    }
                }
            }
        }
    }
    printf("place %s%s: checked %d whole pages, %d misplaced, %d not present\n", nodes.c_str(), use_mmap ? " mmap" : "", checked, bad, absent);
    for (auto & l : model.layers) l.ffn_up_exps = l.ffn_down_exps = nullptr;
    ggml_backend_buffer_free(buf);
    if (map != MAP_FAILED) { munmap(map, total); unlink(shm.c_str()); }
    unlink(hot.c_str());
    ggml_free(ctx);
    return bad || !checked ? 1 : 0;
}

void test_place(const std::string & self, const std::string & nodes) {
    for (const char * mode : { "", " mmap" }) {
        const std::string cmd = self + " place " + nodes + mode + " 2>&1";
        FILE * p = popen(cmd.c_str(), "r");
        std::string out;
        char line[1024];
        while (p && fgets(line, sizeof(line), p)) out += line;
        const bool ok = p && pclose(p) == 0;
        const bool placed = out.find("by the auto plan") != std::string::npos;
        const bool populated = !*mode || out.find("tmpfs mmap: pages populated") != std::string::npos;
        const size_t at = out.find("place " + nodes);
        check(ok && placed && populated, "loader auto plan " + nodes + mode + ": " + (at == std::string::npos ? std::string("no result line") :
              out.substr(at, out.find('\n', at) - at)) + (placed ? "" : " [no auto-plan line]") + (populated ? "" : " [mmap pages not populated]"));
        if (!ok) fputs(out.c_str(), stdout);
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 2 && strcmp(argv[1], "child") == 0) return child(atoi(argv[2]));
    if (argc > 2 && strcmp(argv[1], "place") == 0) return place(argv[2], argc > 3 && strcmp(argv[3], "mmap") == 0);
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    const std::string h = std::to_string(node_of_heap());
    const std::string nodes = argc > 2 ? argv[2] : h + "," + h;
    char self[4096] = {0};
    if (readlink("/proc/self/exe", self, sizeof(self) - 1) <= 0) { printf("FAIL: no /proc/self/exe\n"); return 1; }
    test_partition();
    test_kernel(self, nth);
    if (nth % 2 == 0) test_kernel(self, nth - 1);   // uneven groups: references are the GGML_MOE_GROUPS runs at that count
    test_place(self, nodes);
    printf("%s: %d failed\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
