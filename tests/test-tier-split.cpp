// GGML_TIER_* (tier-concurrent routed-expert reads): every arm must give the bit-identical MoE output of its reference.
// The parent re-runs this binary with one environment per arm (the knobs are read once per process) and compares the
// per-step output hashes of a fused up/gate + down chain (Q2_K experts, 1 and 5 tokens) and the tier counters.
// Not in CMake; build-droopy-stack.sh links it like test-expert-place:
//   ./test-tier-split [threads]           all arms (default 8 threads; an odd count exercises uneven groups)
//   ./test-tier-split child <threads>     one arm from the environment (used by the parent)
#include "ggml.h"
#include "ggml-tier.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

namespace {

constexpr int K = 256, N = 256, E = 32, USED = 6, STEPS = 4;

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
    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, K, N, E);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, K, N, E);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_K, N, K, E);
    fill_q(up, 1); fill_q(gate, 2); fill_q(down, 3);
    if (const char * a = getenv("TEST_FAKE_ADDR_UP")) {
        // class the first half of the up weight's experts as slow by address (sampled pages of experts 0..E/2-1)
        (void) a;
        char buf[96];
        snprintf(buf, sizeof(buf), "addr:0x%" PRIxPTR "-0x%" PRIxPTR, (uintptr_t) up->data, (uintptr_t) up->data + ggml_nbytes(up)/2);
        setenv("GGML_TIER_FAKE_SLOW", buf, 1);
    }
    for (int T : { 1, 5 }) {
        for (int step = 0; step < STEPS; ++step) {
            ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, USED, T);
            rng r{(uint64_t) (100 + step + 10*T)};
            for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = r.next();
            for (int t = 0; t < T; ++t) for (int j = 0; j < USED; ++j) ((int32_t *) ids->data)[t*USED + j] = (t*11 + j*5 + step) % E;
            ggml_tensor * par = ggml_moe_up_gate(ctx, up, gate, x, ids, GGML_UNARY_OP_SILU);
            ggml_tensor * out = ggml_mul_mat_id(ctx, down, par, ids);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            if (ggml_graph_compute_with_ctx(ctx, gf, nth) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
            printf("T%d step%d par %016" PRIx64 " out %016" PRIx64 "\n", T, step, fnv(par->data, ggml_nbytes(par)), fnv(out->data, ggml_nbytes(out)));
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // the copy threads fill the ring between steps
        }
    }
    uint64_t c[7] = {0};
    ggml_tier_counts(c);
    printf("COUNTS ops %" PRIu64 " fast %" PRIu64 " slow %" PRIu64 " hits %" PRIu64 " misses %" PRIu64 " copies %" PRIu64 " dropped %" PRIu64 "\n",
           c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    ggml_free(ctx);
    return 0;
}

struct result { std::string hashes; uint64_t c[7] = {0}; bool ok = false; };

result run(const std::string & self, const std::string & envs, int nth) {
    result r;
    const std::string cmd = "env " + envs + " " + self + " child " + std::to_string(nth) + " 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return r;
    char line[512];
    while (fgets(line, sizeof(line), p)) {
        unsigned long long v[7];
        if (sscanf(line, "COUNTS ops %llu fast %llu slow %llu hits %llu misses %llu copies %llu dropped %llu",
                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) == 7) {
            for (int i = 0; i < 7; ++i) r.c[i] = v[i];
            r.ok = true;
        } else {
            r.hashes += line;
        }
    }
    r.ok = pclose(p) == 0 && r.ok;
    return r;
}

int node_of_heap() {
    std::vector<char> b(1 << 20, 1);
    void * pages[1] = { b.data() + 4096 };
    int status[1] = { -1 };
    return syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0) == 0 ? status[0] : 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 2 && strcmp(argv[1], "child") == 0) return child(atoi(argv[2]));
    const int nth = argc > 1 ? atoi(argv[1]) : 8;
    char self[4096] = {0};
    if (readlink("/proc/self/exe", self, sizeof(self) - 1) <= 0) { printf("FAIL: no /proc/self/exe\n"); return 1; }
    const std::string fake = "GGML_TIER_FAKE_SLOW=experts:0-3,8-13,20,27-31";
    const std::string grp  = "GGML_MOE_GROUPS=2";
    const std::string node = "GGML_TIER_SLOW_NODES=" + std::to_string(node_of_heap());
    // name, environment, reference (0 = plain, 1 = groups), checks
    enum { NONE = 0, SLOW = 1, HITS = 2 };
    struct arm { const char * name; std::string env; int ref; int check; };
    const std::vector<arm> arms = {
        { "plain",                    "GGML_TIER_X=0",                                                         0, NONE },
        { "groups",                   grp,                                                                     1, NONE },
        { "split-nogroups (G=1)",     "GGML_TIER_SPLIT=2 " + fake,                                            0, SLOW },
        { "groups+split",             grp + " GGML_TIER_SPLIT=2 " + fake,                                     1, SLOW },
        { "groups+split S=1,3",       grp + " GGML_TIER_SPLIT=1,3 " + fake,                                   1, SLOW },
        { "groups+split nosteal",     grp + " GGML_TIER_SPLIT=2 GGML_TIER_STEAL=0 " + fake,                   1, SLOW },
        { "groups+split all-slow",    grp + " GGML_TIER_SPLIT=99 " + fake,                                    1, SLOW },
        { "groups+split mainnomoe",   grp + " GGML_MAIN_NO_MOE=1 GGML_TIER_SPLIT=2 " + fake,                  1, SLOW },
        { "groups+prefetch",          grp + " GGML_TIER_PREFETCH=1 GGML_TIER_RING_MIB=1 " + fake,             1, SLOW | HITS },
        { "groups+split+prefetch now",grp + " GGML_TIER_SPLIT=2 GGML_TIER_PREFETCH=1 GGML_TIER_PREFETCH_WHEN=now GGML_TIER_RING_MIB=1 " + fake, 1, SLOW | HITS },
        { "prefetch tiny ring",       grp + " GGML_TIER_PREFETCH=1 GGML_TIER_RING_MIB=0 " + fake,             1, SLOW },
        { "real node (move_pages)",   grp + " GGML_TIER_SPLIT=2 GGML_TIER_PREFETCH=1 GGML_TIER_RING_MIB=2 " + node, 1, SLOW | HITS },
        { "fake addr range",          grp + " GGML_TIER_SPLIT=2 GGML_TIER_PREFETCH=1 TEST_FAKE_ADDR_UP=1",   1, SLOW | HITS },
        { "split off, slow nodes set",grp + " " + node,                                                       1, NONE },
    };
    std::vector<result> ref(2);
    int fails = 0;
    for (const auto & a : arms) {
        const result r = run(self, a.env, nth);
        if (a.name == std::string("plain")) ref[0] = r;
        if (a.name == std::string("groups")) ref[1] = r;
        const bool same = r.ok && r.hashes == ref[a.ref].hashes && !r.hashes.empty();
        const bool slow = !(a.check & SLOW) || r.c[2] > 0;
        const bool hits = !(a.check & HITS) || (r.c[3] > 0 && r.c[5] > 0);
        const bool pass = same && slow && hits;
        fails += pass ? 0 : 1;
        printf("%s %-28s ops %4" PRIu64 " fast %4" PRIu64 " slow %4" PRIu64 " hits %4" PRIu64 " misses %4" PRIu64 " copies %4" PRIu64 " dropped %4" PRIu64 "%s%s%s\n",
               pass ? "PASS" : "FAIL", a.name, r.c[0], r.c[1], r.c[2], r.c[3], r.c[4], r.c[5], r.c[6],
               same ? "" : " [output differs from reference]", slow ? "" : " [no slow expert used]", hits ? "" : " [no ring hit]");
    }
    printf("plain vs groups outputs %s (threads %d)\n", ref[0].hashes == ref[1].hashes ? "identical" : "differ", nth);
    printf("%s: %d of %zu arms failed\n", fails ? "FAIL" : "OK", fails, arms.size());
    return fails ? 1 : 0;
}
