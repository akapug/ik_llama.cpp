#include "llama-expert-place.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-numa-experts.h"
#include "iqk/iqk_quantize.h"

#include <algorithm>
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_MF_STRICT
#define MPOL_MF_STRICT (1 << 0)
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

namespace {

constexpr int k_max_nodes = 1024;

std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(trim(item));
    return out;
}

// "*" = all experts; otherwise a comma list of ids and a-b ranges
std::vector<int> parse_ids(const std::string & spec, int n, const std::string & where) {
    std::vector<int> out;
    if (spec == "*") {
        for (int i = 0; i < n; ++i) out.push_back(i);
        return out;
    }
    for (const auto & part : split(spec, ',')) {
        if (part.empty()) continue;
        const auto dash = part.find('-');
        const int a = std::stoi(part.substr(0, dash));
        const int b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
        if (a < 0 || b < a || b >= n) throw std::runtime_error("expert id out of range in " + where + ": " + part);
        for (int i = a; i <= b; ++i) out.push_back(i);
    }
    return out;
}

// "blk.N" or "N" -> N; a full tensor name -> -1
int parse_layer_key(const std::string & key) {
    std::string k = key;
    if (k.rfind("blk.", 0) == 0) k = k.substr(4);
    if (k.empty() || k.find_first_not_of("0123456789") != std::string::npos) return -1;
    return std::stoi(k);
}

// expert tensors of a layer with the expert ids they hold: local[e] = index of global expert e in the tensor, or -1
// (identity for full tensors; split layers have one set of tensors per partition)
struct layer_expert_tensor {
    ggml_tensor * t;
    std::vector<int> local;
};

std::vector<layer_expert_tensor> layer_expert_tensors(const llama_layer & l) {
    std::vector<layer_expert_tensor> out;
    for (ggml_tensor * t : { l.ffn_up_gate_exps, l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps }) {
        if (!t) continue;
        std::vector<int> local(t->ne[2]);
        for (int e = 0; e < (int) local.size(); ++e) local[e] = e;
        out.push_back({ t, std::move(local) });
    }
    const int n_exp = (int) l.ffn_exps_part_map.size() / 2;
    for (const auto & p : l.ffn_exps_parts) {
        std::vector<int> local(n_exp, -1);
        for (int e = 0; e < n_exp; ++e) if (l.ffn_exps_part_map[e] == p.tag) local[e] = l.ffn_exps_part_map[n_exp + e];
        for (ggml_tensor * t : { p.up_gate, p.up, p.gate, p.down }) if (t) out.push_back({ t, local });
    }
    return out;
}

bool expert_contiguous(const ggml_tensor * t) {
    return t->ne[3] == 1 && t->ne[2] >= 1 && (size_t) t->nb[2] * (size_t) t->ne[2] == ggml_nbytes(t);
}

bool is_host_weight(const ggml_tensor * t) {
    return t && t->buffer && t->data && ggml_backend_buffer_is_host(t->buffer) &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

//
// NUMA placement
//

struct numa_policy {
    std::string text;
    int mode = MPOL_BIND;
    std::vector<int> nodes;
    std::vector<int> parts;   // "rows:A,B,..": the bind policies of each expert's equal row-byte slices, in order
    int min_node() const { return nodes.empty() ? k_max_nodes : *std::min_element(nodes.begin(), nodes.end()); }
    bool has(int node) const { return std::find(nodes.begin(), nodes.end(), node) != nodes.end(); }
};

struct numa_range {
    char * addr;
    size_t len;
    int policy;
};

struct numa_plan {
    std::vector<numa_policy> policies;
    std::vector<numa_range> ranges;
    std::vector<size_t> intended;   // bytes per policy covered by placed pages
    size_t straddle_pages = 0;      // pages shared by two experts with different policies
    size_t unplaced = 0;            // bytes in partial pages at tensor edges (left to the process policy)
    size_t failed = 0;              // bytes whose mbind call failed
    int first_errno = 0;
    size_t page = 4096;
};

numa_plan g_plan;

int policy_index(numa_plan & plan, const std::string & spec) {
    if (spec == "-" || spec == "default") return -1;
    for (size_t i = 0; i < plan.policies.size(); ++i) if (plan.policies[i].text == spec) return (int) i;
    numa_policy p;
    p.text = spec;
    if (spec.rfind("rows:", 0) == 0) {
        // slice k of every expert's bytes (rows are contiguous inside an expert) binds to the k-th node, so a CPU thread
        // group that owns a row range (GGML_MOE_GROUPS, one group per socket via an -cr list in socket order) reads
        // only its own socket's memory
        for (const auto & s : split(spec.substr(5), ',')) if (!s.empty()) p.parts.push_back(policy_index(plan, s));
        if (p.parts.size() < 2) throw std::runtime_error("rows placement needs at least two nodes: " + spec);
        for (int q : p.parts) p.nodes.push_back(plan.policies[q].nodes.front());
        p.mode = -1;
        plan.policies.push_back(p);
        return (int) plan.policies.size() - 1;
    }
    for (const auto & s : split(spec, ',')) {
        if (s.empty()) continue;
        const int node = std::stoi(s);
        if (node < 0 || node >= k_max_nodes) throw std::runtime_error("bad NUMA node in placement: " + spec);
        p.nodes.push_back(node);
    }
    if (p.nodes.empty()) throw std::runtime_error("empty NUMA node list in placement: " + spec);
    p.mode = p.nodes.size() == 1 ? MPOL_BIND : MPOL_INTERLEAVE;
    plan.policies.push_back(p);
    return (int) plan.policies.size() - 1;
}

long sys_mbind(void * addr, size_t len, int mode, const unsigned long * mask, unsigned long maxnode, unsigned flags) {
#ifdef __linux__
    return syscall(SYS_mbind, addr, len, mode, mask, maxnode, flags);
#else
    (void) addr; (void) len; (void) mode; (void) mask; (void) maxnode; (void) flags;
    errno = ENOSYS;
    return -1;
#endif
}

long sys_move_pages(unsigned long count, void ** pages, int * status) {
#ifdef __linux__
    return syscall(SYS_move_pages, 0, count, pages, nullptr, status, 0);
#else
    (void) count; (void) pages; (void) status;
    errno = ENOSYS;
    return -1;
#endif
}

} // namespace

namespace {

struct plan_counts { size_t tensors = 0, dense = 0, skipped = 0, autos = 0; };

// the placement source: a LLAMA_EXPERT_PLACEMENT file, GGML_NUMA_EXPERTS (auto plan), both, or neither (nullptr)
const char * placement_path() {
    const char * p = getenv("LLAMA_EXPERT_PLACEMENT");
    return p && *p ? p : nullptr;
}

bool placement_on() {
    return placement_path() || ggml_numa_experts_on();
}

// LLAMA_HOT_EXPERTS layer -> expert ids banked on a GPU (the same file llama_hot_experts reads; a bad line is left for
// it to report)
std::map<int, std::vector<int>> hot_ids() {
    std::map<int, std::vector<int>> out;
    const char * path = getenv("LLAMA_HOT_EXPERTS");
    if (!path || !*path) return out;
    std::ifstream in(path);
    std::string line;
    for (int ln = 1; std::getline(in, line); ++ln) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::stringstream ss(trim(line));
        std::string key, ids;
        ss >> key >> ids;
        const int il = parse_layer_key(key);
        if (il < 0 || ids.empty()) continue;
        try {
            const auto v = parse_ids(ids, INT_MAX, path);
            out[il].insert(out[il].end(), v.begin(), v.end());
        } catch (const std::exception &) {
        }
    }
    return out;
}

// GGML_NUMA_EXPERTS auto plan: expert e of layer il goes whole to the node of group owner[e] of the id partition (salt
// il, so the deal starts at a different id per layer); experts banked on a GPU (LLAMA_HOT_EXPERTS) are dealt after the
// host ones, so the experts decode reads on the CPU stay balanced across the groups
struct auto_plan {
    std::vector<int> nodes;      // per group
    std::vector<double> w;       // per group
    std::vector<int> policy;     // per group: index of its node's bind policy
    std::map<int, std::vector<int>> hot;
    std::map<std::pair<int, int>, std::vector<int32_t>> owner;   // (layer, n_expert) -> owner group per expert

    const std::vector<int32_t> & owners(int il, int n) {
        auto it = owner.find({ il, n });
        if (it != owner.end()) return it->second;
        std::vector<uint8_t> skip(n, 0);
        auto h = hot.find(il);
        if (h != hot.end()) for (int e : h->second) if (e >= 0 && e < n) skip[e] = 1;
        std::vector<int32_t> o(n, 0);
        ggml_numa_experts_partition(n, skip.data(), w.data(), (int) nodes.size(), il, o.data());
        return owner[{ il, n }] = std::move(o);
    }
};

// the placement plan of the model's host weights (only == nullptr) or of one tensor (only != nullptr: the loader binds a
// freshly allocated, not yet read buffer, so no warning for tensors that are not host weights yet)
numa_plan build_plan(const llama_model & model, const char * path, const ggml_tensor * only, plan_counts & cnt) {
    std::ifstream in;
    if (path) {
        in.open(path);
        if (!in) throw std::runtime_error(std::string("cannot read LLAMA_EXPERT_PLACEMENT file ") + path);
    }

    numa_plan plan;
#ifdef __linux__
    plan.page = (size_t) sysconf(_SC_PAGESIZE);
#endif

    // a key is a layer ("blk.N" / "N"), an exact tensor name, or "re:<ECMAScript regex>" searched in tensor names
    struct rule { std::string key; int layer; std::string experts; int policy; int line; std::shared_ptr<std::regex> re; };
    auto names = [](const rule & r, const char * name) { return r.re ? std::regex_search(name, *r.re) : r.key == name; };
    std::vector<rule> rules;
    std::string line;
    for (int ln = 1; std::getline(in, line); ++ln) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string w; ss >> w;) f.push_back(w);
        if (f.size() != 3) throw std::runtime_error(std::string(path) + ":" + std::to_string(ln) + ": expected 3 fields (tensor-or-layer, experts, node)");
        auto re = f[0].rfind("re:", 0) == 0 ? std::make_shared<std::regex>(f[0].substr(3)) : nullptr;
        rules.push_back({ f[0], re ? -1 : parse_layer_key(f[0]), f[1], policy_index(plan, f[2]), ln, re });
    }
    std::unique_ptr<auto_plan> autop;
    if (ggml_numa_experts_on()) {
        autop = std::make_unique<auto_plan>();
        autop->nodes.resize(ggml_numa_experts_groups(nullptr));
        ggml_numa_experts_groups(autop->nodes.data());
        autop->w.resize(autop->nodes.size());
        ggml_numa_experts_weights(autop->w.data());
        for (int n : autop->nodes) autop->policy.push_back(policy_index(plan, std::to_string(n)));
        autop->hot = hot_ids();
    }
    plan.intended.assign(plan.policies.size(), 0);

    const size_t page = plan.page;
    // bind one tensor: pol[i] = policy of its i-th block of nb2 bytes (an expert, or a group of a grouped dense weight);
    // a rows policy splits each block's bytes in order over its nodes
    auto place = [&](ggml_tensor * t, const std::vector<int> & pol, size_t nb2) {
        const int n_exp = (int) pol.size();
        const uintptr_t base = (uintptr_t) t->data;
        const uintptr_t end  = base + ggml_nbytes(t);
        const uintptr_t p0   = (base + page - 1) / page * page;   // round inward at the tensor edges:
        const uintptr_t p1   = end / page * page;                 // edge pages may hold other tensors
        if (pol.front() >= 0) plan.unplaced += p0 - base;
        if (pol.back()  >= 0) plan.unplaced += end - p1;

        // the bind/interleave policy of one byte: its expert's policy, or for a rows policy the part its offset falls in
        auto at = [&](uintptr_t a) {
            const int e = (int) std::min<uintptr_t>((a - base) / nb2, n_exp - 1);
            if (pol[e] < 0 || plan.policies[pol[e]].parts.empty()) return pol[e];
            const auto & parts = plan.policies[pol[e]].parts;
            return parts[std::min(parts.size() - 1, (size_t) ((a - base - (size_t) e * nb2) * parts.size() / nb2))];
        };
        // one policy per page: the owner of most of the page wins (a page holds at most one boundary, so the owners of
        // the two middle bytes decide); ties go to the policy with the lowest node number (DRAM nodes come before
        // CPU-less PMem nodes on this platform)
        auto page_policy = [&](uintptr_t p) {
            const int a = at(p), b = at(p + page - 1);
            if (a == b) return a;
            ++plan.straddle_pages;
            const int m0 = at(p + page / 2 - 1), m1 = at(p + page / 2);
            if (m0 == m1) return m0;
            if (m0 < 0) return m1;
            if (m1 < 0) return m0;
            return plan.policies[m0].min_node() <= plan.policies[m1].min_node() ? m0 : m1;
        };

        uintptr_t run = p0;
        int run_pol = p0 < p1 ? page_policy(p0) : -1;
        auto flush = [&](uintptr_t stop) {
            if (run_pol < 0 || stop <= run) return;
            plan.ranges.push_back({ (char *) run, (size_t) (stop - run), run_pol });
        };
        for (uintptr_t p = p0 + page; p < p1; p += page) {
            const int pp = page_policy(p);
            if (pp != run_pol) {
                flush(p);
                run = p;
                run_pol = pp;
            }
        }
        flush(p1);
    };

    size_t & n_tensors = cnt.tensors, & n_skipped = cnt.skipped, & n_dense = cnt.dense;
    std::unordered_set<const ggml_tensor *> seen;   // expert tensors: placed by the loop below or left alone
    for (size_t il = 0; il < model.layers.size(); ++il) {
        for (const auto & lt : layer_expert_tensors(model.layers[il])) {
            ggml_tensor * t = lt.t;
            seen.insert(t);
            if (only && t != only) continue;
            std::vector<const rule *> mine;
            for (const auto & r : rules) {
                if (r.layer == (int) il || (r.layer < 0 && names(r, t->name))) mine.push_back(&r);
            }
            if (mine.empty() && !autop) continue;
            if (!is_host_weight(t) || !expert_contiguous(t)) {
                // GPU-resident expert tensors (-ngl without --cpu-moe/-ot) are not the CPU's to shard: quiet for the auto plan
                if (!mine.empty()) ++n_skipped;
                if (!only && !mine.empty()) LLAMA_LOG_WARN("llama_expert_numa_apply: %s is not a host-resident expert-contiguous tensor; skipped\n", t->name);
                continue;
            }
            const int n_exp = (int) t->ne[2];
            const size_t nb2 = t->nb[2];
            std::vector<int> pol(n_exp, -1);
            if (mine.empty()) {
                // GGML_NUMA_EXPERTS and no rule for this tensor: the auto plan (global ids; a split layer maps them)
                const auto & o = autop->owners((int) il, (int) lt.local.size());
                for (size_t e = 0; e < lt.local.size(); ++e) if (lt.local[e] >= 0) pol[lt.local[e]] = autop->policy[o[e]];
                ++cnt.autos;
            }
            for (const rule * r : mine) {
                const std::string where = std::string(path) + ":" + std::to_string(r->line);
                if (r->layer < 0) {
                    // a rule naming the tensor counts experts inside it
                    for (int e : parse_ids(r->experts, n_exp, where)) pol[e] = r->policy;
                } else {
                    // a layer rule counts the model's expert ids; a split layer maps them into its partitions
                    for (int e : parse_ids(r->experts, (int) lt.local.size(), where)) if (lt.local[e] >= 0) pol[lt.local[e]] = r->policy;
                }
            }
            if (std::all_of(pol.begin(), pol.end(), [](int p) { return p < 0; })) continue;
            ++n_tensors;
            place(t, pol, nb2);
        }
    }

    // any other host weight a name or regex rule matches (attention, dense FFN, shared experts, output head): the experts
    // field is "*" (the tensor is one block) or "/G" (G equal consecutive blocks, e.g. the DeepSeek-V4 grouped
    // attn_output_a, whose grouped matmul gives each group's rows to every node's threads); with -cr in node order and the
    // static row split of a dense matmul, node k's threads then read slice k of every block
    for (const auto & [name, t] : model.tensors_by_name) {
        if (seen.count(t) || (only && t != only)) continue;
        const rule * r = nullptr;
        for (const auto & x : rules) if (x.layer < 0 && names(x, name.c_str())) r = &x;   // the last matching rule wins
        if (!r) continue;
        if (!is_host_weight(t) || !ggml_is_contiguous(t)) {
            ++n_skipped;
            if (!only) LLAMA_LOG_WARN("llama_expert_numa_apply: %s is not a contiguous host weight; skipped\n", name.c_str());
            continue;
        }
        const std::string where = std::string(path) + ":" + std::to_string(r->line);
        const int g = r->experts == "*" ? 1 : r->experts.rfind("/", 0) == 0 ? std::stoi(r->experts.substr(1)) : 0;
        if (g < 1 || ggml_nbytes(t) % g) throw std::runtime_error(where + ": a non-expert tensor takes \"*\" or \"/G\" (G dividing its bytes): " + name);
        ++n_dense;
        place(t, std::vector<int>(g, r->policy), ggml_nbytes(t) / g);
    }

    return plan;
}

// an mmap'ed model: keep the planned ranges that lie in a mapping of a tmpfs or hugetlbfs file and fault their pages in
// (MADV_POPULATE_READ, else a read per page). For such shared memory mbind sets the file's own policy and MPOL_MF_MOVE
// migrates the pages this process maps (a page cache page of a disk file is reallocated by the process policy the next
// time it is read, so disk mmaps are left alone). Returns false when no range is left.
bool mmap_ranges(numa_plan & plan) {
#ifdef __linux__
    std::vector<std::pair<uintptr_t, uintptr_t>> shm;
    std::map<std::string, bool> fs;   // path -> on tmpfs/hugetlbfs
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        std::stringstream ss(line);
        std::string range, perms, off, dev, inode, file;
        ss >> range >> perms >> off >> dev >> inode;
        std::getline(ss, file);
        file = trim(file);
        if (file.empty() || file[0] != '/') continue;
        auto it = fs.find(file);
        if (it == fs.end()) {
            struct statfs sf;
            const bool shmem = statfs(file.c_str(), &sf) == 0 &&
                ((unsigned long) sf.f_type == 0x01021994ul /* TMPFS_MAGIC */ || (unsigned long) sf.f_type == 0x958458f6ul /* HUGETLBFS_MAGIC */);
            it = fs.emplace(file, shmem).first;
        }
        if (!it->second) continue;
        const auto dash = range.find('-');
        shm.push_back({ (uintptr_t) std::stoull(range.substr(0, dash), nullptr, 16), (uintptr_t) std::stoull(range.substr(dash + 1), nullptr, 16) });
    }
    std::vector<numa_range> keep;
    for (const auto & r : plan.ranges) {
        const uintptr_t a = (uintptr_t) r.addr, b = a + r.len;
        if (!std::any_of(shm.begin(), shm.end(), [&](const auto & m) { return a >= m.first && b <= m.second; })) continue;
        if (madvise(r.addr, r.len, MADV_POPULATE_READ) != 0) {
            volatile char sink = 0;
            for (size_t o = 0; o < r.len; o += plan.page) sink = sink + r.addr[o];
        }
        keep.push_back(r);
    }
    plan.ranges = std::move(keep);
    return !plan.ranges.empty();
#else
    (void) plan;
    return false;
#endif
}

size_t g_prebound = 0, g_prebound_tensors = 0;   // bytes / tensors bound by llama_expert_numa_prebind before their read

} // namespace

void llama_expert_numa_prebind(const llama_model & model, const ggml_tensor * t) {
    const char * path = placement_path();
    if (!placement_on() || (getenv("LLAMA_EXPERT_PREBIND") && !atoi(getenv("LLAMA_EXPERT_PREBIND")))) return;
    plan_counts cnt;
    const numa_plan plan = build_plan(model, path, t, cnt);
    for (const auto & r : plan.ranges) {
        const numa_policy & np = plan.policies[r.policy];
        unsigned long mask[k_max_nodes / (8 * sizeof(unsigned long))] = {};
        for (int n : np.nodes) mask[n / (8 * sizeof(unsigned long))] |= 1ul << (n % (8 * sizeof(unsigned long)));
        if (sys_mbind(r.addr, r.len, np.mode, mask, k_max_nodes, 0) == 0) g_prebound += r.len;   // no MOVE: nothing faulted yet
    }
    g_prebound_tensors += plan.ranges.empty() ? 0 : 1;
}

void llama_expert_numa_apply(const llama_model & model, bool use_mmap) {
    if (!placement_on()) return;
    const char * path = placement_path();
    const char * what = path ? path : "GGML_NUMA_EXPERTS auto plan";
    const bool strict = getenv("LLAMA_EXPERT_PLACEMENT_STRICT") && atoi(getenv("LLAMA_EXPERT_PLACEMENT_STRICT"));
    plan_counts cnt;
    numa_plan plan = build_plan(model, path, nullptr, cnt);
    const size_t n_tensors = cnt.tensors, n_dense = cnt.dense, n_skipped = cnt.skipped;
    if (use_mmap && !mmap_ranges(plan)) {
        LLAMA_LOG_WARN("%s: %s: no planned range lies in a tmpfs/hugetlbfs mapping; a disk-file mmap ignores mbind, so placement "
                "needs --no-mmap or a tmpfs model copy; placement skipped\n", __func__, what);
        return;
    }
    const unsigned long maxnode = k_max_nodes;
    if (g_prebound) {
        LLAMA_LOG_INFO("%s: %zu tensors were bound before their read (%.1f MiB; LLAMA_EXPERT_PREBIND=0 turns this off)\n",
                __func__, g_prebound_tensors, g_prebound / 1048576.0);
        g_prebound = g_prebound_tensors = 0;
    }

    // bind (and migrate, for pages already faulted in) the policy with the highest node first: on this platform the
    // CPU-less PMem node has the highest number, so cold pages leave DRAM before hot pages need room in it
    std::vector<size_t> order(plan.policies.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return plan.policies[a].min_node() > plan.policies[b].min_node(); });
    std::vector<numa_range> placed;
    for (size_t pi : order) {
        const numa_policy & np = plan.policies[pi];
        unsigned long mask[k_max_nodes / (8 * sizeof(unsigned long))] = {};
        for (int n : np.nodes) mask[n / (8 * sizeof(unsigned long))] |= 1ul << (n % (8 * sizeof(unsigned long)));
        const unsigned flags = MPOL_MF_MOVE | (strict ? MPOL_MF_STRICT : 0);
        for (const auto & r : plan.ranges) {
            if (r.policy != (int) pi) continue;
            if (sys_mbind(r.addr, r.len, np.mode, mask, maxnode, flags) != 0) {
                if (!plan.first_errno) plan.first_errno = errno;
                plan.failed += r.len;
                continue;
            }
            plan.intended[pi] += r.len;
            placed.push_back(r);
        }
    }
    plan.ranges = std::move(placed);

    LLAMA_LOG_INFO("%s: %s: %zu expert (%zu by the auto plan) + %zu other tensors placed in %zu ranges (%zu skipped), %zu straddle pages, %.2f MiB left at tensor edges%s\n",
            __func__, what, n_tensors, cnt.autos, n_dense, plan.ranges.size(), n_skipped, plan.straddle_pages, plan.unplaced / 1048576.0,
            use_mmap ? " (tmpfs mmap: pages populated, then moved)" : "");
    for (size_t i = 0; i < plan.policies.size(); ++i) {
        if (!plan.policies[i].parts.empty()) continue;   // a rows policy's bytes are counted under its parts
        LLAMA_LOG_INFO("%s:   node(s) %-8s %s %10.1f MiB\n", __func__, plan.policies[i].text.c_str(),
                plan.policies[i].mode == MPOL_BIND ? "bind      " : "interleave", plan.intended[i] / 1048576.0);
    }
    if (plan.failed) {
        LLAMA_LOG_ERROR("%s: mbind failed for %.2f GiB (first error: %s)%s\n", __func__, plan.failed / 1073741824.0,
                strerror(plan.first_errno), plan.first_errno == ENOMEM ? " - vm.max_map_count too low?" : "");
    }
    g_plan = std::move(plan);
}

void llama_expert_numa_report(const llama_model & model) {
    (void) model;
    if (g_plan.ranges.empty()) return;
    const char * env = getenv("LLAMA_EXPERT_PLACEMENT_VERIFY");
    const size_t stride = env ? (size_t) std::max(0, atoi(env)) : 64;
    if (stride == 0) return;

    const size_t page = g_plan.page;
    std::map<int, size_t> per_node;   // sampled pages per node (negative = -errno, e.g. not present)
    size_t sampled = 0, outside = 0;
    std::vector<void *> pages;
    std::vector<int> owner;
    std::vector<int> status;
    auto drain = [&]() {
        if (pages.empty()) return;
        status.assign(pages.size(), 0);
        if (sys_move_pages(pages.size(), pages.data(), status.data()) != 0) {
            std::fill(status.begin(), status.end(), -errno);
        }
        for (size_t i = 0; i < pages.size(); ++i) {
            per_node[status[i]]++;
            if (status[i] < 0 || !g_plan.policies[owner[i]].has(status[i])) ++outside;
        }
        sampled += pages.size();
        pages.clear();
        owner.clear();
    };
    for (const auto & r : g_plan.ranges) {
        for (size_t off = 0; off < r.len; off += page * stride) {
            pages.push_back(r.addr + off);
            owner.push_back(r.policy);
            if (pages.size() == 16384) drain();
        }
    }
    drain();
    const double scale = (double) page * (double) stride / 1048576.0;
    std::string line;
    for (const auto & [node, n] : per_node) {
        char buf[96];
        if (node >= 0) snprintf(buf, sizeof(buf), " node%d=%.1fMiB", node, n * scale);
        else           snprintf(buf, sizeof(buf), " err%d=%.1fMiB", -node, n * scale);
        line += buf;
    }
    LLAMA_LOG_INFO("%s: verified (1 page in %zu):%s; %.2f%% of sampled pages are off their target\n",
            __func__, stride, line.c_str(), sampled ? 100.0 * outside / sampled : 0.0);
}

//
// Expert partitions: split model tensors + static GPU hot-expert set
//

const char * llama_expert_part_tag(int tag) {
    static const char * tags[LLAMA_EXPERT_PART_MAX] = { "hot", "warm", "cold", "p3", "p4", "p5", "p6", "p7" };
    return tag >= 0 && tag < LLAMA_EXPERT_PART_MAX ? tags[tag] : "?";
}

struct llama_hot_banks {
    struct bank { ggml_tensor * t; std::vector<int32_t> slices; }; // slices: source expert indices, in bank order
    std::map<std::string, bank> by_name;          // "hot.<source tensor name>" -> bank
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};

struct llama_hot_experts::impl {
    struct layer {
        std::vector<part> small; // batches <= max_batch (with GPU banks)
        std::vector<part> large; // larger batches; empty for an unsplit layer (single path)
        bool active = false;
    };
    std::vector<layer> layers;
    std::vector<ggml_context_ptr> ctxs;           // per-context maps
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::shared_ptr<llama_hot_banks> banks;       // GPU banks, shared with the model's other contexts
    int64_t max_batch = 8;
};

llama_hot_experts::~llama_hot_experts() = default;

namespace {

// a source of experts in a layer: the full tensors, or one partition of a split layer
struct expert_source {
    llama_hot_experts::part w;   // tensors (map unset)
    std::vector<int32_t> local;  // global expert id -> index in these tensors, or -1
    std::vector<int> banked;     // global expert ids copied to the GPU bank of this source
};

std::vector<expert_source> layer_sources(const llama_layer & l, int n_expert) {
    std::vector<expert_source> out;
    if (!l.ffn_exps_parts.empty()) {
        for (const auto & p : l.ffn_exps_parts) {
            expert_source s;
            s.w.up = p.up; s.w.gate = p.gate; s.w.up_gate = p.up_gate; s.w.down = p.down;
            s.w.tag = llama_expert_part_tag(p.tag);
            s.local.assign(n_expert, -1);
            for (int e = 0; e < n_expert; ++e) {
                if (l.ffn_exps_part_map[e] == p.tag) s.local[e] = l.ffn_exps_part_map[n_expert + e];
            }
            out.push_back(std::move(s));
        }
    } else if (l.ffn_down_exps) {
        expert_source s;
        s.w.up = l.ffn_up_exps; s.w.gate = l.ffn_gate_exps; s.w.up_gate = l.ffn_up_gate_exps; s.w.down = l.ffn_down_exps;
        s.w.tag = "cold";
        s.local.resize(n_expert);
        for (int e = 0; e < n_expert; ++e) s.local[e] = e;
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<ggml_tensor *> part_tensors(const llama_hot_experts::part & p) {
    std::vector<ggml_tensor *> out;
    for (ggml_tensor * t : { p.up_gate, p.up, p.gate, p.down }) if (t) out.push_back(t);
    return out;
}

} // namespace

bool llama_backend_runs_expert_type(ggml_backend_t backend, ggml_type type) {
    if (ggml_backend_is_cpu(backend)) return true;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) return false;
    const int64_t ne0 = 8 * ggml_blck_size(type);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx.get(), type, ne0, 8, 2);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    return ggml_backend_supports_op(backend, ggml_mul_mat_id(ctx.get(), as, b, ids));
}

std::unique_ptr<llama_hot_experts> llama_hot_experts::create(const llama_model & model,
        const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, bool allow_banks) {
    const bool split = std::any_of(model.layers.begin(), model.layers.end(), [](const llama_layer & l) { return !l.ffn_exps_parts.empty(); });
    const char * path = getenv("LLAMA_HOT_EXPERTS");
    if (path && !*path) path = nullptr;
    if (path && !allow_banks) {
        LLAMA_LOG_WARN("llama_hot_experts: --moe-cache-mib is active; LLAMA_HOT_EXPERTS is ignored\n");
        path = nullptr;
    }
    if (!split && !path) return nullptr;
    if (backends.size() != bufts.size()) throw std::runtime_error("hot experts: backend metadata is inconsistent");

    ggml_backend_t cpu = nullptr;
    for (auto * b : backends) if (ggml_backend_is_cpu(b)) cpu = b;
    if (!cpu) throw std::runtime_error("hot experts: no CPU backend");

    auto res = std::unique_ptr<llama_hot_experts>(new llama_hot_experts());
    auto & p = *(res->pimpl = std::make_unique<impl>());
    p.layers.resize(model.layers.size());
    // The CUDA fast decode kernels skip negative ids (rows stay zero); the two-path graph with GPU banks is limited
    // to small batches (prefill reads the model's own tensors).
    if (const char * mb = getenv("LLAMA_HOT_EXPERTS_MAX_BATCH")) p.max_batch = std::clamp<int64_t>(atoi(mb), 0, 8);
    const int n_exp = (int) model.hparams.n_expert;

    std::map<int, std::vector<int>> hot;
    std::map<int, std::string> hot_dev;  // optional third column: the backend that holds this layer's bank (e.g. CUDA0)
    if (path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error(std::string("cannot read LLAMA_HOT_EXPERTS file ") + path);
        std::string line;
        for (int ln = 1; std::getline(in, line); ++ln) {
            const auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            line = trim(line);
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string key, ids, dev;
            ss >> key >> ids >> dev;
            const int il = parse_layer_key(key);
            if (il >= (int) model.layers.size() && !ids.empty()) {
                // the file names a layer this model lacks: it was written for another model (e.g. the target, while this
                // context loads a small speculative draft that reads the same env) -> no hot set for this model
                LLAMA_LOG_WARN("llama_hot_experts: %s:%d names blk.%d but this model has %d layers; hot experts inactive for this model\n",
                    path, ln, il, (int) model.layers.size());
                hot.clear();
                break;
            }
            if (il < 0 || ids.empty()) {
                throw std::runtime_error(std::string(path) + ":" + std::to_string(ln) + ": expected 'blk.N<TAB>expert,list'");
            }
            const auto parsed = parse_ids(ids, INT_MAX, std::string(path) + ":" + std::to_string(ln));
            if (!parsed.empty() && *std::max_element(parsed.begin(), parsed.end()) >= n_exp) {
                // the file names an expert this model lacks: written for another model (a speculative draft with fewer
                // experts reads the same env) -> no hot set for this model, as for a missing layer above
                LLAMA_LOG_WARN("llama_hot_experts: %s:%d names expert %d but this model has %d experts; hot experts inactive for this model\n",
                    path, ln, *std::max_element(parsed.begin(), parsed.end()), n_exp);
                hot.clear();
                break;
            }
            auto & v = hot[il];
            v.insert(v.end(), parsed.begin(), parsed.end());
            if (!dev.empty()) hot_dev[il] = dev;
        }
        if (hot.empty()) path = nullptr;
        if (!split && !path) return nullptr;
    }

    // backend that runs a partition: its weights' backend (host buffers -> CPU)
    auto backend_of = [&](const ggml_tensor * t) -> int {
        if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
            for (size_t i = 0; i < backends.size(); ++i) if (backends[i] == cpu) return (int) i;
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            if (!ggml_backend_is_cpu(backends[i]) && ggml_backend_supports_buft(backends[i], ggml_backend_buffer_get_type(t->buffer))) return (int) i;
        }
        throw std::runtime_error(std::string("hot experts: no backend for ") + t->name);
    };

    struct bank_fill { ggml_tensor * bank; const ggml_tensor * src; std::vector<int32_t> slices; };

    // bank type of a source tensor on a device: its own type when the device runs it, else the base type of a
    // run-time repacked type (-rtr repacks host tensors to CPU-only interleaved layouts; the fill un-repacks),
    // else GGML_TYPE_COUNT (no bank)
    std::map<std::pair<int, int>, bool> runs_cache;
    auto runs = [&](int device, ggml_type type) {
        auto it = runs_cache.find({ device, (int) type });
        if (it != runs_cache.end()) return it->second;
        return runs_cache[{ device, (int) type }] = llama_backend_runs_expert_type(backends[device], type);
    };
    auto bank_type = [&](int device, const ggml_tensor * src) {
        if (runs(device, src->type)) return src->type;
        const auto base = (ggml_type) iqk_unrepacked_type(src->type);
        return base != src->type && runs(device, base) ? base : GGML_TYPE_COUNT;
    };
    struct map_fill  { ggml_tensor * map; std::vector<int32_t> values; };
    std::vector<bank_fill> bank_fills;
    std::vector<map_fill> map_fills;
    std::map<int, ggml_context *> ctx_of;       // backend index -> metadata context of this context's maps
    std::map<int, ggml_context *> bank_ctx_of;  // backend index -> metadata context of new banks (owned by the model)
    std::map<int, ggml_backend_buffer_type_t> buft_of;
    auto banks = model.hot_banks.lock();        // banks another context of this model already holds
    std::vector<ggml_context_ptr> bank_ctxs;
    auto new_ctx = [&](int b, ggml_backend_buffer_type_t buft, bool bank) {
        auto & of = bank ? bank_ctx_of : ctx_of;
        auto it = of.find(b);
        if (it != of.end()) return it->second;
        ggml_init_params ip = { 8 * LLAMA_EXPERT_PART_MAX * model.layers.size() * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (!c) throw std::runtime_error("hot experts: ggml_init failed");
        (bank ? bank_ctxs : p.ctxs).emplace_back(c);
        buft_of[b] = buft;
        return of[b] = c;
    };
    // a bank of this model on this buffer type with this type and shape, or nullptr
    auto shared_bank = [&](const std::string & name, ggml_type type, const ggml_tensor * src, const std::vector<int32_t> & slices,
            ggml_backend_buffer_type_t buft) -> ggml_tensor * {
        if (!banks) return nullptr;
        auto it = banks->by_name.find(name);
        if (it == banks->by_name.end()) return nullptr;
        ggml_tensor * t = it->second.t;
        const bool same = it->second.slices == slices && t->type == type && t->ne[0] == src->ne[0] && t->ne[1] == src->ne[1] &&
                t->buffer && ggml_backend_buffer_get_type(t->buffer) == buft;
        return same ? t : nullptr;
    };
    auto new_map = [&](int b, ggml_backend_buffer_type_t buft, std::vector<int32_t> values, const char * what, int il, const char * tag) {
        ggml_tensor * m = ggml_new_tensor_2d(new_ctx(b, buft, false), GGML_TYPE_I32, 1, n_exp);
        ggml_format_name(m, "%s.map-%s-%d", what, tag, il);
        map_fills.push_back({ m, std::move(values) });
        return m;
    };

    size_t n_hot = 0, n_split = 0, n_bank_layers = 0, n_shared = 0, shared_bytes = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const llama_layer & l = model.layers[il];
        const bool is_split = !l.ffn_exps_parts.empty();
        auto hit = hot.find((int) il);
        std::vector<int> want;
        if (hit != hot.end()) {
            want = hit->second;
            std::sort(want.begin(), want.end());
            want.erase(std::unique(want.begin(), want.end()), want.end());
        }
        if (!is_split && want.empty()) continue;
        auto sources = layer_sources(l, n_exp);
        if (sources.empty()) continue;

        // GPU banks: each listed expert is copied from the source that holds it; one bank per source (one type each)
        int device = -1;
        if (!want.empty()) {
            bool ok = true;
            for (auto & s : sources) {
                for (int e : want) if (s.local[e] >= 0) s.banked.push_back(e);
                if (s.banked.empty()) continue;
                for (const ggml_tensor * t : part_tensors(s.w)) {
                    if (!is_host_weight(t)) {
                        throw std::runtime_error("hot experts: requested global expert " + std::to_string(s.banked.front()) +
                                " is already device-resident in " + t->name);
                    }
                    ok = ok && t->ne[3] == 1 && (size_t) t->nb[2] * (size_t) t->ne[2] == ggml_nbytes(t);
                }
            }
            if (il < model.default_layer_device.size() && model.default_layer_device[il] >= 0) {
                const auto * buft = model.default_buffer_type_offload(model.default_layer_device[il]);
                for (size_t i = 0; i < bufts.size(); ++i) if (bufts[i] == buft && !ggml_backend_is_cpu(backends[i])) device = (int) i;
            }
            // LLAMA_HOT_EXPERTS_DEVICE=<backend name> (e.g. Vulkan0): the banks go to that device, not the layer's;
            // a third column on the layer's line names the device for that layer only
            const auto dit = hot_dev.find((int) il);
            const char * env_dev = getenv("LLAMA_HOT_EXPERTS_DEVICE");
            if (const char * want_dev = dit != hot_dev.end() ? dit->second.c_str() : env_dev; want_dev && *want_dev) {
                device = -1;
                for (size_t i = 0; i < backends.size(); ++i) if (strcmp(ggml_backend_name(backends[i]), want_dev) == 0) device = (int) i;
                if (device < 0) throw std::runtime_error(std::string("hot experts: LLAMA_HOT_EXPERTS_DEVICE=") + want_dev + " names no backend");
            }
            if (device < 0) {
                for (size_t i = 0; i < backends.size() && device < 0; ++i) if (!ggml_backend_is_cpu(backends[i])) device = (int) i;
            }
            if (!ok) {
                LLAMA_LOG_WARN("llama_hot_experts: layer %zu experts are not all host-resident and expert-contiguous; no GPU bank\n", il);
                want.clear();
            } else if (device < 0) {
                throw std::runtime_error("hot experts: no GPU backend");
            } else {
                for (const auto & s : sources) {
                    if (s.banked.empty()) continue;
                    for (const ggml_tensor * t : part_tensors(s.w)) {
                        if (want.empty() || bank_type(device, t) != GGML_TYPE_COUNT) continue;
                        LLAMA_LOG_WARN("llama_hot_experts: layer %zu: %s is %s, which %s cannot run; no GPU bank\n",
                                il, t->name, ggml_type_name(t->type), ggml_backend_name(backends[device]));
                        want.clear();
                    }
                }
            }
        }

        auto & L = p.layers[il];
        for (const auto & s : sources) {
            const int b = backend_of(s.w.down);
            const auto buft = ggml_backend_is_cpu(backends[b]) ? ggml_backend_cpu_buffer_type() : ggml_backend_buffer_get_type(s.w.down->buffer);
            if (is_split) {
                part q = s.w;
                q.backend = backends[b];
                q.map = new_map(b, buft, s.local, "part", (int) il, s.w.tag);
                L.large.push_back(q);
            }
            if (!want.empty()) {
                std::vector<int32_t> v = s.local;
                for (int e : s.banked) v[e] = -1;
                // every expert of this source is banked: no rest path (it would only add a host split of zero rows)
                if (std::all_of(v.begin(), v.end(), [](int32_t x) { return x < 0; })) continue;
                part q = s.w;
                q.backend = backends[b];
                q.map = new_map(b, buft, std::move(v), "hot.rest", (int) il, s.w.tag);
                L.small.push_back(q);
            }
        }
        if (!want.empty()) {
            for (const auto & s : sources) {
                if (s.banked.empty()) continue;
                const int K = (int) s.banked.size();
                part q;
                q.backend = backends[device];
                q.bank = true;
                q.tag = s.w.tag;
                std::vector<int32_t> slices;
                for (int e : s.banked) slices.push_back(s.local[e]);
                for (ggml_tensor * const * pt : { &s.w.up, &s.w.gate, &s.w.up_gate, &s.w.down }) {
                    const ggml_tensor * src = *pt;
                    if (!src) continue;
                    // a repacked source keeps the base row size, so an un-repacked bank has the same expert stride
                    const ggml_type type = bank_type(device, src);
                    const std::string name = (std::string("hot.") + src->name).substr(0, GGML_MAX_NAME - 1);
                    ggml_tensor * bank = shared_bank(name, type, src, slices, bufts[device]);
                    if (bank) {
                        ++n_shared;
                        shared_bytes += ggml_nbytes(bank);
                    } else {
                        bank = ggml_new_tensor_3d(new_ctx(device, bufts[device], true), type, src->ne[0], src->ne[1], K);
                        if (bank->nb[2] != src->nb[2]) throw std::runtime_error("hot experts: bank layout differs from the source tensor");
                        ggml_set_name(bank, name.c_str());
                        bank_fills.push_back({ bank, src, slices });
                    }
                    if (pt == &s.w.up) q.up = bank; else if (pt == &s.w.gate) q.gate = bank; else if (pt == &s.w.up_gate) q.up_gate = bank; else q.down = bank;
                }
                std::vector<int32_t> hm(n_exp, -1);
                for (int k = 0; k < K; ++k) hm[s.banked[k]] = k;
                q.map = new_map(device, bufts[device], std::move(hm), "hot", (int) il, s.w.tag);
                L.small.push_back(q);
                n_hot += K;
            }
            ++n_bank_layers;
        }
        if (L.small.empty()) L.small = L.large;
        L.active = true;
        n_split += is_split;
    }
    if (n_split == 0 && n_bank_layers == 0) {
        LLAMA_LOG_WARN("llama_hot_experts: %s selects no eligible layer; hot experts inactive\n", path ? path : "(no file)");
        return nullptr;
    }

    size_t dev_bytes = 0;
    std::vector<ggml_backend_buffer_ptr> bank_bufs;
    for (auto * of : { &ctx_of, &bank_ctx_of }) {
        for (auto & [b, c] : *of) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft_of[b]);
            if (!buf) throw std::runtime_error(ggml_backend_is_cpu(backends[b]) ? "hot experts: host map allocation failed" :
                    "hot experts: GPU bank allocation failed (lower the budget)");
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            if (!ggml_backend_is_cpu(backends[b])) dev_bytes += ggml_backend_buffer_get_size(buf);
            (of == &bank_ctx_of ? bank_bufs : p.bufs).emplace_back(buf);
        }
    }
    size_t n_unrepacked = 0;
    std::vector<uint8_t> staging;
    for (const auto & f : bank_fills) {
        const size_t nb2 = f.src->nb[2];
        const auto & ex = f.slices;
        if (f.bank->type != f.src->type) {
            staging.resize(nb2);
            for (size_t i = 0; i < ex.size(); ++i) {
                if (!iqk_unrepack(f.src->type, f.src->ne[1], f.src->ne[0], (const char *) f.src->data + (size_t) ex[i] * nb2, staging.data())) {
                    throw std::runtime_error(std::string("hot experts: cannot un-repack ") + f.src->name);
                }
                ggml_backend_tensor_set(f.bank, staging.data(), i * nb2, nb2);
            }
            n_unrepacked += ex.size();
            continue;
        }
        for (size_t i = 0; i < ex.size();) {
            size_t n = 1;
            while (i + n < ex.size() && ex[i + n] == ex[i] + (int) n) ++n;
            ggml_backend_tensor_set(f.bank, (const char *) f.src->data + (size_t) ex[i] * nb2, i * nb2, n * nb2);
            i += n;
        }
    }
    for (const auto & f : map_fills) ggml_backend_tensor_set(f.map, f.values.data(), 0, f.values.size() * sizeof(int32_t));

    // filled banks join the model's shared set (created on first use); this context keeps the set alive
    if (!bank_ctxs.empty()) {
        if (!banks) banks = std::make_shared<llama_hot_banks>();
        for (const auto & f : bank_fills) banks->by_name[ggml_get_name(f.bank)] = { f.bank, f.slices };
        for (auto & c : bank_ctxs) banks->ctxs.push_back(std::move(c));
        for (auto & b : bank_bufs) banks->bufs.push_back(std::move(b));
        model.hot_banks = banks;
    }
    if (n_hot) p.banks = banks;

    if (n_split) {
        LLAMA_LOG_INFO("llama_hot_experts: %zu layers with split expert tensors (every batch runs one chain per partition)\n", n_split);
    }
    if (n_bank_layers) {
        LLAMA_LOG_INFO("llama_hot_experts: %s: %zu hot experts in %zu layers, %.2f MiB on GPU, banks for batches <= %lld tokens\n",
                path, n_hot, n_bank_layers, dev_bytes / 1048576.0, (long long) p.max_batch);
        if (n_shared) {
            LLAMA_LOG_INFO("llama_hot_experts: %zu bank tensors (%.2f MiB) shared with another context of this model, not allocated again\n",
                    n_shared, shared_bytes / 1048576.0);
        }
        if (n_unrepacked) {
            LLAMA_LOG_INFO("llama_hot_experts: %zu expert tensor slices un-repacked from run-time repacked (-rtr) host tensors\n", n_unrepacked);
        }
    }
    return res;
}

const std::vector<llama_hot_experts::part> * llama_hot_experts::layer(int il, int64_t n_tokens) const {
    if (il < 0 || il >= (int) pimpl->layers.size() || !pimpl->layers[il].active || n_tokens <= 0) return nullptr;
    const auto & L = pimpl->layers[il];
    const auto & v = n_tokens <= pimpl->max_batch ? L.small : L.large;
    return v.empty() ? nullptr : &v;
}
