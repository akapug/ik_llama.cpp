#include "llama-expert-place.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
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

std::vector<ggml_tensor *> layer_expert_tensors(const llama_layer & l) {
    std::vector<ggml_tensor *> out;
    for (ggml_tensor * t : { l.ffn_up_gate_exps, l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps }) {
        if (t) out.push_back(t);
    }
    return out;
}

bool expert_contiguous(const ggml_tensor * t) {
    return t->ne[3] == 1 && t->ne[2] > 1 && (size_t) t->nb[2] * (size_t) t->ne[2] == ggml_nbytes(t);
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

void llama_expert_numa_apply(const llama_model & model, bool use_mmap) {
    const char * path = getenv("LLAMA_EXPERT_PLACEMENT");
    if (!path || !*path) return;
    if (use_mmap) {
        LLAMA_LOG_WARN("%s: LLAMA_EXPERT_PLACEMENT needs --no-mmap (file-backed pages ignore mbind); placement skipped\n", __func__);
        return;
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot read LLAMA_EXPERT_PLACEMENT file ") + path);

    numa_plan plan;
#ifdef __linux__
    plan.page = (size_t) sysconf(_SC_PAGESIZE);
#endif
    const bool strict = getenv("LLAMA_EXPERT_PLACEMENT_STRICT") && atoi(getenv("LLAMA_EXPERT_PLACEMENT_STRICT"));

    struct rule { std::string key; int layer; std::string experts; int policy; int line; };
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
        rules.push_back({ f[0], parse_layer_key(f[0]), f[1], policy_index(plan, f[2]), ln });
    }
    plan.intended.assign(plan.policies.size(), 0);

    const size_t page = plan.page;
    const unsigned long maxnode = k_max_nodes;
    size_t n_tensors = 0, n_skipped = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        for (ggml_tensor * t : layer_expert_tensors(model.layers[il])) {
            std::vector<const rule *> mine;
            for (const auto & r : rules) {
                if (r.layer == (int) il || (r.layer < 0 && r.key == t->name)) mine.push_back(&r);
            }
            if (mine.empty()) continue;
            if (!is_host_weight(t) || !expert_contiguous(t)) {
                ++n_skipped;
                LLAMA_LOG_WARN("%s: %s is not a host-resident expert-contiguous tensor; skipped\n", __func__, t->name);
                continue;
            }
            const int n_exp = (int) t->ne[2];
            const size_t nb2 = t->nb[2];
            std::vector<int> pol(n_exp, -1);
            for (const rule * r : mine) {
                for (int e : parse_ids(r->experts, n_exp, std::string(path) + ":" + std::to_string(r->line))) pol[e] = r->policy;
            }
            if (std::all_of(pol.begin(), pol.end(), [](int p) { return p < 0; })) continue;
            ++n_tensors;

            const uintptr_t base = (uintptr_t) t->data;
            const uintptr_t end  = base + ggml_nbytes(t);
            const uintptr_t p0   = (base + page - 1) / page * page;   // round inward at the tensor edges:
            const uintptr_t p1   = end / page * page;                 // edge pages may hold other tensors
            if (pol.front() >= 0) plan.unplaced += p0 - base;
            if (pol.back()  >= 0) plan.unplaced += end - p1;

            // one policy per page: the expert that owns most of the page wins; ties go to the policy
            // with the lowest node number (DRAM nodes come before CPU-less PMem nodes on this platform)
            auto page_policy = [&](uintptr_t p) {
                const int e0 = (int) ((p - base) / nb2);
                const int e1 = (int) std::min<uintptr_t>((p + page - 1 - base) / nb2, n_exp - 1);
                if (e0 == e1 || pol[e0] == pol[e1]) return pol[e0];
                ++plan.straddle_pages;
                const size_t b0 = base + (size_t) (e0 + 1) * nb2 - p;
                const size_t b1 = page - b0;
                if (b0 != b1) return b0 > b1 ? pol[e0] : pol[e1];
                if (pol[e0] < 0) return pol[e1];
                if (pol[e1] < 0) return pol[e0];
                return plan.policies[pol[e0]].min_node() <= plan.policies[pol[e1]].min_node() ? pol[e0] : pol[e1];
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
        }
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

    LLAMA_LOG_INFO("%s: %s: %zu expert tensors placed in %zu ranges (%zu skipped), %zu straddle pages, %.2f MiB left at tensor edges\n",
            __func__, path, n_tensors, plan.ranges.size(), n_skipped, plan.straddle_pages, plan.unplaced / 1048576.0);
    for (size_t i = 0; i < plan.policies.size(); ++i) {
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
// Static GPU hot-expert set
//

struct llama_hot_experts::impl {
    struct layer {
        bool active = false;
        layer_view view;
    };
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> banks;
    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
    int64_t max_batch = 8;
};

llama_hot_experts::~llama_hot_experts() = default;

std::unique_ptr<llama_hot_experts> llama_hot_experts::create(const llama_model & model,
        const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts) {
    const char * path = getenv("LLAMA_HOT_EXPERTS");
    if (!path || !*path) return nullptr;
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot read LLAMA_HOT_EXPERTS file ") + path);
    if (backends.size() != bufts.size()) throw std::runtime_error("hot experts: backend metadata is inconsistent");

    ggml_backend_t cpu = nullptr;
    for (auto * b : backends) if (ggml_backend_is_cpu(b)) cpu = b;
    if (!cpu) throw std::runtime_error("hot experts: no CPU backend");

    auto res = std::unique_ptr<llama_hot_experts>(new llama_hot_experts());
    auto & p = *(res->pimpl = std::make_unique<impl>());
    p.layers.resize(model.layers.size());
    // The CUDA fast decode kernels skip negative ids (rows stay zero); larger batches take
    // paths that do not, so the two-path graph is limited to at most 8 tokens.
    if (const char * mb = getenv("LLAMA_HOT_EXPERTS_MAX_BATCH")) p.max_batch = std::clamp<int64_t>(atoi(mb), 0, 8);

    std::map<int, std::vector<int>> hot;
    std::string line;
    for (int ln = 1; std::getline(in, line); ++ln) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string key, ids;
        ss >> key >> ids;
        const int il = parse_layer_key(key);
        if (il >= (int) model.layers.size() && !ids.empty()) {
            // the file names a layer this model lacks: it was written for another model (e.g. the target, while this
            // context loads a small speculative draft that reads the same env) -> no hot set for this model
            LLAMA_LOG_WARN("llama_hot_experts: %s:%d names blk.%d but this model has %d layers; hot experts inactive for this model\n",
                path, ln, il, (int) model.layers.size());
            return nullptr;
        }
        if (il < 0 || ids.empty()) {
            throw std::runtime_error(std::string(path) + ":" + std::to_string(ln) + ": expected 'blk.N<TAB>expert,list'");
        }
        const int n_exp = (int) model.hparams.n_expert;
        const auto parsed = parse_ids(ids, INT_MAX, std::string(path) + ":" + std::to_string(ln));
        if (!parsed.empty() && *std::max_element(parsed.begin(), parsed.end()) >= n_exp) {
            // the file names an expert this model lacks: written for another model (a speculative draft with fewer
            // experts reads the same env) -> no hot set for this model, as for a missing layer above
            LLAMA_LOG_WARN("llama_hot_experts: %s:%d names expert %d but this model has %d experts; hot experts inactive for this model\n",
                path, ln, *std::max_element(parsed.begin(), parsed.end()), n_exp);
            return nullptr;
        }
        auto & v = hot[il];
        v.insert(v.end(), parsed.begin(), parsed.end());
    }

    struct pending { int il; int device; std::vector<int> experts; std::vector<ggml_tensor *> src; };
    std::vector<pending> todo;
    for (auto & [il, v] : hot) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        auto src = layer_expert_tensors(model.layers[il]);
        if (src.empty() || v.empty()) continue;
        const bool ok = std::all_of(src.begin(), src.end(), [&](const ggml_tensor * t) {
            return is_host_weight(t) && expert_contiguous(t) && t->ne[2] == src[0]->ne[2];
        });
        if (!ok) {
            LLAMA_LOG_WARN("llama_hot_experts: layer %d experts are not all host-resident and expert-contiguous; layer skipped\n", il);
            continue;
        }
        int device = -1;
        if (il < (int) model.default_layer_device.size() && model.default_layer_device[il] >= 0) {
            const auto * buft = model.default_buffer_type_offload(model.default_layer_device[il]);
            for (size_t i = 0; i < bufts.size(); ++i) if (bufts[i] == buft && !ggml_backend_is_cpu(backends[i])) device = (int) i;
        }
        if (device < 0) {
            for (size_t i = 0; i < backends.size() && device < 0; ++i) if (!ggml_backend_is_cpu(backends[i])) device = (int) i;
        }
        if (device < 0) throw std::runtime_error("hot experts: no GPU backend");
        todo.push_back({ il, device, v, src });
    }
    if (todo.empty()) {
        LLAMA_LOG_WARN("llama_hot_experts: %s selects no eligible layer; hot experts inactive\n", path);
        return nullptr;
    }

    // one metadata context per GPU device for the banks + hot maps, one host context for the cold maps
    std::map<int, ggml_context *> dev_ctx;
    auto new_ctx = [&](size_t n) {
        ggml_init_params ip = { n * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (!c) throw std::runtime_error("hot experts: ggml_init failed");
        p.ctxs.emplace_back(c);
        return c;
    };
    for (const auto & t : todo) if (!dev_ctx.count(t.device)) dev_ctx[t.device] = new_ctx(todo.size() * 5);
    ggml_context * host_ctx = new_ctx(todo.size());

    struct fill { ggml_tensor * bank; const ggml_tensor * src; const std::vector<int> * experts; };
    std::vector<fill> fills;
    size_t n_hot = 0;
    for (const auto & t : todo) {
        ggml_context * c = dev_ctx[t.device];
        const int K = (int) t.experts.size();
        for (const ggml_tensor * s : t.src) {
            ggml_tensor * b = ggml_new_tensor_3d(c, s->type, s->ne[0], s->ne[1], K);
            if (b->nb[2] != s->nb[2]) throw std::runtime_error("hot experts: bank layout differs from the source tensor");
            ggml_format_name(b, "hot.%s", s->name);
            p.banks[s] = b;
            fills.push_back({ b, s, &t.experts });
        }
        auto & lv = p.layers[t.il];
        lv.view.hot_map = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, t.src[0]->ne[2]);
        ggml_format_name(lv.view.hot_map, "hot.map-%d", t.il);
        lv.view.cold_map = ggml_new_tensor_2d(host_ctx, GGML_TYPE_I32, 1, t.src[0]->ne[2]);
        ggml_format_name(lv.view.cold_map, "hot.cold_map-%d", t.il);
        lv.view.backend = backends[t.device];
        lv.view.cpu = cpu;
        lv.active = true;
        n_hot += K;
    }

    size_t dev_bytes = 0;
    for (auto & [dev, c] : dev_ctx) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, bufts[dev]);
        if (!buf) throw std::runtime_error("hot experts: GPU bank allocation failed (lower the budget)");
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        dev_bytes += ggml_backend_buffer_get_size(buf);
        p.bufs.emplace_back(buf);
    }
    {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx, ggml_backend_cpu_buffer_type());
        if (!buf) throw std::runtime_error("hot experts: host map allocation failed");
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        p.bufs.emplace_back(buf);
    }

    for (const auto & f : fills) {
        const size_t nb2 = f.src->nb[2];
        const auto & ex = *f.experts;
        for (size_t i = 0; i < ex.size();) {
            size_t n = 1;
            while (i + n < ex.size() && ex[i + n] == ex[i] + (int) n) ++n;
            ggml_backend_tensor_set(f.bank, (const char *) f.src->data + (size_t) ex[i] * nb2, i * nb2, n * nb2);
            i += n;
        }
    }
    for (const auto & t : todo) {
        const int n_exp = (int) t.src[0]->ne[2];
        std::vector<int32_t> hm(n_exp, -1), cm(n_exp);
        for (int e = 0; e < n_exp; ++e) cm[e] = e;
        for (size_t s = 0; s < t.experts.size(); ++s) {
            hm[t.experts[s]] = (int32_t) s;
            cm[t.experts[s]] = -1;
        }
        auto & lv = p.layers[t.il].view;
        ggml_backend_tensor_set(lv.hot_map, hm.data(), 0, hm.size() * sizeof(int32_t));
        ggml_backend_tensor_set(lv.cold_map, cm.data(), 0, cm.size() * sizeof(int32_t));
    }

    LLAMA_LOG_INFO("llama_hot_experts: %s: %zu hot experts in %zu layers, %.2f MiB on GPU, two-path for batches <= %lld tokens\n",
            path, n_hot, todo.size(), dev_bytes / 1048576.0, (long long) p.max_batch);
    return res;
}

bool llama_hot_experts::layer(int il, int64_t n_tokens, layer_view & out) const {
    if (il < 0 || il >= (int) pimpl->layers.size() || !pimpl->layers[il].active) return false;
    if (n_tokens <= 0 || n_tokens > pimpl->max_batch) return false;
    out = pimpl->layers[il].view;
    return true;
}

ggml_tensor * llama_hot_experts::bank(const ggml_tensor * w) const {
    if (!w) return nullptr;
    auto it = pimpl->banks.find(w);
    return it == pimpl->banks.end() ? nullptr : it->second;
}
