// Synthetic memory/backends only. Exercises the real llama_hot_experts::create
// consumer. The doubles model buft host/device identity and memcpy storage, not
// a GPU kernel or the predicate under test. No device enumeration or model file.
#include "llama-expert-bank-plan.h"
#include "llama-model.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

static void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what);
}

struct memory_device {
    ggml_backend_buffer_type buft {};
    ggml_backend backend {};
    ggml_backend_t cpu;
    ggml_guid guid {};

    explicit memory_device(ggml_backend_t cpu, uint8_t identity) : cpu(cpu) {
        guid[0] = identity;
        buft.context = this;
        buft.iface.get_name = [](ggml_backend_buffer_type_t) { return "synthetic-device-storage"; };
        buft.iface.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 64; };
        buft.iface.get_max_size = [](ggml_backend_buffer_type_t) -> size_t { return SIZE_MAX; };
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t bt, size_t size) {
            auto * storage = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            ggml_backend_buffer_i iface {};
            iface.get_name = [](ggml_backend_buffer_t) { return "synthetic-device-storage"; };
            iface.free_buffer = [](ggml_backend_buffer_t b) { ggml_backend_buffer_free((ggml_backend_buffer_t) b->context); };
            iface.get_base = [](ggml_backend_buffer_t b) { return ggml_backend_buffer_get_base((ggml_backend_buffer_t) b->context); };
            iface.set_tensor = [](ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t n) {
                memcpy((char *) t->data + offset, data, n);
            };
            iface.get_tensor = [](ggml_backend_buffer_t, const ggml_tensor * t, void * data, size_t offset, size_t n) {
                memcpy(data, (char *) t->data + offset, n);
            };
            iface.clear = [](ggml_backend_buffer_t b, uint8_t value) { memset(ggml_backend_buffer_get_base(b), value, b->size); };
            return ggml_backend_buffer_init(bt, iface, storage, size);
        };
        backend.guid = &guid;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "synthetic-device"; };
        backend.iface.get_default_buffer_type = [](ggml_backend_t b) { return &((memory_device *) b->context)->buft; };
        backend.iface.supports_buft = [](ggml_backend_t b, ggml_backend_buffer_type_t bt) {
            return bt == &((memory_device *) b->context)->buft;
        };
        backend.iface.supports_op = [](ggml_backend_t b, const ggml_tensor * t) {
            return ggml_backend_supports_op(((memory_device *) b->context)->cpu, t);
        };
    }
};

static void planner_controls() {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    size_t checked = 0;
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_MXFP4}) {
        auto * up = ggml_new_tensor_3d(ctx.get(), type, 5120, 2304, 384);
        auto * down = ggml_new_tensor_3d(ctx.get(), type, 2304, 5120, 384);
        const size_t expected = type == GGML_TYPE_Q8_0 ? 4812963840ULL : 2406481920ULL;
        require(ggml_nbytes(up) == expected && ggml_nbytes(down) == expected, "actual V4.1 packed geometry differs");
        // Synthetic limits, intentionally unequal and not an assumed MI25 value.
        const llama_expert_bank_limits limits {up->nb[2] * 70 + 127, up->nb[2] * 65, 256};
        const auto plan = llama_expert_bank_plan({up, down}, 384, limits, 1);
        require(plan.size() == 6, "planner must use the stricter descriptor limit and actual tag budget");
        int64_t next = 0;
        for (const auto & bank : plan) {
            require(bank.first == next && bank.count > 0, "bank coverage has a hole or duplicate");
            next += bank.count;
            for (auto * source : {up, down}) {
                ggml_tensor view = *source;
                view.ne[2] = bank.count;
                view.nb[3] = view.nb[2] * bank.count;
                const size_t bytes = ggml_nbytes(&view);
                const size_t padding = (limits.alignment - bytes % limits.alignment) % limits.alignment;
                require(bytes <= limits.descriptor && bytes + padding <= limits.allocation, "illegal bank extent");
                ++checked;
            }
        }
        require(next == 384, "global expert 383 must be covered");
        for (const auto limits_bad : {llama_expert_bank_limits{up->nb[2] - 1, SIZE_MAX, 1},
                                      llama_expert_bank_limits{SIZE_MAX, up->nb[2] * 47, 1},
                                      llama_expert_bank_limits{SIZE_MAX, SIZE_MAX, 0}}) {
            bool refused = false;
            try { (void) llama_expert_bank_plan({up}, 384, limits_bad); }
            catch (const std::runtime_error &) { refused = true; }
            require(refused, "insufficient limits/alignment/part count must refuse");
            ++checked;
        }
    }
    printf("bank planner: %zu real-geometry extent/refusal checks PASS (synthetic limits, no device measurement)\n", checked);
}

static void mixed_controls(const char * path, bool request_device) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    memory_device hot(cpu.get(), 101), whole(cpu.get(), 102);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> host_ctx(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> dev_ctx(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
    llama_model model;
    model.hparams.n_expert = 384;
    model.layers.resize(1);
    auto & layer = model.layers[0];
    for (int tag = 0; tag < 2; ++tag) {
        llama_layer::expert_part p {};
        p.tag = tag;
        auto * ctx = tag ? dev_ctx.get() : host_ctx.get();
        const int count = tag ? 64 : 320;
        p.up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, count);
        p.gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, count);
        p.down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, count);
        for (auto * t : {p.up, p.gate, p.down}) ggml_format_name(t, "synthetic-part-%d", tag);
        layer.ffn_exps_parts.push_back(p);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> host_buf(
        ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), ggml_backend_cpu_buffer_type()), ggml_backend_buffer_free);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> dev_buf(
        ggml_backend_alloc_ctx_tensors_from_buft(dev_ctx.get(), &whole.buft), ggml_backend_buffer_free);
    require(host_buf && dev_buf, "synthetic allocation failed");
    ggml_backend_buffer_set_usage(host_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(dev_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    layer.ffn_exps_part_map.resize(768);
    for (int e = 0; e < 384; ++e) {
        layer.ffn_exps_part_map[e] = e >= 320;
        layer.ffn_exps_part_map[384 + e] = e >= 320 ? e - 320 : e;
    }
    for (const auto & p : layer.ffn_exps_parts) {
        for (auto * t : {p.up, p.gate, p.down}) {
            std::vector<float> row(32 * 32);
            for (int e = 0; e < t->ne[2]; ++e) {
                std::fill(row.begin(), row.end(), float(e + 1));
                ggml_backend_tensor_set(t, row.data(), e * t->nb[2], t->nb[2]);
            }
        }
    }
    // Producer boundary matches create's file grammar: layer key, global IDs.
    { std::ofstream file(path); require(bool(file), "cannot create synthetic heat file"); file << "blk.0\t" << (request_device ? "383" : "0,319") << "\n"; }
    setenv("LLAMA_HOT_EXPERTS", path, 1);
    unsetenv("LLAMA_HOT_EXPERTS_MAX_BATCH");
    bool refused = false;
    try {
        auto parts = llama_hot_experts::create(model, {&hot.backend, &whole.backend, cpu.get()},
                {&hot.buft, &whole.buft, ggml_backend_cpu_buffer_type()}, true);
        require(!request_device, "requested device-resident expert did not refuse");
        require(bool(parts), "real consumer produced no partition context");
        const auto * small = parts->layer(0, 1);
        const auto * large = parts->layer(0, 9);
        require(small && small->size() == 3 && large && large->size() == 2, "mixed source lost its hot bank or a whole partition");
        std::vector<int> owners(384);
        size_t banks = 0;
        for (const auto & p : *small) {
            std::vector<int32_t> map(384);
            ggml_backend_tensor_get(p.map, map.data(), 0, map.size() * sizeof(int32_t));
            for (int e = 0; e < 384; ++e) {
                const int expected = p.bank ? (e == 0 ? 0 : e == 319 ? 1 : -1) :
                    p.backend == &whole.backend ? (e >= 320 ? e - 320 : -1) :
                    (e < 320 && e != 0 && e != 319 ? e : -1);
                require(map[e] == expected, "global/local map or noncopied ID changed");
                owners[e] += map[e] >= 0;
            }
            if (p.bank) {
                ++banks;
                require(p.backend == &hot.backend && p.up->ne[2] == 2, "wrong bank backend or copied count");
                for (auto * t : {p.up, p.gate, p.down}) {
                    std::vector<float> bytes(32 * 32 * 2);
                    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size() * sizeof(float));
                    for (size_t i = 0; i < bytes.size(); ++i) require(bytes[i] == (i < 1024 ? 1.0f : 320.0f), "real bank copy selected the wrong source slice");
                }
            }
        }
        require(banks == 1 && std::all_of(owners.begin(), owners.end(), [](int n) { return n == 1; }), "experts must route exactly once");
    } catch (const std::runtime_error & e) {
        if (!request_device || std::string(e.what()).find("already device-resident") == std::string::npos) throw;
        refused = true;
    }
    require(!request_device || refused, "nonhost request was not explicitly refused");
    unsetenv("LLAMA_HOT_EXPERTS");
    printf("real hot-bank consumer: %s PASS\n", request_device ? "requested nonhost ID refused" : "mixed-source copy, preserved maps, exactly-once global IDs");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: test-expert-banks mixed|nonhost SYNTHETIC_HEAT_PATH");
        planner_controls();
        require(!strcmp(argv[1], "mixed") || !strcmp(argv[1], "nonhost"), "unknown control");
        mixed_controls(argv[2], !strcmp(argv[1], "nonhost"));
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "required expert-bank control FAIL: %s\n", e.what());
        return 1;
    }
}
