#pragma once

// Per-expert weight placement for CPU-resident routed experts.
//
// 1. NUMA placement (env LLAMA_EXPERT_PLACEMENT=/path.tsv, needs --no-mmap):
//    each expert of a 3D expert tensor [ne0, ne1, n_expert] is a contiguous byte slice
//    (nb[2] bytes). The placement file binds slices to NUMA nodes with mbind(2) before
//    the tensor data is read, so every page faults directly onto its target node
//    (MPOL_MF_MOVE also migrates pages that are already present).
//
// 2. Static GPU hot-expert set (env LLAMA_HOT_EXPERTS=/path.tsv):
//    copies of the listed experts live in a per-layer bank on the GPU. For small
//    batches (decode) the MoE layer runs two paths: the GPU computes the routed
//    experts that are in the bank, the CPU computes the rest from the full tensors,
//    and the two outputs are added. Prefill keeps the single CPU path.

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct ggml_tensor;

void llama_expert_numa_apply(const llama_model & model, bool use_mmap);
void llama_expert_numa_report(const llama_model & model);

class llama_hot_experts {
public:
    struct layer_view {
        ggml_tensor  * hot_map  = nullptr; // I32 [1, n_expert] on the GPU: bank slot or -1
        ggml_tensor  * cold_map = nullptr; // I32 [1, n_expert] on the CPU: expert id or -1
        ggml_backend_t backend  = nullptr; // GPU backend that owns the bank
        ggml_backend_t cpu      = nullptr;
    };

    // nullptr when LLAMA_HOT_EXPERTS is unset; throws on a bad file or a failed allocation
    static std::unique_ptr<llama_hot_experts> create(const llama_model & model,
            const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts);

    ~llama_hot_experts();

    bool layer(int il, int64_t n_tokens, layer_view & out) const;
    ggml_tensor * bank(const ggml_tensor * w) const;

    struct impl;
private:
    std::unique_ptr<impl> pimpl;
};
