#pragma once

// Per-expert weight placement for CPU-resident routed experts.
//
// 1. NUMA placement (env LLAMA_EXPERT_PLACEMENT=/path.tsv and/or GGML_NUMA_EXPERTS; needs --no-mmap or a model file on
//    tmpfs/hugetlbfs: an mmap'ed tmpfs copy has its planned pages faulted in and moved after the load, since mbind on
//    shared memory sets the file's own policy, while a disk-file mmap ignores mbind and is skipped):
//    each expert of a 3D expert tensor [ne0, ne1, n_expert] is a contiguous byte slice
//    (nb[2] bytes). The placement file binds slices to NUMA nodes with mbind(2) before
//    the tensor data is read, so every page faults directly onto its target node
//    (MPOL_MF_MOVE also migrates pages that are already present).
//    Node field: "N" binds, "A,B" interleaves, "rows:A,B" binds slice k of the expert's bytes (its rows,
//    in order) to the k-th node, so with GGML_MOE_GROUPS=2 and an -cr list in socket order each thread group
//    reads only its own socket's memory.
//    Rule keys: a layer ("blk.N"), an exact tensor name, or "re:<regex>" searched in tensor names. A name or regex rule
//    also reaches non-expert host weights (attention, dense FFN, shared experts, output head); for those the experts
//    field is "*" (one block) or "/G" (G equal consecutive blocks, each split by a rows policy: a grouped weight such as
//    the DeepSeek-V4 attn_output_a). With -cr in node order, the static row split of a dense matmul then gives node k's
//    threads slice k of every block.
//    GGML_NUMA_EXPERTS=<node per thread group> (ggml/src/ggml-numa-experts.h) adds the auto plan for every expert tensor
//    no rule names: expert e of layer il binds whole to the node of group owner[e] of ggml_numa_experts_partition
//    (salt il, weights GGML_NUMA_EXPERTS_WEIGHTS or the nodes' memory/CPU shares; experts LLAMA_HOT_EXPERTS banks on a
//    GPU are dealt separately so the host ones stay balanced), and the CPU kernels hand each such expert to that group.
//
// 2. Static GPU hot-expert set (env LLAMA_HOT_EXPERTS=/path.tsv):
//    copies of the listed experts live in a per-layer bank on the GPU. For small
//    batches (decode) the MoE layer runs two paths: the GPU computes the routed
//    experts that are in the bank, the CPU computes the rest from the full tensors,
//    and the two outputs are added. Prefill keeps the single CPU path.
//    The banks belong to the model: every context of the model (the target and an MTP draft context of the
//    same model) shares one allocation; the maps and backend handles are per context.
//
// 3. Split expert tensors (scripts/split_experts.py): a layer's experts live in several
//    partition tensors (e.g. hot experts MXFP4, cold experts IQ2_KS), each with its own type
//    and NUMA placement. Every batch size runs one mul_mat_id chain per partition and sums them.
//    A GPU hot set on a split model copies each listed expert from the partition that holds it.

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;
struct ggml_tensor;

// GPU bank tensors of one model, shared by every context of that model (see llama_model::hot_banks)
struct llama_hot_banks;

// Partition tags of split expert tensors: blk.N.ffn_{up,gate,down}_exps_<tag>.weight (scripts/split_experts.py).
// The tag index is what blk.N.ffn_exps_part.weight stores.
constexpr int LLAMA_EXPERT_PART_MAX = 8;
const char * llama_expert_part_tag(int tag);

void llama_expert_numa_apply(const llama_model & model, bool use_mmap);
// bind one freshly allocated host weight to its placement before the loader reads it (deferred --no-mmap loads, whose
// buffers exist only during the read); llama_expert_numa_apply still runs after the load and finds the pages placed.
// LLAMA_EXPERT_PREBIND=0 turns it off.
void llama_expert_numa_prebind(const llama_model & model, const ggml_tensor * t);

// whether a backend computes MUL_MAT_ID with weights of this type (a CPU backend always does)
bool llama_backend_runs_expert_type(ggml_backend_t backend, ggml_type type);
void llama_expert_numa_report(const llama_model & model);

// MoE expert partitions of a context. A layer's routed experts are computed as a sum of partitions: each partition
// is a set of expert tensors [ne0, ne1, K] plus a map (expert id -> index in the partition, or -1); every routed
// expert is computed in exactly one partition and the others write zero rows for it.
// Sources: split models (each layer's own partition tensors, every batch size) and the static GPU hot set
// (LLAMA_HOT_EXPERTS: GPU copies of listed experts, taken from whichever partition holds them; small batches only).
class llama_hot_experts {
public:
    struct part {
        ggml_tensor  * up      = nullptr;
        ggml_tensor  * gate    = nullptr;
        ggml_tensor  * up_gate = nullptr;
        ggml_tensor  * down    = nullptr;
        ggml_tensor  * map     = nullptr; // I32 [1, n_expert]: index in this partition or -1
        ggml_backend_t backend = nullptr; // backend that holds the map (and the weights)
        bool           bank    = false;   // GPU copy from LLAMA_HOT_EXPERTS
        const char   * tag     = "";
    };

    // nullptr when the model has no split layer and LLAMA_HOT_EXPERTS is unset (or banks are not allowed);
    // throws on a bad file or a failed allocation
    static std::unique_ptr<llama_hot_experts> create(const llama_model & model,
            const std::vector<ggml_backend_t> & backends,
            const std::vector<ggml_backend_buffer_type_t> & bufts, bool allow_banks);

    ~llama_hot_experts();

    // partitions for this layer and batch size (model partitions first, GPU banks last); nullptr = single path
    const std::vector<part> * layer(int il, int64_t n_tokens) const;

    struct impl;
private:
    std::unique_ptr<impl> pimpl;
};
