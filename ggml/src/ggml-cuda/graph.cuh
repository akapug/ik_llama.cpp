#pragma once

#include "ggml.h"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op   node_op;
    ggml_type type;
    void * view_src;
    size_t view_offs;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    uint64_t uid = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;
    // GGML_CUDA_GRAPH_CACHE: last use tick (LRU) and "this exact segment cannot be captured" (props kept, no instance)
    uint64_t last_use = 0;
    bool incompatible = false;
#endif
};

// GGML_CUDA_GRAPH_CACHE (research, homelab hostcut lane): up to N captured executables per segment key (first node),
// each tied to the exact node properties it was captured with. Hybrid decode alternates shapes (speculative verify
// batches of 1..n_max+1 tokens, DSV4 compressor phases), which with one executable per key means a re-capture on
// every change and, after 4 in a row, graphs off for that segment for good.
struct ggml_cuda_graph_slot {
#ifdef USE_CUDA_GRAPH
    std::vector<std::unique_ptr<ggml_cuda_graph>> entries;
    uint64_t tick = 0;
    int hits = 0;     // decayed counts (halved every 64 calls)
    int misses = 0;
    int calls = 0;
    bool disabled = false;
#endif
};

