#pragma once

// NUMA-local expert sharding (research, every knob default off; read once from the environment).
//
// Each routed expert of a CPU MoE layer lives whole on one NUMA node (the llama loader places it there, see
// LLAMA_EXPERT_PLACEMENT in src/llama-expert-place.h) and only the thread group pinned to that node computes it. A
// GGML_MOE_GROUPS + rows:A,B placement is the other NUMA-local layout: every node holds a row slice of every expert and
// every group computes its slice. Both read only local memory; expert sharding keeps each expert's up/gate output and
// down input on one node, at the cost of load balance when a step selects few experts (a group with no selected expert
// idles unless it steals).
//
// Which group computes an expert follows the NUMA node of its pages (sampled with move_pages once per weight), so any
// placement works: an expert whose sampled pages are mostly on a listed node goes to that node's group; an expert whose
// pages are spread (interleave, a rows placement) or on an unlisted node is shared and every group computes its own row
// slice of it, as with GGML_MOE_GROUPS. The chunk partition (which rows a chunk covers) is the static split's, only the
// thread that computes a chunk changes, so results are bit-identical with the switch on or off.
//
//  GGML_NUMA_EXPERTS=list          NUMA node of each thread group, in thread order (e.g. 0-7 with -cr in node order:
//                                  group g = the g-th contiguous 1/G of the threads). Turns the mode on; G = list size
//                                  (at most 8; it replaces GGML_MOE_GROUPS). A node may repeat (several groups serve it;
//                                  experts on it are split among them by id; "0,0" exercises the mode on a one-node box).
//  GGML_NUMA_EXPERTS_STEAL=0|1     a group whose own list is done takes chunks from the other groups' lists (remote
//                                  reads); default 1
//  GGML_NUMA_EXPERTS_SAMPLES=P     pages sampled per expert (default 4); an expert is owned when more than half of them
//                                  sit on one listed node
//  GGML_NUMA_EXPERTS_WEIGHTS=w,..  share of the experts each group gets in the loader's placement (one value per group);
//                                  default: per node min(share of MemTotal, share of CPUs) over the listed nodes, split
//                                  evenly among the groups that serve the node
//  GGML_NUMA_EXPERTS_FAKE=1        TEST ONLY: owners come from the id partition (salt 0), not from page nodes
//  GGML_NUMA_EXPERTS_STATS=N       every N routed-expert ops (and at exit) print the counters

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_NUMA_EXPERTS_MAX_GROUPS 8 // = GGML_MOE_MAX_GROUPS

bool ggml_numa_experts_on(void);
// number of groups G and their nodes (out may be NULL); 0 when the mode is off
int  ggml_numa_experts_groups(int * out);
bool ggml_numa_experts_steal(void);
// share of the experts per group (G values, sum 1)
void ggml_numa_experts_weights(double * out);

// Deterministic partition of n experts over G groups, shared by the loader (placement) and the FAKE owners:
// the experts with skip[e] == 0 (skip may be NULL) are dealt in id order, starting at position salt % (their count),
// by smooth weighted round-robin, so group g gets round(count * w[g] / sum w) of them (largest remainder) spread over
// the id range; the skipped experts (e.g. copies banked on a GPU) are dealt the same way afterwards. owner[e] in [0, G).
void ggml_numa_experts_partition(int n, const uint8_t * skip, const double * w, int G, int salt, int32_t * owner);

// thread 0 of a MUL_MAT_ID (w1 NULL) or MOE_FUSED_UP_GATE op, after the expert row counts are known: the owner group of
// every expert (-1 = shared), or NULL when the op is not sharded (mode off, not an expert-contiguous host weight)
const int8_t * ggml_numa_experts_op_begin(const struct ggml_tensor * w0, const struct ggml_tensor * w1,
        const int64_t * counts, int n_as);
// a thread took n chunks from another group's list
void ggml_numa_experts_count_steal(int n);

// a host buffer [base, base + size) is about to be freed: drop the owner maps of weights in it
void ggml_numa_experts_forget(const void * base, size_t size);

// counters for tests: [0] ops, [1] selected experts owned, [2] selected experts shared, [3] stolen chunks,
// [4] sum over ops of the largest group's selected-expert count, [5] weights mapped
void ggml_numa_experts_counts(uint64_t out[6]);

#ifdef __cplusplus
}
#endif
