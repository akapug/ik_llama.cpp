#pragma once

// Tier-concurrent routed-expert reads (research, every knob default off; read once from the environment).
//
// A routed expert whose weight pages sit on a slow memory tier (Optane PMem onlined as a CPU-less NUMA node) costs
// several times the read time of a DRAM expert. With GGML_MOE_GROUPS every thread of a group takes chunks of the
// same expert at the same time, so all of them stall on PMem together while DRAM idles, then all read DRAM while
// PMem idles. The tier split keeps two chunk lists per group, slow and fast, and starts a few threads of each group
// on the slow list, so PMem and DRAM reads overlap; a thread whose list runs dry takes chunks from the other list.
// The chunk partition (which rows a chunk covers) is unchanged, only which thread computes a chunk, so results are
// bit-identical with the switch on or off.
//
// The prefetch keeps a DRAM ring of slow-tier experts: experts a step read from the slow tier are copied into the
// ring after the CPU graph compute that used them (while the GPU and the draft run), and the next step reads the ring
// copy instead. The predictor is the previous step's routing of the same layer (a DSpark draft is a separate MoE
// with its own expert ids, so its routes do not name the target's experts). Copies are exact, so results are
// bit-identical too.
//
//  GGML_TIER_SPLIT=S[,S1,...]   threads of group g that start on the slow list (one value: every group); 0 = off.
//                               Uses GGML_MOE_GROUPS (one group when unset).
//  GGML_TIER_STEAL=1            a thread whose list is empty takes chunks from the other list (default 1; 0 needs
//                               1 <= S < active threads of the group, else steal stays on)
//  GGML_TIER_SLOW_NODES=list    NUMA nodes that count as the slow tier (default: every node without CPUs)
//  GGML_TIER_SAMPLES=P          pages sampled per expert to find its node (default 4; slow if any sample is slow)
//  GGML_TIER_FAKE_SLOW=spec     TEST ONLY: also class as slow "experts:<ids and a-b ranges>" (every routed-expert
//                               weight) or "addr:<A>-<B>" (sampled pages with a virtual address in [A, B))
//  GGML_TIER_PREFETCH=1         the DRAM ring prefetch (independent of the split)
//  GGML_TIER_RING_MIB=M         ring size (default 1024)
//  GGML_TIER_RING_NODES=list    slot k-th part bound to the k-th node, like a rows:A,B placement (default: the NUMA
//                               nodes with CPUs, ascending; one node = no binding); "-" = no binding
//  GGML_TIER_PREFETCH_THREADS=N copy threads (default 1)
//  GGML_TIER_PREFETCH_CPUS=list CPUs the copy threads run on (default: the process mask)
//  GGML_TIER_PREFETCH_WHEN=end  "end": a step's copy requests start when its CPU graph compute ends (default);
//                               "now": at once, while the step still computes
//  GGML_TIER_STATS=N            every N CPU graph computes (and at exit) print the counters

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// one routed-expert op, filled by its thread 0 before the op's barrier
struct ggml_tier_op {
    const uint8_t      * slow;   // [n_expert] 1 = slow tier; NULL = no tier split for this op
    const char * const * src[2]; // [n_expert] per weight: where expert e is read (ring copy or the weight); NULL = weight
};

bool ggml_tier_on(void);           // any tier knob set
int  ggml_tier_split(int group);   // GGML_TIER_SPLIT of a group
bool ggml_tier_steal(void);

// thread 0 of a MUL_MAT_ID (w1 NULL) or MOE_FUSED_UP_GATE op, after the expert row counts are known
void ggml_tier_op_begin(const struct ggml_tensor * w0, const struct ggml_tensor * w1, const int64_t * counts, int n_as,
        struct ggml_tier_op * op);

// the calling thread's CPU graph compute ended (all workers joined): unpin its ring slots, start its copy requests
void ggml_tier_graph_end(void);

// counters for tests: [0] ops with a slow expert, [1] fast experts used, [2] slow experts used, [3] ring hits,
// [4] ring misses, [5] copies, [6] dropped requests
void ggml_tier_counts(uint64_t out[7]);

#ifdef __cplusplus
}
#endif
