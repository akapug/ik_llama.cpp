# Patches in this fork

Research patches on top of ik_llama.cpp for running large MoE models (DeepSeek V4.1 Flash, GLM-5.3-Flash, Qwen3 MoE)
on old multi-socket CPUs plus small or mixed GPUs. Nothing here has been proposed upstream. Every patch is behind a
runtime switch that is off by default, with one exception: `GGML_SCHED_INPLACE_FOLLOW` is on (a correctness fix). Expert pre-binding
(`LLAMA_EXPERT_PREBIND`) is on whenever the off-by-default `LLAMA_EXPERT_PLACEMENT` is used with `--no-mmap`.

AI assistance: most of this code was written with AI coding agents (Claude Opus 5.5 and GPT-6.1 Sol via Codex), then reviewed,
tested and measured by us on our own hardware.

## Branches

This branch line holds the build behind our fastest DeepSeek V4.1 Flash run on Box B (12.69-12.71 tok/s). It starts
from the fork's patches up to "CUDA+Vulkan in one process" (see `PATCHES.md` on `mixed-cuda-vulkan-expert-banks` for
those and their switches); the later MI25 bank work of that branch is not in it. Each branch contains the one before it.

| branch | adds |
|---|---|
| `expert-prebind` | expert weights bound to their NUMA node before the loader reads them |
| `prebind-f32-indexer` | upstream ik_llama.cpp `15e7ecac` merged in (F32 indexer top-k, #2712) |
| `tier-split` | tier-concurrent expert reads for DRAM + PMem boxes (`GGML_TIER_*`) |

## Test hardware

- Box A: 2x EPYC 7601 (8 NUMA nodes, DDR4), GTX 1080 Ti 11 GB (sm_61), Radeon Instinct MI25 16 GB (Vega10, RADV).
- Box B: 2x Xeon Cascade Lake with DRAM plus Optane PMem as a CPU-less NUMA node, Quadro RTX 6000 24 GB (sm_75).

Unless noted, a "V4.1" figure is DeepSeek V4.1 Flash with MXFP4 routed experts and Q8_0 non-experts. The runs are greedy,
use 3 coding-agent prompts x 300 tokens, and load once per arm. `n4` = speculative decode with a 4-token draft, and `n0` =
no draft. The values are tok/s. Run-to-run noise in one session is about 3 %. These are single-session research numbers,
not a benchmark suite.

## Expert pre-binding (`expert-prebind`)

**`LLAMA_EXPERT_PREBIND`** (default on whenever `LLAMA_EXPERT_PLACEMENT` is set and the model loads with `--no-mmap`;
`=0` turns it off). Each deferred host weight is bound to its placement node before the loader reads it, so its pages
fault onto the right node and the after-load placement pass has nothing left to migrate. Box B, V4.1 split store, the
`fs1p` layout: 12.13 tok/s n4 (mean of 3 prompts) against 11.88 without it, about +2 %, with the same greedy text.
`tests/test-expert-place.cpp` checks the pages before the after-load pass, so a prebind that did nothing fails.

## Upstream F32 indexer (`prebind-f32-indexer`)

`expert-prebind` merged with upstream ik_llama.cpp `15e7ecac` (F32 indexer top-k #2712, sparse FA #2714, MMVQ for 2
rows on sm_75 #2701, MoE fusion and padding fixes #2706 #2709, new-MMA FA fixes #2698 #2699). No switch of ours. On
Box B it added +2.2-2.5 % to V4.1 decode (12.40 tok/s n4 on the `fs1p` layout). It did not change long-context recall
there: the build before it recalled the same facts, byte-identical, at about 190k tokens.

## Tier-concurrent expert reads (`tier-split`)

For a box whose routed experts sit on two memory tiers (Box B: DRAM on the two sockets plus Optane PMem as a CPU-less
NUMA node). Without it, all threads of a MoE thread group take chunks of the same expert at once, so a group stalls
together on a PMem expert while its DRAM bandwidth idles. Every switch is a runtime environment variable, off when unset.

- **`GGML_TIER_SPLIT=S[,S1]`**: per thread group, two chunk lists, slow and fast; the last `S` threads of a group start
  on the slow list. **`GGML_TIER_STEAL`** (default 1) lets a thread whose list ran dry take from the other. The chunk
  partition is unchanged, only which thread takes a chunk, so the result is bit-identical.
  The tier of each expert comes from the real page nodes (`move_pages` on `GGML_TIER_SAMPLES`, default 4, pages per
  expert at first use). Slow means a node in `GGML_TIER_SLOW_NODES` (default: every NUMA node with no CPUs).
  `GGML_TIER_FAKE_SLOW=experts:<ids>|addr:<A>-<B>` is for tests only.
- **`GGML_TIER_PREFETCH=1`**: a DRAM ring (`GGML_TIER_RING_MIB`, default 1024; `GGML_TIER_RING_NODES`,
  `GGML_TIER_RING_POLICY=preferred|bind`) of slow experts, predicted from the previous step's routing of the same
  weight and copied after a step's CPU compute (`GGML_TIER_PREFETCH_WHEN=end|now`, `GGML_TIER_PREFETCH_THREADS`,
  `GGML_TIER_PREFETCH_CPUS`).
- **`GGML_TIER_CTL=<file>`**: a line `split=S prefetch=0|1 steal=0|1`, re-read while the process runs
  (`GGML_TIER_CTL_MS`, default 250), with the counters printed before each change, so one server load can run several
  arms. `GGML_TIER_STATS=N` prints the counters every N computes.

Measured on Box B, DeepSeek V4.1 Flash 1017 split store (hot experts MXFP4 in DRAM, cold experts Q2_K over DRAM and
PMem), DSpark draft, `fs1p` layout, 3 coding prompts x 300 greedy tokens, fresh prefill per arm, one load per sweep:

| arm | n4 tok/s | n0 tok/s |
|---|---|---|
| off | 12.33 / 12.40 (two off arms) | 10.14 / 10.16 |
| `GGML_TIER_SPLIT=8` | **12.71** | 10.29 |
| `GGML_TIER_SPLIT=12` | 12.71 | 10.30 |
| `GGML_TIER_SPLIT=16` | 12.54 | 10.25 |

Split 8 is +2.5 % n4 and +1.25 % n0 over the better off arm; every arm gave the same text and draft acceptance as off.
An earlier sweep on the same build measured split 8 at 12.69 n4 / 10.29 n0. Prefetch was a **loss**: the ring hit
0.21 of slow-expert reads and its copies (about 1.5 GB/s) slowed the compute; it stays off.
`tests/test-tier-split.cpp` runs 16 arms (groups, split sizes, no steal, all slow, prefetch, tiny ring, real
`move_pages` node, fake address range, control-file toggling) and requires output bit-identical to off.

## Older patches (in every branch)

All of these are research switches, default off unless stated.
- Smart Expert Reduction for routed MoE layers re-enabled (`-ser Kmin,t`, `--ser-max-batch`).
- Per-expert NUMA binding plus a static GPU hot-expert set. Placement also works under `-rtr`, `--no-mmap` and deferred
  tables, with the PMem policy first.
- Split experts: a hot/cold expert tensor split with per-partition quant types (`scripts/split_experts*.py`, with a
  round-trip verifier and a KLD gate).
- DeepSeek V4: multi-sequence decode (`-np > 1`) with flash attention, and a per-step speculative checkpoint
  (`LLAMA_DSV4_PER_STEP_MAX_ROWS`, default 64). Test: `tests/test-dsv4-spec-ckpt.cpp`.
- Speculative decoding: a self-spec -> DFlash/DSpark chain, DSpark confidence truncation (`conf_min` / `conf_cum_min`),
  a suffix acceptance gate (`LLAMA_SUFFIX_GATE_ACC/COOL`), `LLAMA_MTP_UBATCH`, and reduced MTP draft heads
  (`LLAMA_MTP_DRAFT_VOCAB`, picked per request with `draft_vocab`). On GLM-5.3-Flash the reduced head measured +0.7 %,
  which is inside the noise.
- CUDA host path: a pageable H2D ring through pinned chunks (`GGML_CUDA_H2D_STAGE`), a pinned staging budget
  (`GGML_CUDA_PIN_STAGING`, `GGML_CUDA_PIN_MAX_MIB`), a CUDA graph cache (`GGML_CUDA_GRAPH_CACHE`), graphs on pre-Ampere
  GPUs (`GGML_CUDA_GRAPHS_PRE_AMPERE`), and a run-time offload batch (`GGML_CUDA_OFFLOAD_BATCH_FILE`).
- CPU decode fixed costs: `GGML_AFFINITY_ONCE`, `GGML_BARRIER_SPIN`, `GGML_OMP_FAST_ENTRY`, `GGML_MAIN_NO_MOE`,
  `GGML_SERIAL_SMALL`, `GGML_SCHED_OVERLAP`, `LLAMA_MOE_OVERLAP`, and `LLAMA_UNION_CAP`.
- Instruments: `GGML_OP_PROF=<file>` (per-op timing), `LLAMA_ROUTE_DUMP=<file>` (routed expert ids), `GGML_CPU_STATS`,
  `GGML_HOSTCUT_STATS`, and a per-request `ctx_checkpoints_tolerance`.

Most of these were measured only on the boxes above. Several gave no gain there and stay off. If you need a number, read
the commit message for that switch.

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in arc (https://github.com/akapug/arc) + fab (https://github.com/akapug/fab) for @akapug
