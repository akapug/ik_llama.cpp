# Patches in this fork

Research patches on top of ik_llama.cpp for running large MoE models (DeepSeek V4.1 Flash, GLM-5.3-Flash, Qwen3 MoE)
on old multi-socket CPUs plus small or mixed GPUs. Nothing here has been proposed upstream. Every patch is behind a
runtime switch that is off by default, with two exceptions: `GGML_SCHED_INPLACE_FOLLOW` is on (a correctness fix), and
`GGML_VK_MMVID_MAX_TOKENS` is 8 (set it to 0 to restore upstream's tiled Vulkan `mul_mat_id` path).

AI assistance: most of this code was written with AI coding agents (Claude Opus 5.5 and GPT-6.1 Sol via Codex), then reviewed,
tested and measured by us on our own hardware.

## Branches

Each branch contains the one before it.

| branch | adds |
|---|---|
| `mtp-draft-vocab` | NUMA row placement of experts, reduced MTP draft heads, and the earlier research stack listed under "Older patches" |
| `expert-placement-dense-rows` | placement rules that name any tensor (`rows:0..N` for attention, dense FFN, shared experts, output head) |
| `no-host-weights` | `LLAMA_NO_HOST_WEIGHTS=1` / `--no-host`: CPU-side weights are never pinned |
| `sched-inplace-follow` | scheduler fix that makes DeepSeek V4 attention correct on GPUs that cannot run its flash attention (sm_61) |
| `mixed-cuda-vulkan-expert-banks` | CUDA and Vulkan in one process, hot expert banks on a Vulkan device (MI25), MXFP4 on Vulkan, staged Vulkan I/O |

## Test hardware

- Box A: 2x EPYC 7601 (8 NUMA nodes, DDR4), GTX 1080 Ti 11 GB (sm_61), Radeon Instinct MI25 16 GB (Vega10, RADV).
- Box B: 2x Xeon Cascade Lake with DRAM plus Optane PMem as a CPU-less NUMA node, Quadro RTX 6000 24 GB (sm_75).

Unless noted, a "V4.1" figure is DeepSeek V4.1 Flash with MXFP4 routed experts and Q8_0 non-experts. The runs are greedy,
use 3 coding-agent prompts x 300 tokens, and load once per arm. `n4` = speculative decode with a 4-token draft, and `n0` =
no draft. The values are tok/s. Run-to-run noise in one session is about 3 %. These are single-session research numbers,
not a benchmark suite.

## Placement: experts and dense weights by row slice

**`rows:A,B,..` expert placement** (`LLAMA_EXPERT_PLACEMENT=<file>`, with `GGML_MOE_GROUPS=<n>`; default off). Each expert's
row slices are bound to NUMA nodes in order. The MoE thread groups are socket-ordered (use `-cr` in node order), so each
group reads local memory.
- Box A, Qwen3-30B-A3B Q4_K_S, CPU only, T64: interleave 19.26, rows over 8 nodes 21.94 (+14 %). The greedy text is the
  same.
- Box A, V4.1 (hot MXFP4 / cold Q2_K split store), CPU only: +27 % n4 and +8 % n0 over a plain mmap layout.
- Box B, Qwen3.6-35B-A3B with dense weights on the GPU: +3 %. A small model that is mostly on the GPU gains little.
- `LLAMA_EXPERT_PLACEMENT_VERIFY` prints the placement per node. Check it: with pinned host buffers (CUDA builds that do
  not set `GGML_CUDA_NO_PINNED=1`), `mbind` cannot move pages, and the placement can silently miss its target.

**Dense rows** (`expert-placement-dense-rows`). A placement rule can name any host weight, by tensor name or by
`re:<regex>`. The experts field `*` means one block. `/G` means G row-split blocks, for example V4's grouped
`attn_output_a` with `/8`. So `rows:0..7` puts slice k of attention, the shared experts and the output head on node k,
where node k's threads read it in the static dense row split. The unit test is `tests/test-expert-place.cpp`.
- Box A, Qwen3-30B-A3B, T64: 21.26 -> 23.81 (+11 %).
- Box A, V4.1, CPU only, T64: n4 6.45 -> 7.02 (+9 %), n0 4.42 -> 5.16 (+17 %). The n4 texts are identical.
- Known limit: a tensor whose rows do not split evenly over the threads is read partly remote. One example is the
  shared-expert gate/up, which is about 80 % local.

## GPU offload fixes

**No host weights** (`LLAMA_NO_HOST_WEIGHTS=1` or `--no-host`; default off). CPU-side weights take the plain CPU buffer
type and are never pinned. The pinned budget then holds only the compute, output and staging buffers, and NUMA placement
can move weight pages. When a GPU host buffer falls back to pageable memory, a WARN now names the NVRM Xid 31 it can
cause. The test is `tests/test-no-host-weights.cpp`.
- Box A, V4.1, dense and shared experts on the 1080 Ti: 0 Xid (before, Xid 31). n0 5.60 against 5.16 CPU-only, n4 6.59
  against 7.02. The draft loses here because of per-layer hops on the 5-token verify batch.

**Scheduler inplace-follow** (`GGML_SCHED_INPLACE_FOLLOW`, **default on**, `=0` restores the old behaviour). An inplace op
now runs on the backend that owns its memory. On sm_61, V4's flash attention (D=512 with sinks) is not supported on CUDA,
so it runs on the CPU. The inverse RoPE that follows is an inplace view of that CPU tensor, but it was pinned to CUDA0. The
CUDA kernel wrote into the CPU buffer after the next CUDA split had already copied the stale data. The result was garbage
text on V4/V4.1 whenever attention was on the GPU. With `=0` the test case reproduces an Xid 31 write fault. The test is
`tests/test-cuda-grouped-mm.cpp`.
- Box A, V4.1, attention, KV, dense, shared experts and head on the 1080 Ti, experts on the CPU (rows:0..7): the text
  is correct. n0 6.97 against 5.16 CPU-only (+35 %), and prefill 48.2 against 36.2 (+33 %). Peak VRAM 9.6 GiB.
- Any GPU that cannot run a model's flash attention can hit the same bug. It is not specific to Pascal.

## Mixed CUDA + Vulkan expert banks (`mixed-cuda-vulkan-expert-banks`)

Goal: put a second, different GPU (an MI25 on Vulkan) next to a CUDA card and give it the hottest routed experts.

| switch | default | does |
|---|---|---|
| `LLAMA_EXTRA_VULKAN=1` | off | in a CUDA+Vulkan build, adds the Vulkan devices after CUDA (layers stay on CUDA; `GGML_VK_VISIBLE_DEVICES` picks the card) |
| `LLAMA_HOT_EXPERTS=<file>` | off | static hot-expert set placed in GPU banks (banks belong to the model and are shared by its contexts) |
| `LLAMA_HOT_EXPERTS_DEVICE=<backend>` | layer device | puts the hot banks on that backend, e.g. `Vulkan0` |
| heat-file third column | none | `blk.N <expert ids> <backend>` puts that layer's bank on that backend (over `LLAMA_HOT_EXPERTS_DEVICE`); a layer whose every expert is banked gets no host rest split |
| `GGML_VK_OFFLOAD_MIN_BATCH` | 32 | keeps host-resident experts from being copied to Vulkan for prefill |
| `LLAMA_HOT_BANK_FIRST=1` | off | issues the bank split before the CPU cold-expert split (every partition output is expanded in graph order, side work first) |
| `LLAMA_HOT_COMBINE=<backend>` | bank backend | runs the partition sum there |
| `GGML_VK_ASYNC_GRAPH=1` | off | the Vulkan graph returns after its last submit; the fence wait moves to synchronize |
| `GGML_VK_STAGED_INPUTS=1` (`_MIB`) | off | split inputs through one host-visible staging buffer, plus a staged readback of the last node |
| `GGML_VK_MMVID_MAX_TOKENS` | 8 (0 = off) | Vulkan `mul_mat_id` batches up to N tokens use the mat-vec kernel per token (affects every Vulkan MoE user) |

Also on this branch: an MXFP4 port to Vulkan, lowering of the fork's fused MoE ops (`MOE_FUSED_UP_GATE`,
`MUL_MULTI_ADD`) for Vulkan with one reused device buffer per backend, bank planning within device allocation and
descriptor limits, and device checks in `test-backend-ops` (`--hotbank-cpu`, `--hotbank-required <backend>`,
`--hotbank-v41 <backend>`, `--hotbank-perf <backend>`).

Measured:
- Correctness: on the MI25 the device checks pass 35 of 36. The one failure is the Vulkan `ARGSORT` over 384 experts
  (NMSE 0.554), so routing must stay off the Vulkan device. CPU controls pass 36 of 36. On V4.1 with 799 hot experts
  (14 GiB) on the MI25, the greedy text is identical to the CPU-expert arm.
- Speed on V4.1: **the MI25 banks are correct but slower.** V4.1 on Box A gives n4 2.66 and n0 4.08 with the banks,
  against 7.13 and 6.97 without them. With the MI25 clocks forced high, `GGML_VK_MMVID_MAX_TOKENS=8` and a bank-first
  order that really reorders the graph, it is n4 6.19 and n0 5.61: still behind. A per-op profile of one decode graph
  shows why. The banks cover 31.6 % of the routed traffic and save about 13.5 ms of CPU expert time. The extra device
  hops (Vulkan splits, the bank down split, a larger CUDA combine) add about 40 ms. On V4.1 the lever is bound by hop
  latency, so we stopped it there.
- Before the bank-first fix, `LLAMA_HOT_BANK_FIRST=1` reordered nothing. The bank output was not expanded into the
  graph and was pulled in only by the final sum, so its device split still ran after the CPU split.
- Fixed-cost cures, Qwen3-30B-A3B smoke, decode ms per graph: banks 141 -> 107 with async graph, bank-first and combine on
  CUDA0. CPU experts alone take 86 ms. Staged inputs cut the Vulkan split's copy-in from 4.3 to 3.0 ms. Staged readback
  cuts the next CUDA split's copy-in from 14.2 to 3.5 ms per graph.
- `LLAMA_HOT_COMBINE=CUDA0` adds about 1.7 GiB of CUDA compute buffer. On V4.1 with an 11 GB card that runs out of memory.
- `GGML_VK_MMVID_MAX_TOKENS`: on the MI25, the tiled `matmul_id` path took about 11 ms for a 5-token V4.1 bank chain,
  against 0.42 ms at 1 token (`--hotbank-perf`). The end-to-end effect is not measured yet.
- Box B with the same stack, CUDA banks only (RTX 6000, Vulkan switches off, rows placement, no-host weights,
  inplace-follow), V4.1: n4 11.88 against 10.58 for the previous package (+12 %), n0 9.72 against 8.75 (+11 %), and
  prefill +25 %. Prompt 0 is identical; prompts 1-2 drift after 850 and 244 characters. The draft acceptance is 0.90.

### Ornith-1.5-35B-A3B on Box A: every routed expert on the two old GPUs

The banks pay off when they hold whole layers, so that no layer needs a CPU expert split. Ornith-1.5-35B-A3B Q4_K_M
(`ornith-ai/Ornith-1.5-35B-A3B-GGUF`) has 40 main MoE layers of about 0.42-0.49 GiB of experts each. A heat file
generated from the GGUF tensor sizes lists all 256 experts of every layer. It fills the MI25 with whole layers within
a 14 GiB budget (30 layers, 13.49 GiB), and the third column puts the other 10 layers on the 1080 Ti (4.67
GiB). The engine reports 10240 banked experts in 40 layers, 18.2 GiB on the two GPUs. Attention, the
dense and shared weights, KV and the head stay on the 1080 Ti (`-ngl 99 --cpu-moe`). The host copy of the experts
remains in place (row-placed over 8 NUMA nodes), and prefill uses it.

Conditions: Box A, one load per arm, `-t 64 -cr 0-63`, `-c 8192`, `-ub 1024`, f16 KV, `--no-mmap`, weights never
pinned. The workload is 3 coding-agent prompts (about 1.5-2.2k tokens) x 300 tokens, greedy, raw `/completion`.
`m3` and `m2` are the model's native MTP head drafting 3 or 2 tokens (`--spec-type mtp:n_max=3,p_min=0.75`).
`n0` = no draft. Watts are the median at the wall for the whole box.

| arm | layout | decode tok/s | prefill | wall |
|---|---|---|---|---|
| o1 | experts on the CPU (rows over 8 nodes) | n0 41.99 | 370 | 542 W |
| o2 | every expert in GPU banks: MI25 + 1080 Ti | n0 44.62 (+6 %) | 368 | 484 W |
| o5 | o1 + MTP | m3 47.34, m2 46.96 (accept 85 %) | 320 | 598 W |
| **o6** | **o2 + MTP** | **m3 59.13, m2 60.01, n0 44.59 (accept 85 %)** | 331 | 508 W |

- o6 1080 Ti memory in use: 8,315 of 11,264 MiB. The MI25 ran at its high DPM state, with junction and HBM peaks of 59 and
  57 C.
- The same o6 layout serving chat requests at the model card's sampling (0.6 / 0.95 / top-k 20, thinking budget 8192,
  `-c 32768`) decoded 45.7-53.3 tok/s at 13-21k tokens of depth, with 85.7 % MTP acceptance and 318-354 tok/s prefill.
- MTP keeps the experts of its own layer (`blk.40`) on the device, so that layer must not appear in the heat file.
  Otherwise the load aborts with "already device-resident".
- The external DFlash draft for this model is accepted at only 1-6 % in this engine (o1: 0.9 %), which points to a
  bug in the DFlash path. We have not investigated it. Use MTP.
- For comparison on the same box: gpt-oss-120b MXFP4 with experts on the CPU decodes 31.0 tok/s (34.6 with expert
  layers 0-2 on the 1080 Ti via `-ot`). The banks cannot hold gpt-oss experts yet, because the bank partitions and the
  Vulkan fused MoE op do not handle expert biases or `SWIGLU_OAI`.

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

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in arc (https://github.com/akapug/arc) + fab (https://github.com/akapug/fabric) for @akapug
