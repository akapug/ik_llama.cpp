#!/usr/bin/env bash
# Build the SER (Smart Expert Reduction, -ser Kmin,t) branch and smoke-test it on a small MoE.
# Usage: ~/src/ik-ser/BUILD-AND-SMOKE.sh            (build + smoke)
#        SKIP_BUILD=1 ~/src/ik-ser/BUILD-AND-SMOKE.sh (smoke only)
#        FORCE=1 ...   run even while optane-* timing units are active (they WILL be perturbed)
# Env: SER (default 5,0.3), MODEL (default the Qwen3.6 35B-A3B Q4_K_XL), PPL_TEXT (default ~/a5-corpus.txt), PPL_CHUNKS (4).
set -u
cd "$(dirname "$(readlink -f "$0")")"
B=build-cuda-ec0
MODEL=${MODEL:-$HOME/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
SER=${SER:-5,0.3}
PPL_TEXT=${PPL_TEXT:-$HOME/a5-corpus.txt}
OUT=${OUT:-$HOME/ser-smoke}; mkdir -p "$OUT"

busy=$(systemctl --user list-units --state=active --plain --no-legend 'optane-*' 2>/dev/null | awk '{print $1}')
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "refusing: timing units active ($(echo $busy)); a compile or model run perturbs them. Re-run when idle, or FORCE=1."
  exit 2
fi

# ---- (a) configure + build (same flags as ~/src/ik_llama.cpp/build-cuda-ec0) ----
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  cmake -B $B -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DGGML_NATIVE=ON -DGGML_EXPERT_CHUNKING=OFF -DCMAKE_BUILD_TYPE=Release || exit 1
  nice -n 10 cmake --build $B --target llama-server llama-cli llama-perplexity -j 30 || exit 1
fi
CLI=$B/bin/llama-cli; PPL=$B/bin/llama-perplexity
[ -x $CLI ] && [ -x $PPL ] || { echo "missing binaries in $B/bin"; exit 1; }

# ---- (b) decode smoke: stock vs -ser $SER vs -ser 4,1 (= fixed top-4, the #239 t=1 identity) ----
# Experts on CPU, router on GPU: exercises CUDA argsort_thresh + get_rows(-1) + sum_rows/div,
# CPU moe_up_gate / mul_mat_id (zero rows for -1) and CPU mul_multi_add (skips -1).
export GGML_CUDA_NO_PINNED=1 GGML_CUDA_NO_PINNED_WEIGHTS=1
PROMPT="Explain in three short paragraphs how a hash map handles collisions."
run_cli() {  # $1 tag, rest = extra args
  local tag=$1; shift
  $CLI -m "$MODEL" -ngl 99 -ot exps=CPU -t 8 -c 2048 --temp 0 -n 64 --seed 1 -p "$PROMPT" "$@" \
    > "$OUT/cli-$tag.txt" 2> "$OUT/cli-$tag.log"
  echo "rc=$?" >> "$OUT/cli-$tag.log"
}
run_cli stock
run_cli ser -ser "$SER"
run_cli fixk4 -ser 4,1
for t in stock ser fixk4; do
  echo "=================== $t"
  grep -E "ser +=|rc=" "$OUT/cli-$t.log"
  grep -E "eval time|prompt eval time" "$OUT/cli-$t.log"
  echo "--- output:"; cat "$OUT/cli-$t.txt"; echo
  grep -qiE "\bnan\b|inf\b" "$OUT/cli-$t.txt" && echo "!!! WARNING: nan/inf text in $t output"
done
cmp -s "$OUT/cli-stock.txt" "$OUT/cli-ser.txt" && echo "NOTE: -ser $SER output identical to stock (threshold may never fire on this prompt)"
cmp -s "$OUT/cli-stock.txt" "$OUT/cli-fixk4.txt" && echo "!!! fixed-k4 output identical to stock: SER is NOT active (check the 'ser =' log line)"

# ---- (b2) prefill smoke: batch 512 offloads the CPU experts to the GPU (offload policy, only-active-experts
# upload, CUDA MMQ id helper, CUDA mul_multi_add). PPL must be finite; SER PPL slightly above stock.
if [ -f "$PPL_TEXT" ]; then
  for t in stock ser; do
    a=(); [ $t = ser ] && a=(-ser "$SER")
    $PPL -m "$MODEL" -ngl 99 -ot exps=CPU -t 8 -c 512 -b 512 -f "$PPL_TEXT" --chunks ${PPL_CHUNKS:-4} "${a[@]}" \
      > "$OUT/ppl-$t.log" 2>&1
    echo "=================== ppl $t rc=$?"; grep -E "ser +=|Final estimate|nan" "$OUT/ppl-$t.log" | head -5
  done
fi

# ---- (c) DeepSeek V4 (arch deepseek4) ----
cat <<EOT

DeepSeek V4 check (all routed layers go through llm_build_moe_ffn; the 3 hash-routed layers keep their fixed ids):
 - load log must show:  llama_init_from_model: ser = Kmin, t, max_batch N
 - t compares UNBIASED sqrt_softplus weights (V4 adds a near-constant bias of ~8-13 to scores of O(1),
   so a threshold on biased scores only fired at t~1). Top-6 weights are flat: the 6th is typically
   ~0.82-0.88 of the top (synthetic-input estimate from the real gate weights), so useful t is 0.75-0.9;
   t<=0.6 almost never drops. Expected drops per token (estimate): t=0.8 ~0.2-1, t=0.85 ~1-2, t=0.9 ~2+ (capped by Kmin).
 - use --ser-max-batch 1 with mmap first-touch arms: otherwise the warm-up prefill already runs with SER,
   touches fewer experts, and the measured decodes page-fault on the rest (P9 4,1: resident 123.6 vs
   143.4 GiB, gen0 10.46 -> gen2 12.44 tok/s). With max_batch 1 prefill/warm-up uses all 6 experts.
 - decode speed, same harness as the optane arms (same binary for every arm):
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-stock
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-k4    --override-kv deepseek4.expert_used_count=int:4   # control on THIS layout
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-4-1   -ser 4,1    --ser-max-batch 1   # should match the k4 control
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-4-085 -ser 4,0.85 --ser-max-batch 1
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-4-08  -ser 4,0.8  --ser-max-batch 1
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-5-085 -ser 5,0.85 --ser-max-batch 1
      S=$PWD/$B/bin/llama-server ~/srv-arm.sh ser-4-09  -ser 4,0.9  --ser-max-batch 1
 - quality: PPL ratio vs stock on the same text (no --ser-max-batch here, so prefill uses SER):
      $PWD/$B/bin/llama-perplexity -m <v4.gguf> -ngl 99 -ot exps=CPU -fa 1 -t 16 -c 512 -f ~/a5b-corpus.txt --chunks 20 [-ser 4,0.85]
   a ratio of 1.000002 vs stock means the threshold never fired.
 - read gen*.json text for coherence (no repeated garbage, no NaN-driven '!!!!').
EOT
