#!/usr/bin/env bash
# Equivalence + KLD gate of split expert tensors on a small cut model, CPU only.
#  1. split_experts.py ORIG -> SPLIT (same types), round-trip verify (map bijection + byte-identical slices)
#  2. llama-perplexity: ORIG and SPLIT logits (saved as --kl-divergence-base files) must be byte-identical, PPL equal,
#     KLD(SPLIT || ORIG) = 0, at the prefill ubatch and at ubatch 4 (the small-batch/decode partition list)
#  3. llama-quantize SPLIT -> MIXED with the per-partition / per-layer rules split_experts.py --type writes (hot keeps
#     its source bytes), verify every tensor type against the rules and the untouched hot slices, then
#     KLD(MIXED || ORIG) and, for scale, KLD(every expert at the cold recipe || ORIG)
# usage: split_experts_gate.sh BIN_DIR ORIG.gguf WORKDIR HOT_LIST [WARM_LIST]   (BASE_B=bin dir of the base build: unsplit regression)
# env: T threads (16), CHUNKS (16), CTX (512), TYPES ('hot=keep warm=WARM_T cold.down@0-1=q3_K cold=COLD_T'),
#      COLD_T (q2_K), WARM_T (iq3_k), IMATRIX (an imatrix .dat of ORIG), ALLCOLD (1), SKIP_EQ (0)
set -u -o pipefail
B=$1 ORIG=$2 W=$3 HOT=$4 WARM=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd)
T=${T:-16} CHUNKS=${CHUNKS:-16} CTX=${CTX:-512} COLD_T=${COLD_T:-q2_K} WARM_T=${WARM_T:-iq3_k}
mkdir -p "$W"; cd "$W" || exit 2
rc=0
fail() { echo "GATE FAIL: $*"; rc=1; }
EVAL=${EVAL:-eval.txt}
[ -s "$EVAL" ] || { echo "need $W/$EVAL (text for perplexity)"; exit 2; }

parts=(--part "hot=$HOT"); [ -n "$WARM" ] && parts+=(--part "warm=$WARM")
im=(); [ -n "${IMATRIX:-}" ] && im=(--imatrix "$IMATRIX" --imatrix-out split.imatrix.dat)
echo "== 1. split + verify"
[ -s split.gguf ] || python3 -I "$HERE/split_experts.py" "$ORIG" split.gguf "${parts[@]}" --rest cold "${im[@]}" || fail split
python3 -I "$HERE/split_experts_verify.py" "$ORIG" split.gguf --bytes || fail "verify split"

ppl() { # model tag ubatch [extra]
  local m=$1 tag=$2 ub=$3; shift 3
  "$B/llama-perplexity" -m "$m" -f "$EVAL" -c "$CTX" -b "$CTX" -ub "$ub" --chunks "$CHUNKS" -t "$T" "$@" > "ppl-$tag.log" 2>&1 \
    || { tail -20 "ppl-$tag.log"; fail "perplexity $tag"; }
  grep -a -E "Final estimate|llama_hot_experts|Mean +KLD|Maximum KLD|99.9% +KLD|Median +KLD|Same top p|RMS Δp" "ppl-$tag.log" | sed "s/^/  [$tag] /"
}
echo "== 2. equivalence (same types)"
for ub in $CTX 4; do
  [ "${SKIP_EQ:-0}" = 1 ] && [ -s orig-ub$ub.kld ] && { echo "  ub=$ub: SKIP_EQ (orig-ub$ub.kld kept)"; continue; }
  ppl "$ORIG" orig-ub$ub $ub --kl-divergence-base orig-ub$ub.kld
  ppl split.gguf split-ub$ub $ub --kl-divergence-base split-ub$ub.kld
  if cmp -s orig-ub$ub.kld split-ub$ub.kld; then echo "  ub=$ub: logit files BYTE-IDENTICAL ($(stat -c %s orig-ub$ub.kld) bytes)"
  else fail "ub=$ub: logit files differ"; cmp orig-ub$ub.kld split-ub$ub.kld | head -2; fi
  ppl split.gguf split-vs-orig-ub$ub $ub --kl-divergence-base orig-ub$ub.kld --kl-divergence
done
if [ -n "${BASE_B:-}" ]; then
  # a model without split tensors must run exactly as on the base build (BASE_B = bin dir of the branch's base)
  B0=$B; B=$BASE_B; ppl "$ORIG" base-orig-ub$CTX $CTX --kl-divergence-base base-orig-ub$CTX.kld; B=$B0
  cmp -s base-orig-ub$CTX.kld orig-ub$CTX.kld && echo "  unsplit model: base build and this build give BYTE-IDENTICAL logits" \
    || fail "unsplit model: logits differ between the base build and this build"
fi
cmp -s orig-ub$CTX.kld orig-ub4.kld && echo "  (orig ub=$CTX vs ub=4 also byte-identical)" || echo "  (orig ub=$CTX vs ub=4 differ: ubatch changes the arithmetic, expected)"

echo "== 3. mixed types: llama-quantize with the rules split_experts.py writes for TYPES"
# hot keeps the source bytes (native MXFP4); every non-expert tensor is pinned to its source type by the rules
read -r -a ts <<< "${TYPES:-hot=keep warm=$WARM_T cold.down@0-1=q3_K cold=$COLD_T}"
targs=(); for x in "${ts[@]}"; do targs+=(--type "$x"); done
python3 -I "$HERE/split_experts.py" "$ORIG" --dry-run "${parts[@]}" --rest cold "${targs[@]}" --quant-rules mixed.rules | grep -E "type|rules" || fail "rules"
q=(--allow-requantize); [ -s split.imatrix.dat ] && q+=(--imatrix split.imatrix.dat)
if [ ! -s mixed.gguf ]; then
  "$B/llama-quantize" "${q[@]}" --custom-q "$(cat mixed.rules)" split.gguf mixed.gguf Q8_0 "$T" > quant-mixed.log 2>&1 || { tail -30 quant-mixed.log; fail "quantize mixed"; }
fi
python3 -I "$HERE/split_experts_verify.py" "$ORIG" mixed.gguf --bytes --expect-rules mixed.rules || fail "verify mixed"
ls -la split.gguf mixed.gguf | awk '{print "  " $5 " " $9}'
ppl mixed.gguf mixed-vs-orig $CTX --kl-divergence-base orig-ub$CTX.kld --kl-divergence
ppl mixed.gguf mixed-ub4-vs-orig 4 --kl-divergence-base orig-ub4.kld --kl-divergence
if [ "${ALLCOLD:-1}" = 1 ]; then
  # reference: every expert at the cold recipe, unsplit (the same rules on the unsplit tensor names)
  python3 -I "$HERE/split_experts.py" "$ORIG" --dry-run --rest cold "${targs[@]}" --quant-rules allcold.rules > /dev/null || fail "allcold rules"
  sed -i 's/_exps_cold\\\./_exps\\./g' allcold.rules
  imo=(); [ -n "${IMATRIX:-}" ] && imo=(--imatrix "$IMATRIX")
  [ -s allcold.gguf ] || "$B/llama-quantize" --allow-requantize "${imo[@]}" --custom-q "$(cat allcold.rules)" "$ORIG" allcold.gguf Q8_0 "$T" > quant-allcold.log 2>&1 \
    || { tail -30 quant-allcold.log; fail "quantize allcold"; }
  ls -la allcold.gguf | awk '{print "  " $5 " " $9}'
  ppl allcold.gguf allcold-vs-orig $CTX --kl-divergence-base orig-ub$CTX.kld --kl-divergence
fi
echo "gate rc=$rc"; exit $rc
