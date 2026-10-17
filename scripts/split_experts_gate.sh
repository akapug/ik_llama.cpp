#!/usr/bin/env bash
# Equivalence + KLD gate of split expert tensors on a small cut model, CPU only.
#  1. split_experts.py ORIG -> SPLIT (same types), round-trip verify (map bijection + byte-identical slices)
#  2. llama-perplexity: ORIG and SPLIT logits (saved as --kl-divergence-base files) must be byte-identical, PPL equal,
#     KLD(SPLIT || ORIG) = 0, at the prefill ubatch and at ubatch 4 (the small-batch/decode partition list)
#  3. llama-quantize SPLIT -> MIXED with per-partition types (COLD_T / WARM_T; hot stays), verify types and the
#     untouched hot slices, then KLD(MIXED || ORIG) and, for scale, KLD(ALL experts at COLD_T || ORIG)
# usage: split_experts_gate.sh BIN_DIR ORIG.gguf WORKDIR HOT_LIST [WARM_LIST]
# env: T threads (16), CHUNKS (16), CTX (512), COLD_T (iq2_ks), WARM_T (iq3_k), IMATRIX (an imatrix .dat of ORIG), KEEP, ALLCOLD (1)
set -u -o pipefail
B=$1 ORIG=$2 W=$3 HOT=$4 WARM=${5:-}
HERE=$(cd "$(dirname "$0")" && pwd)
T=${T:-16} CHUNKS=${CHUNKS:-16} CTX=${CTX:-512} COLD_T=${COLD_T:-iq2_ks} WARM_T=${WARM_T:-iq3_k}
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
  ppl "$ORIG" orig-ub$ub $ub --kl-divergence-base orig-ub$ub.kld
  ppl split.gguf split-ub$ub $ub --kl-divergence-base split-ub$ub.kld
  if cmp -s orig-ub$ub.kld split-ub$ub.kld; then echo "  ub=$ub: logit files BYTE-IDENTICAL ($(stat -c %s orig-ub$ub.kld) bytes)"
  else fail "ub=$ub: logit files differ"; cmp orig-ub$ub.kld split-ub$ub.kld | head -2; fi
  ppl split.gguf split-vs-orig-ub$ub $ub --kl-divergence-base orig-ub$ub.kld --kl-divergence
done
cmp -s orig-ub$CTX.kld orig-ub4.kld && echo "  (orig ub=$CTX vs ub=4 also byte-identical)" || echo "  (orig ub=$CTX vs ub=4 differ: ubatch changes the arithmetic, expected)"

echo "== 3. mixed types"
q=(--allow-requantize)
[ -s split.imatrix.dat ] && q+=(--imatrix split.imatrix.dat)
# KEEP: rules that pin the source type of 2D F32 tensors quantize would otherwise convert to Q8_0 (V4: hc_*_fn, *_ape)
KEEP=${KEEP:-hc_.*_fn\.=f32,_ape\.=f32}
rules="_exps_cold\.=$COLD_T,_exps_warm\.=$WARM_T,_exps_hot\.=mxfp4,$KEEP"
if [ ! -s mixed.gguf ]; then
  "$B/llama-quantize" "${q[@]}" --custom-q "$rules" split.gguf mixed.gguf Q8_0 "$T" > quant-mixed.log 2>&1 || { tail -30 quant-mixed.log; fail "quantize mixed"; }
fi
exp="hot=MXFP4,cold=${COLD_T^^}"; [ -n "$WARM" ] && exp+=",warm=${WARM_T^^}"
python3 -I "$HERE/split_experts_verify.py" "$ORIG" mixed.gguf --bytes --expect-types "$exp" > verify-mixed.log 2>&1; vr=$?
cat verify-mixed.log | head -30
[ $vr = 0 ] || echo "  (verify-mixed FAIL lines above: non-expert tensors that quantize changed are listed; experts are what matter)"
grep -c "type/bytes differ" verify-mixed.log | sed 's/^/  non-expert tensors changed by quantize: /'
ls -la split.gguf mixed.gguf | awk '{print "  " $5 " " $9}'
ppl mixed.gguf mixed-vs-orig $CTX --kl-divergence-base orig-ub$CTX.kld --kl-divergence
ppl mixed.gguf mixed-ub4-vs-orig 4 --kl-divergence-base orig-ub4.kld --kl-divergence
if [ "${ALLCOLD:-1}" = 1 ]; then
  imo=(); [ -n "${IMATRIX:-}" ] && imo=(--imatrix "$IMATRIX")
  [ -s allcold.gguf ] || "$B/llama-quantize" --allow-requantize "${imo[@]}" --custom-q "_exps\.=$COLD_T,$KEEP" "$ORIG" allcold.gguf Q8_0 "$T" > quant-allcold.log 2>&1 \
    || { tail -30 quant-allcold.log; fail "quantize allcold"; }
  ls -la allcold.gguf | awk '{print "  " $5 " " $9}'
  ppl allcold.gguf allcold-vs-orig $CTX --kl-divergence-base orig-ub$CTX.kld --kl-divergence
fi
echo "gate rc=$rc"; exit $rc
