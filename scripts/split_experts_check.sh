#!/usr/bin/env bash
# Greedy token-for-token check of a split GGUF against its source through llama-server, at two prefill batch sizes.
# usage: split_experts_check.sh BIN_DIR ORIG.gguf SPLIT.gguf PROMPTS.json OUTDIR [extra server args...]
# env: UBS="1 512" (ubatch sizes), T threads (12), NPRED (32), PORT (8790), CTX (16384)
set -u
B=$1 ORIG=$2 SPLIT=$3 PROMPTS=$4 OUT=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd); PORT=${PORT:-8790}; mkdir -p "$OUT"; rc=0
for ub in ${UBS:-1 512}; do
  for m in orig split; do
    M=$ORIG; [ $m = split ] && M=$SPLIT
    log=$OUT/$m-ub$ub.server.log
    $B/llama-server -m "$M" -c ${CTX:-16384} -ub $ub -b $((ub < 2048 ? 2048 : ub)) -t ${T:-12} -np 1 \
      --host 127.0.0.1 --port $PORT "$@" > "$log" 2>&1 &
    spid=$!
    timeout 900 bash -c "until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 2; kill -0 $spid || exit 1; done" \
      || { echo "server $m ub=$ub failed"; tail -20 "$log"; kill $spid 2>/dev/null; exit 1; }
    PORT=$PORT python3 -I "$HERE/split_experts_greedy.py" "$PROMPTS" "$OUT/$m-ub$ub.jsonl" --npredict ${NPRED:-32} || rc=1
    kill $spid; wait $spid 2>/dev/null
    grep -a 'llama_hot_experts' "$log" | head -3
  done
  echo "=== ubatch $ub: split vs orig"
  python3 -I "$HERE/split_experts_greedy.py" --compare "$OUT/orig-ub$ub.jsonl" "$OUT/split-ub$ub.jsonl" || rc=1
done
echo "check rc=$rc"; exit $rc
