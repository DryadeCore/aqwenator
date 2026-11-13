#!/usr/bin/env bash
# Memory-policy lane: agentic sweep of a given aq-serve binary with extra server flags.
# Usage: tools/agentic_sweep_mem.sh <gpu> <tag> [server flags...]
#   BIN=~/aq/src-mem/target/release/aq-serve AGENTS="8 16 32" WARMUP=30 DURATION=150
# Results: ~/aq/results/agentic/mem-<tag>/agents<N>.json, server.log, mem.txt
set -euo pipefail
GPU=$1
TAG=$2
shift 2
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
BIN=${BIN:-$HOME/aq/src-mem/target/release/aq-serve}
AGENTS=${AGENTS:-"8 16 32"}
DURATION=${DURATION:-150}
WARMUP=${WARMUP:-30}
OUT=$HOME/aq/results/agentic/mem-$TAG
PORT=$((18300 + GPU))
mkdir -p "$OUT"
source ~/aq/venv-ref/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
cp "$BIN" "$OUT/aq-serve.bin"
"$OUT/aq-serve.bin" "$MODEL" --port "$PORT" --max-ctx 33792 "$@" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 300); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 1
done
echo "flags: $*" | tee "$OUT/meta.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$SERVER," | tee "$OUT/mem.txt" || true
for n in $AGENTS; do
  python ~/aq/src/tools/agentic_bench.py --url "http://localhost:$PORT" --model "$MODEL" --engine "aq-mem-$TAG" \
    --agents "$n" --system-len 4096 --task-len 1024 --tool-len 512:4096 --gen-len 256 --think-ms 500:3000 \
    --max-ctx 32768 --warmup "$WARMUP" --duration "$DURATION" --out "$OUT/agents$n.json" | tee -a "$OUT/summary.txt"
  grep -a "\[stats\]" "$OUT/server.log" | tail -1 | tee -a "$OUT/summary.txt"
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$SERVER," | tee -a "$OUT/mem.txt" || true
echo SWEEP_DONE
