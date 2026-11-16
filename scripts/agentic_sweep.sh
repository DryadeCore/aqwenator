#!/usr/bin/env bash
# Agentic multi-turn sweep against one engine under a fixed memory budget.
# Usage: scripts/agentic_sweep.sh <vllm|aq> <model> <gpu> <tag>
#   AGENTS="4 8 16 32"   agent counts
#   MEM_GB=32            total device memory the engine may use (emulates the target GPU)
#   AQ_KV_GB=9           aqwenator KV page pool (fits MEM_GB with 9B weights + state slots)
#   DURATION=240 WARMUP=60
#   AQ_SERVE_BIN (aq engine)  AQ_HOME=~/aq  AQ_VENV=$AQ_HOME/venv-ref
# Results: $AQ_HOME/results/agentic/<tag>/agents<N>.json
set -euo pipefail
ENGINE=$1
MODEL=$2
GPU=$3
TAG=$4
AGENTS=${AGENTS:-"4 8 16 32"}
MEM_GB=${MEM_GB:-32}
DURATION=${DURATION:-240}
WARMUP=${WARMUP:-60}
MAX_CTX=33792
AQ_HOME=${AQ_HOME:-$HOME/aq}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$AQ_HOME/results/agentic/$TAG
PORT=$((18200 + GPU))
mkdir -p "$OUT"
source "${AQ_VENV:-$AQ_HOME/venv-ref}/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU

if [ "$ENGINE" = vllm ]; then
  if [ -d "$AQ_HOME/cuda-shim" ]; then
    export CUDA_HOME=$AQ_HOME/cuda-shim PATH=$PATH:$AQ_HOME/cuda-shim/bin NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
  else
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$PATH:${CUDA_HOME:-/usr/local/cuda}/bin
  fi
  if [ -n "${VLLM_UTIL:-}" ]; then
    UTIL=$VLLM_UTIL   # unified-memory devices report no memory.total
  else
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
    UTIL=$(python -c "print(round($MEM_GB * 1024 / $TOTAL, 4))")
  fi
  vllm serve "$MODEL" --port "$PORT" --max-model-len $MAX_CTX --gpu-memory-utilization "$UTIL" \
    --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 64 > "$OUT/server.log" 2>&1 &
  LABEL="vLLM $(python -c 'import vllm; print(vllm.__version__)')"
else
  "${AQ_SERVE_BIN:?set AQ_SERVE_BIN}" "$MODEL" --port "$PORT" --max-seqs 64 --max-running 32 --max-ctx $MAX_CTX \
    --kv-gb "${AQ_KV_GB:-9}" --mixed-prefill "${AQ_MIXED:-2048}" --spec-k "${AQ_SPEC_K:-0}" > "$OUT/server.log" 2>&1 &
  LABEL="aqwenator $TAG"
fi
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 900); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 1
done
echo "engine=$LABEL model=$MODEL gpu=$GPU mem_gb=$MEM_GB" | tee "$OUT/meta.txt"

for n in $AGENTS; do
  python "$HERE/agentic_bench.py" --url "http://localhost:$PORT" --model "$MODEL" --engine "$LABEL" \
    --agents "$n" --system-len 4096 --task-len 1024 --tool-len 512:4096 --gen-len 256 --think-ms 500:3000 \
    --max-ctx 32768 --warmup "$WARMUP" --duration "$DURATION" --out "$OUT/agents$n.json" | tee -a "$OUT/summary.txt"
done
echo SWEEP_DONE
