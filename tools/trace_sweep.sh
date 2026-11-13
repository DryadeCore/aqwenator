#!/usr/bin/env bash
# Real-trace replay sweep (tools/trace_bench.py) against one engine.
# Usage: tools/trace_sweep.sh <vllm|aq> <model> <traces.json> <tag>
#   AGENTS="4 8 16"  DURATION=240 WARMUP=60
#   aq:   AQ_SERVE_BIN, AQ_KV_GB=48, AQ_ARGS="extra aq-serve flags"
#   vllm: VLLM_UTIL=0.6 (fraction of device memory), VLLM_SPEC='{"method":"ngram",...}'
# Results: ~/aq/results/trace/<tag>/agents<N>.json
set -euo pipefail
ENGINE=$1
MODEL=$2
TRACES=$3
TAG=$4
AGENTS=${AGENTS:-"4 8 16"}
DURATION=${DURATION:-240}
WARMUP=${WARMUP:-60}
MAX_CTX=33792
OUT=$HOME/aq/results/trace/$TAG
PORT=${PORT:-18300}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
source ~/aq/venv-ref/bin/activate

if [ "$ENGINE" = vllm ]; then
  # VLLM_SPEC: JSON for --speculative-config (e.g. n-gram prompt lookup).
  VLLM_ARGS=()
  [ -n "${VLLM_SPEC:-}" ] && VLLM_ARGS=(--speculative-config "$VLLM_SPEC")
  if [ -d "$HOME/aq/cuda-shim" ]; then
    export CUDA_HOME=$HOME/aq/cuda-shim PATH=$PATH:$HOME/aq/cuda-shim/bin NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
  else
    export CUDA_HOME=/usr/local/cuda-13.0 PATH=$PATH:/usr/local/cuda-13.0/bin
  fi
  vllm serve "$MODEL" --port "$PORT" --max-model-len $MAX_CTX --gpu-memory-utilization "${VLLM_UTIL:-0.6}" \
    --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 64 "${VLLM_ARGS[@]}" > "$OUT/server.log" 2>&1 &
  LABEL="vLLM $(python -c 'import vllm; print(vllm.__version__)')"
else
  "${AQ_SERVE_BIN:?set AQ_SERVE_BIN}" "$MODEL" --port "$PORT" --max-seqs 128 --max-running 32 --max-ctx $MAX_CTX \
    --kv-gb "${AQ_KV_GB:-48}" --mixed-prefill 2048 --prewarm true --pin-snapshots true ${AQ_ARGS:-} > "$OUT/server.log" 2>&1 &
  LABEL="aqwenator $TAG"
fi
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 1200); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 1
done
echo "engine=$LABEL model=$MODEL traces=$TRACES" | tee "$OUT/meta.txt"
for n in $AGENTS; do
  python "$HERE/trace_bench.py" run --traces "$TRACES" --url "http://localhost:$PORT" --model "$MODEL" --engine "$LABEL" \
    --agents "$n" --warmup "$WARMUP" --duration "$DURATION" --out "$OUT/agents$n.json" | tee -a "$OUT/summary.txt"
done
echo SWEEP_DONE
