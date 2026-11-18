#!/usr/bin/env bash
# Real-trace replay (scripts/trace_bench.py) against llama.cpp's llama-server.
# Usage: scripts/baseline_llamacpp.sh <model dir (HF snapshot)> <traces.json> <tag> [extra llama-server args]
#   LLAMA_DIR=~/llama.cpp (built with -DGGML_CUDA=ON; its convert_hf_to_gguf.py makes the GGUF)
#   GGUF_TYPE=f16 (f16|bf16|q8_0)  AGENTS="4 16"  DURATION=240 WARMUP=60  MAX_CTX=33792
#   AQ_HOME=~/aq  AQ_VENV=$AQ_HOME/venv-ref (python with torch, transformers, aiohttp)
# Results: $AQ_HOME/results/trace/<tag>/agents<N>.json. One server per agent count: llama-server
# splits its context between its slots, so both are set from the agent count.
set -euo pipefail
SNAP=$1
TRACES=$2
TAG=$3
shift 3
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
TYPE=${GGUF_TYPE:-f16}
AGENTS=${AGENTS:-"4 16"}
MAX_CTX=${MAX_CTX:-33792}
AQ_HOME=${AQ_HOME:-$HOME/aq}
OUT=$AQ_HOME/results/trace/$TAG
PORT=${PORT:-18390}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" "$AQ_HOME/models"
source "${AQ_VENV:-$AQ_HOME/venv-ref}/bin/activate"
GGUF=$AQ_HOME/models/$(basename "$(dirname "$(dirname "$SNAP")")" | sed 's/.*--//' | tr 'A-Z' 'a-z')-$TYPE.gguf
[ -f "$GGUF" ] || PYTHONPATH=$LLAMA_DIR/gguf-py python "$LLAMA_DIR/convert_hf_to_gguf.py" "$SNAP" --outtype "$TYPE" --outfile "$GGUF" > "$OUT/convert.log" 2>&1
VERSION=$("$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | sed -n 's/^version: //p')
for n in $AGENTS; do
  "$LLAMA_DIR/build/bin/llama-server" -m "$GGUF" -ngl 999 -c $((MAX_CTX * n)) -np "$n" --port "$PORT" --host 127.0.0.1 --no-webui "$@" > "$OUT/server$n.log" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
  for i in $(seq 1 600); do
    curl -sf "localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/server$n.log"; exit 1; }
    sleep 1
  done
  python "$HERE/trace_bench.py" run --traces "$TRACES" --url "http://localhost:$PORT" --model "$(basename "$GGUF")" --engine "llama.cpp $VERSION $TYPE $*" \
    --agents "$n" --warmup "${WARMUP:-60}" --duration "${DURATION:-240}" --out "$OUT/agents$n.json" | tee -a "$OUT/summary.txt"
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true
done
echo SWEEP_DONE
