#!/usr/bin/env bash
# vLLM baseline sweep (synthetic prompts) on one GPU.
# Usage: scripts/baseline_vllm.sh <model> <tag> [gpu]
#   CONC="1 4 8 16 32 64"      concurrency levels
#   SHAPES="1024:256 8192:512" input:output token lengths
#   AQ_HOME=~/aq  AQ_VENV=$AQ_HOME/venv-ref (see scripts/setup_ref.sh)
# Writes vllm bench JSON results to $AQ_HOME/results/baseline/<tag>/.
set -euo pipefail
MODEL=$1
TAG=$2
GPU=${3:-0}
CONC=${CONC:-"1 4 8 16 32 64"}
SHAPES=${SHAPES:-"1024:256 8192:512"}
AQ_HOME=${AQ_HOME:-$HOME/aq}
OUT=$AQ_HOME/results/baseline/$TAG
PORT=$((18000 + GPU))
mkdir -p "$OUT"
source "${AQ_VENV:-$AQ_HOME/venv-ref}/bin/activate"
# vLLM JIT-compiles some kernels (flashinfer); use the CUDA 13 toolkit that ships with torch's
# pip wheels so headers and libs match the torch build.
if [ -d /usr/local/cuda/include ] && [ ! -d "$AQ_HOME/cuda-shim" ]; then
  export CUDA_HOME=/usr/local/cuda        # a system toolkit exists
else
  export CUDA_HOME=$AQ_HOME/cuda-shim     # built by scripts/cuda_shim.sh
  # conda's GCC 15 headers are too new for CUDA 13.0's front end; use the system GCC 13.
  export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
fi
export PATH=$PATH:$CUDA_HOME/bin
export CUDA_VISIBLE_DEVICES=$GPU

echo "model=$MODEL gpu=$GPU port=$PORT" | tee "$OUT/meta.txt"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee -a "$OUT/meta.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" | tee -a "$OUT/meta.txt"

T0=$(date +%s.%N)
vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --gpu-memory-utilization ${GPU_UTIL:-0.85} \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 256 ${VLLM_EXTRA_ARGS:-} \
  > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

READY=0
for i in $(seq 1 900); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && { READY=1; break; }
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
  sleep 1
done
[ $READY = 1 ] || { echo "server not ready after 900 s"; exit 1; }
T1=$(date +%s.%N)
echo "cold_start_ready_s=$(echo "$T1 - $T0" | bc)" | tee -a "$OUT/meta.txt"

run() { # in out conc
  local in=$1 out=$2 c=$3
  local n=$(( $3 * ${NPROMPTS_MULT:-16} > ${NPROMPTS_MIN:-200} ? $3 * ${NPROMPTS_MULT:-16} : ${NPROMPTS_MIN:-200} ))
  vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --num-prompts "$n" --max-concurrency "$c" --ignore-eos --temperature 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99,99.9 \
    --save-result --result-dir "$OUT" --result-filename "in${in}_out${out}_c${c}.json" \
    > "$OUT/bench_in${in}_out${out}_c${c}.log" 2>&1
  python - "$OUT/in${in}_out${out}_c${c}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
g = lambda k: r.get(k, float("nan"))
print(f"{sys.argv[1].split('/')[-1]:28s} out_tok/s={g('output_throughput'):9.1f} "
      f"ttft p50/p99/p999={g('median_ttft_ms'):7.1f}/{g('p99_ttft_ms'):7.1f}/{g('p99.9_ttft_ms'):7.1f}ms "
      f"itl p50/p99/p999={g('median_itl_ms'):6.2f}/{g('p99_itl_ms'):6.2f}/{g('p99.9_itl_ms'):6.2f}ms")
PY
}

for shape in $SHAPES; do
  for c in $CONC; do run "${shape%%:*}" "${shape##*:}" "$c"; done
done
echo BASELINE_DONE
