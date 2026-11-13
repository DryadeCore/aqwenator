#!/usr/bin/env bash
# vLLM baseline sweep on one GPU of the shared B200 box.
# Usage: tools/baseline_vllm.sh <model> <tag> [gpu]
# Writes vllm bench JSON results to ~/aq/results/baseline/<tag>/.
set -euo pipefail
MODEL=$1
TAG=$2
GPU=${3:-$(bash ~/aq/src/scripts/remote/pick_gpu.sh)}
OUT=$HOME/aq/results/baseline/$TAG
PORT=$((18000 + GPU))
mkdir -p "$OUT"
source ~/aq/venv-ref/bin/activate
# vLLM JIT-compiles some kernels (flashinfer); use the CUDA 13 toolkit that ships with torch's
# pip wheels so headers and libs match the torch build.
export CUDA_HOME=$HOME/aq/cuda-shim  # built by scripts/remote/make_cuda_shim.sh
export PATH=$PATH:$CUDA_HOME/bin
# conda's GCC 15 headers are too new for CUDA 13.0's front end; host-compile with the system GCC 13.
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"
export CUDA_VISIBLE_DEVICES=$GPU

echo "model=$MODEL gpu=$GPU port=$PORT" | tee "$OUT/meta.txt"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee -a "$OUT/meta.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" | tee -a "$OUT/meta.txt"

vllm serve "$MODEL" --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 256 \
  > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -50 "$OUT/server.log"; exit 1; }
  sleep 5
done

run() { # in out conc
  local in=$1 out=$2 c=$3 n=$(( $3 * 4 > 32 ? $3 * 4 : 32 ))
  vllm bench serve --backend vllm --model "$MODEL" --port "$PORT" \
    --dataset-name random --random-input-len "$in" --random-output-len "$out" \
    --num-prompts "$n" --max-concurrency "$c" --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "in${in}_out${out}_c${c}.json" \
    > "$OUT/bench_in${in}_out${out}_c${c}.log" 2>&1
  python - "$OUT/in${in}_out${out}_c${c}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{sys.argv[1].split('/')[-1]:28s} out_tok/s={r['output_throughput']:9.1f} "
      f"ttft_p50={r['median_ttft_ms']:8.1f}ms ttft_p99={r['p99_ttft_ms']:8.1f}ms "
      f"tpot_p50={r['median_tpot_ms']:7.2f}ms itl_p99={r['p99_itl_ms']:7.2f}ms")
PY
}

for c in 1 8 32 64 128 256; do run 1024 256 "$c"; done
for c in 1 16 64; do run 8192 512 "$c"; done
echo BASELINE_DONE
