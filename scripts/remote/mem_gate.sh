#!/usr/bin/env bash
# Correctness gate for the memory-policy lane (run on the B200 from ~/aq/src-mem2).
# Waits until the driver answers, then runs aq-check on the gate goldens in default mode,
# with odd step budgets/chunks, and under the policy combinations the lane benchmarks.
cd ~/aq/src-mem2
until timeout 10 nvidia-smi -i 6 --query-gpu=index --format=csv,noheader >/dev/null 2>&1; do sleep 20; done
echo "driver ok $(date)"
export CUDA_VISIBLE_DEVICES=6
model() { echo "$1" | sed "s/qwen35-0.8b.*/Qwen\/Qwen3.5-0.8B/; s/qwen35-9b.*/Qwen\/Qwen3.5-9B/"; }
for env in "" "AQ_CHECK_BUDGET=1000 AQ_CHECK_CHUNK=777" "AQ_CHECK_POLICY=idle,pin" "AQ_CHECK_POLICY=idle,pin,load,q4" "AQ_CHECK_POLICY=idle,pin,load,q4,srpt AQ_CHECK_BUDGET=1000 AQ_CHECK_CHUNK=777"; do
  for g in qwen35-0.8b qwen35-0.8b-long qwen35-9b qwen35-9b-long qwen35-9b-16k; do
    case "$env" in AQ_CHECK_POLICY*) case $g in *long*) ;; *) continue ;; esac ;; esac
    echo "== $g [$env]"
    env $env timeout 900 ./target/release/aq-check "$(model $g)" ~/aq/golden/$g.safetensors 2>&1 \
      | grep -E "teacher|greedy vs|batched seq|session|shared|pressure|PASS|FAIL|MISMATCH|rror|panic"
  done
done
echo GATE_DONE
