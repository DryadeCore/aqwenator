#!/usr/bin/env bash
# Speculative decoding A/B on one GPU: 9B, random prompts (1024/256, 8192/512) and quote-back
# prompts, spec off vs prompt-lookup k=4. Results in ~/aq/results/spec/.
set -euo pipefail
GPU=${1:-1}
BIN=${BIN:-$HOME/aq/src-spec/target/release/aq-bench}
OUT=$HOME/aq/results/spec
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
M=Qwen/Qwen3.5-9B
for k in 0 4; do
  echo "== random 1024/256 spec_k=$k"
  $BIN $M --input-len 1024 --output-len 256 --concurrency 1,4,8,16 --num-prompts 64 --max-seqs 64 --spec-k $k --out $OUT/rand1k-k$k.json 2>&1 | grep -E "c=|spec:"
  echo "== random 8192/512 spec_k=$k"
  $BIN $M --input-len 8192 --output-len 512 --concurrency 1,4,8 --num-prompts 32 --max-seqs 32 --spec-k $k --out $OUT/rand8k-k$k.json 2>&1 | grep -E "c=|spec:"
  echo "== quote-back spec_k=$k"
  $BIN $M --input-len 1024 --output-len 256 --concurrency 1,4,8,16 --num-prompts 88 --max-seqs 64 --spec-k $k --prompts-file $HOME/aq/prompts/quote9b.json --out $OUT/quote-k$k.json 2>&1 | grep -E "c=|spec:"
done
echo SPEC_BENCH_DONE
