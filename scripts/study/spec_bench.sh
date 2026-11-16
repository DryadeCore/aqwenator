#!/usr/bin/env bash
# Speculative decoding A/B on one GPU: 9B, random prompts (1024/256, 8192/512) and quote-back
# prompts, spec off vs prompt-lookup k=4.
#   BIN=path/to/aq-bench  PROMPTS=quote prompts from make_quote_prompts.py  OUT=results dir
set -euo pipefail
GPU=${1:-1}
BIN=${BIN:?set BIN to an aq-bench binary}
OUT=${OUT:-${AQ_HOME:-$HOME/aq}/results/spec}
PROMPTS=${PROMPTS:?set PROMPTS to the quote prompts file}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
M=Qwen/Qwen3.5-9B
for k in 0 4; do
  echo "== random 1024/256 spec_k=$k"
  $BIN $M --input-len 1024 --output-len 256 --concurrency 1,4,8,16 --num-prompts 64 --max-seqs 64 --spec-k $k --out $OUT/rand1k-k$k.json 2>&1 | grep -E "c=|spec:"
  echo "== random 8192/512 spec_k=$k"
  $BIN $M --input-len 8192 --output-len 512 --concurrency 1,4,8 --num-prompts 32 --max-seqs 32 --spec-k $k --out $OUT/rand8k-k$k.json 2>&1 | grep -E "c=|spec:"
  echo "== quote-back spec_k=$k"
  $BIN $M --input-len 1024 --output-len 256 --concurrency 1,4,8,16 --num-prompts 88 --max-seqs 64 --spec-k $k --prompts-file $PROMPTS --out $OUT/quote-k$k.json 2>&1 | grep -E "c=|spec:"
done
echo SPEC_BENCH_DONE
