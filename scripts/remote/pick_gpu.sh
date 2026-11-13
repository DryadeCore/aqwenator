#!/usr/bin/env bash
# Print the index of an idle GPU on the shared box (lowest memory used, 0% util). Exit 1 if none.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' '$3 < 5 && $2 < 2000 {print $2, $1}' | sort -n | head -1 | awk '{print $2}' | grep . || { echo "no idle GPU" >&2; exit 1; }
