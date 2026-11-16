#!/usr/bin/env bash
# Build $AQ_HOME/cuda-shim (default ~/aq): a CUDA_HOME layout (bin, include, lib64 with unversioned .so names,
# nvvm) made of symlinks into the CUDA 13 toolkit that ships with torch's pip wheels, so
# vLLM/FlashInfer JIT builds compile and link against exactly what torch was built with.
# Only needed for the vLLM baselines on a machine without a system CUDA 13 toolkit.
set -euo pipefail
AQ_HOME=${AQ_HOME:-$HOME/aq}
SRC=$(echo "${AQ_VENV:-$AQ_HOME/venv-ref}"/lib/python3.*/site-packages/nvidia/cu13)
DST=$AQ_HOME/cuda-shim
rm -rf "$DST"
mkdir -p "$DST/lib64"
ln -s "$SRC/bin" "$DST/bin"
ln -s "$SRC/include" "$DST/include"
ln -s "$SRC/nvvm" "$DST/nvvm"
for f in "$SRC"/lib/*; do
  ln -s "$f" "$DST/lib64/$(basename "$f")"
  base=$(basename "$f" | sed -E 's/\.so\..*$/.so/')
  [ "$base" != "$(basename "$f")" ] && [ ! -e "$DST/lib64/$base" ] && ln -s "$f" "$DST/lib64/$base"
done
ln -s lib64 "$DST/lib"
ls "$DST/lib64" | grep -E "cudart|cublas" | head
