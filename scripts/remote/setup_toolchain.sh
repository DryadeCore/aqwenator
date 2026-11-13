#!/usr/bin/env bash
# User-space toolchain for the B200 box (no sudo, no docker there).
#   ~/aq/bin/micromamba   conda env manager
#   ~/aq/env              CUDA 13 toolkit + clang/libclang 21 (for cuda-oxide's bindgen) + nsight compute
#   rustup 1.99.1         pinned cuda-oxide toolchain
#   ~/aq/cuda-rust        NVIDIA/cuda-rust checkout, cargo-oxide installed from it
set -euxo pipefail
AQ=$HOME/aq
mkdir -p "$AQ/bin" "$AQ/logs"
cd "$AQ"

if [ ! -x "$AQ/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$AQ" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$AQ/mamba
if [ ! -d "$AQ/env" ]; then
  "$AQ/bin/micromamba" create -y -p "$AQ/env" -c conda-forge \
    'cuda-toolkit=13.0' 'clang=21' 'clangdev=21' 'libclang=21' 'llvmdev=21' 'nsight-compute' pkg-config
fi

source "$HOME/.cargo/env"
rustup toolchain install 1.99.1 --profile minimal \
  -c rust-src -c rustc-dev -c rust-analyzer -c clippy -c rustfmt -c llvm-tools

if [ ! -d "$AQ/cuda-rust" ]; then
  git clone --depth 1 https://github.com/NVIDIA/cuda-rust.git "$AQ/cuda-rust"
fi
source "$AQ/src/scripts/remote/env.sh"
cargo +1.99.1 install --path "$AQ/cuda-rust/cuda-oxide/crates/cargo-oxide" --locked || \
  cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
cd "$AQ/cuda-rust/cuda-oxide"
cargo oxide doctor
echo TOOLCHAIN_OK
