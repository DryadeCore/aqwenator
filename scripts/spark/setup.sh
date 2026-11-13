#!/usr/bin/env bash
# User-space toolchain on the DGX Spark (aarch64, sm_121). No sudo, no GPU use.
set -euxo pipefail
AQ=$HOME/aq
mkdir -p "$AQ/bin" "$AQ/logs"
cd "$AQ"
if [ ! -x "$AQ/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C "$AQ" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$AQ/mamba
[ -d "$AQ/env" ] || "$AQ/bin/micromamba" create -y -p "$AQ/env" -c conda-forge 'clang=21' 'clangdev=21' 'libclang=21' pkg-config
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none --profile minimal
fi
source "$HOME/.cargo/env"
rustup toolchain install 1.99.1 --profile minimal -c rust-src -c rustc-dev -c rust-analyzer -c clippy -c rustfmt -c llvm-tools
[ -d "$AQ/cuda-rust" ] || git clone --depth 1 https://github.com/NVIDIA/cuda-rust.git "$AQ/cuda-rust"
source "$AQ/src/scripts/spark/env.sh"
nice -n 10 cargo +1.99.1 install --path "$AQ/cuda-rust/cuda-oxide/crates/cargo-oxide" --locked
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$AQ/cuda-rust/cuda-oxide" && cargo oxide doctor || true
echo SPARK_TOOLCHAIN_OK
