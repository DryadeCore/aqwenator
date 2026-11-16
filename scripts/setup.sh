#!/usr/bin/env bash
# Build toolchain in user space (no sudo), under $AQ_HOME (default ~/aq):
#   bin/micromamba, env/   clang/libclang 21 for cuda-oxide's bindgen, plus the CUDA 13 toolkit
#                          when the machine has none in /usr/local/cuda
#   rustup 1.99.1          the toolchain cuda-oxide is pinned to
#   cargo-oxide            at the cuda-rust revision in Cargo.toml
# Then: source scripts/env.sh && cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins
set -euxo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AQ_HOME=${AQ_HOME:-$HOME/aq}
mkdir -p "$AQ_HOME/bin"
case $(uname -m) in aarch64) MAMBA_ARCH=linux-aarch64 ;; *) MAMBA_ARCH=linux-64 ;; esac
if [ ! -x "$AQ_HOME/bin/micromamba" ]; then
  curl -Ls "https://micro.mamba.pm/api/micromamba/$MAMBA_ARCH/latest" | tar -xj -C "$AQ_HOME" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$AQ_HOME/mamba
if [ ! -d "$AQ_HOME/env" ]; then
  PKGS=('clang=21' 'clangdev=21' 'libclang=21' pkg-config)
  [ -d /usr/local/cuda/include ] || PKGS+=('cuda-toolkit=13.0' 'llvmdev=21')
  "$AQ_HOME/bin/micromamba" create -y -p "$AQ_HOME/env" -c conda-forge "${PKGS[@]}"
fi
if [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  # rustup-init at a fixed version, checked against its published SHA-256.
  case $(uname -m) in
    aarch64) RUSTUP_TARGET=aarch64-unknown-linux-gnu RUSTUP_SHA=9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282 ;;
    *) RUSTUP_TARGET=x86_64-unknown-linux-gnu RUSTUP_SHA=b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55 ;;
  esac
  curl -sSf -o "$AQ_HOME/rustup-init" "https://static.rust-lang.org/rustup/archive/1.29.3/$RUSTUP_TARGET/rustup-init"
  echo "$RUSTUP_SHA  $AQ_HOME/rustup-init" | sha256sum -c -
  chmod +x "$AQ_HOME/rustup-init"
  "$AQ_HOME/rustup-init" -y --no-modify-path --default-toolchain none --profile minimal
  rm "$AQ_HOME/rustup-init"
fi
source "$HERE/env.sh"
rustup toolchain install 1.99.1 --profile minimal -c rust-src -c rustc-dev -c rust-analyzer -c clippy -c rustfmt -c llvm-tools
REV=$(sed -n 's/^cuda-core = .*rev = "\([0-9a-f]*\)".*/\1/p' "$HERE/../Cargo.toml")
cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git --rev "$REV" --locked cargo-oxide
cd "$HERE/.." && cargo oxide doctor
echo TOOLCHAIN_OK
