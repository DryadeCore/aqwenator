# Source on the Spark: system CUDA 13.0, clang/libclang 21 from ~/aq/env (bindgen), Rust from rustup.
export AQ=$HOME/aq
export MAMBA_ROOT_PREFIX=$AQ/mamba
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$AQ/bin:$HOME/.local/bin:$PATH:$AQ/env/bin
export LIBCLANG_PATH=$AQ/env/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}
