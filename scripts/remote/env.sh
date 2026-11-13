# Source on the B200 box: CUDA 13 + clang 21 from ~/aq/env, Rust from rustup.
export AQ=$HOME/aq
export MAMBA_ROOT_PREFIX=$AQ/mamba
source "$HOME/.cargo/env"
export CUDA_HOME=$AQ/env
export CUDA_PATH=$CUDA_HOME
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$AQ/bin:$HOME/.local/bin:$PATH
export LIBCLANG_PATH=$CUDA_HOME/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=1
# Shared box: callers pick a free GPU with scripts/remote/pick_gpu.sh.
