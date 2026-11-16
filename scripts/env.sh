# Source this before building: paths for the user-space toolchain that scripts/setup.sh installs.
#   AQ_HOME    where setup.sh put it (default ~/aq)
#   CUDA_HOME  a CUDA 13 toolkit; default /usr/local/cuda if the machine has one, else the
#              toolkit in the conda env under $AQ_HOME/env
export AQ_HOME=${AQ_HOME:-$HOME/aq}
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -d /usr/local/cuda/include ]; then CUDA_HOME=/usr/local/cuda; else CUDA_HOME=$AQ_HOME/env; fi
fi
case $(uname -m) in aarch64) _aq_target=sbsa-linux ;; *) _aq_target=x86_64-linux ;; esac
export CUDA_HOME CUDA_PATH=$CUDA_HOME CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export MAMBA_ROOT_PREFIX=$AQ_HOME/mamba
export PATH=$CUDA_HOME/bin:$AQ_HOME/bin:$HOME/.local/bin:$PATH:$AQ_HOME/env/bin
export LIBCLANG_PATH=$AQ_HOME/env/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$CUDA_HOME/targets/$_aq_target/lib:${LD_LIBRARY_PATH:-}
export CPATH=$CUDA_HOME/targets/$_aq_target/include:${CPATH:-}
export HF_HUB_ENABLE_HF_TRANSFER=1
unset _aq_target
