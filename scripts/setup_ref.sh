#!/usr/bin/env bash
# Reference Python env in $AQ_HOME/venv-ref (default ~/aq): vLLM for the baselines and
# transformers for the reference data of aq-check (scripts/golden.py). Needs uv.
#   MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B"  checkpoints to download
set -euxo pipefail
AQ=${AQ_HOME:-$HOME/aq}
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 "$AQ/venv-ref"
source "$AQ/venv-ref/bin/activate"
uv pip install -U vllm transformers accelerate hf_transfer 'huggingface_hub[cli]'
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in ${MODELS:-Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B}; do hf download "$m"; done
echo REF_OK
