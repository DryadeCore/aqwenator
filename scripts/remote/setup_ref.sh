#!/usr/bin/env bash
# Reference Python env on the B200 box: latest vLLM (baselines) + transformers (golden activations).
# Kept separate from the user's existing venvs.
set -euxo pipefail
AQ=$HOME/aq
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 "$AQ/venv-ref"
source "$AQ/venv-ref/bin/activate"
uv pip install -U vllm transformers accelerate hf_transfer 'huggingface_hub[cli]'
python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.5-0.8B
hf download Qwen/Qwen3.5-9B
hf download Qwen/Qwen3.8-27B-FP8
echo REF_OK
