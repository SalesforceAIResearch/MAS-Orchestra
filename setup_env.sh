#!/bin/bash
# Environment setup for MAS-Orchestra (follows README / verl_setup.txt).
set -e
source /sw/eb/sw/Miniforge3/25.3.0-3/etc/profile.d/conda.sh

ENV_NAME=mas-orchestra
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n ${ENV_NAME} python=3.10
fi
conda activate ${ENV_NAME}
ml CUDA/12.9.1 || true

cd "${REPO_DIR}/verl"
# FSDP only: Megatron/TransformerEngine are not needed for evaluation
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
rm -f flash_attn-*.whl flashinfer_python-*.whl
pip install --no-deps -e .

pip install ray==2.49.2 --force-reinstall
pip install protobuf==4.25.8 --force-reinstall
pip install together
pip install "math-verify[antlr4_13_2]"
pip install antlr4-python3-runtime==4.9.3

pip install langchain-core langchain-together langchain-community duckduckgo-search tavily-python pydantic ddgs langchain_brightdata bs4
pip install pyserini faiss-gpu
pip install git+https://github.com/texttron/tevatron.git

conda install -y -c conda-forge openjdk=21

# pyserini/faiss-gpu/sglang deps upgrade torch, torchao, numpy, etc. past what vllm 0.8.5 supports
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "torchao==0.9.0" \
    "numpy==1.26.4" "setuptools<81" "wandb<0.22" "peft==0.15.2" "liger-kernel==0.5.8" \
    "cupy-cuda12x==13.4.0" "opencv-python==4.11.0.86" "opencv-python-headless==4.11.0.86" \
    "opentelemetry-sdk>=1.26.0,<1.27.0" "opentelemetry-api>=1.26.0,<1.27.0" \
    "opentelemetry-exporter-otlp>=1.26.0,<1.27.0" "opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0"

echo "MAS-Orchestra env setup finished"
