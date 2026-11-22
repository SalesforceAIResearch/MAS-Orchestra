#!/bin/bash
#SBATCH --job-name=MAS_Orchestra_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=0:40:00
#SBATCH --qos=high
#SBATCH --output=logs/slurm_smoke_%j.log

# Usage: sbatch smoke_test.sh [checkpoint_name]
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$PROJECT_DIR"
CKPT_DIR="${PROJECT_DIR}/checkpoints/${1:-harmony-grpo-7b-global-step-180}"
MODEL_PATH="${CKPT_DIR}/hf_model"

ml CUDA/12.9.1
source /sw/eb/sw/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate mas-orchestra
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('flash_attn', flash_attn.__version__)"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Merging FSDP checkpoint ${CKPT_DIR}/actor -> ${MODEL_PATH}"
    python verl/scripts/model_merger.py merge --backend fsdp \
        --local_dir "${CKPT_DIR}/actor" --target_dir "${MODEL_PATH}" || exit 1
fi
ls -la "${MODEL_PATH}"

python smoke_test.py "${MODEL_PATH}"
