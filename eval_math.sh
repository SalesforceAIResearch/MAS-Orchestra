#!/bin/bash
#SBATCH --job-name=MAS_Orchestra_eval_math
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_eval_math_%j.log

# Evaluate the released math orchestrator (low DoM, gpt-oss-120b/low sub-agents via Together)
# on AIME24 + AIME25 with Avg@8, matching the paper's public-benchmark setup.
#
# Usage: TOGETHER_API_KEY=... sbatch eval_math.sh [checkpoint_name]
#    or: OPENROUTER_API_KEY=... sbatch eval_math.sh [checkpoint_name]

set -o pipefail

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started on $(hostname)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$PROJECT_DIR"

CKPT_NAME=${1:-harmony-grpo-7b-global-step-180}
CKPT_DIR="${PROJECT_DIR}/checkpoints/${CKPT_NAME}"
MODEL_PATH="${CKPT_DIR}/hf_model"
VAL_FILES="['${PROJECT_DIR}/data/aime24/test.parquet','${PROJECT_DIR}/data/aime25/test.parquet']"
TRAIN_FILES="${PROJECT_DIR}/data/aime24/test.parquet"

if [ -z "$TOGETHER_API_KEY" ] && [ -f ~/.together_api_key ]; then
    export TOGETHER_API_KEY=$(cat ~/.together_api_key)
fi
if [ -z "$OPENROUTER_API_KEY" ] && [ -f ~/.openrouter_api_key ]; then
    export OPENROUTER_API_KEY=$(cat ~/.openrouter_api_key)
fi
SAMPLER_OVERRIDES=()
if [ -z "$TOGETHER_API_KEY" ] && [ -n "$OPENROUTER_API_KEY" ]; then
    # gpt-oss-120b sub-agents through OpenRouter, preferring Together as the provider
    SAMPLER_OVERRIDES=(
        "azr.mas_r1.agent.model_sampler_map.gpt-oss-120b.type=OpenRouterCompletionSampler"
        "+azr.mas_r1.agent.model_sampler_map.gpt-oss-120b.provider.order=[together]"
        "+azr.mas_r1.agent.model_sampler_map.gpt-oss-120b.provider.allow_fallbacks=true"
    )
    export TOGETHER_API_KEY=unused   # other Together samplers in the map are constructed at startup
    echo "Using OpenRouter for sub-agent calls"
fi
if [ -z "$TOGETHER_API_KEY" ] || [ "$TOGETHER_API_KEY" = "YOUR_KEY" ]; then
    echo "ERROR: set TOGETHER_API_KEY or OPENROUTER_API_KEY (or ~/.together_api_key / ~/.openrouter_api_key)"
    exit 1
fi

ml CUDA/12.9.1
source /sw/eb/sw/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate mas-orchestra

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_usage_threshold=0.9
export RAY_memory_monitor_refresh_ms=0

# Merge FSDP shards into a HF model once
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Merging FSDP checkpoint ${CKPT_DIR}/actor -> ${MODEL_PATH}"
    python verl/scripts/model_merger.py merge --backend fsdp \
        --local_dir "${CKPT_DIR}/actor" --target_dir "${MODEL_PATH}" || exit 1
fi

ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS

RUN_NAME="eval_math_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${PROJECT_DIR}/eval_results/${RUN_NAME}"
mkdir -p "$SAVE_PATH"

LOGGER="['console']"
[ -n "$WANDB_API_KEY" ] && LOGGER="['console','wandb']"

python -u -m mas_r1_reasoner.main_mas_r1 \
    --config-path=configs \
    --config-name=grpo_trainer \
    data.raw_data=True \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=15000 \
    data.max_validation_prompt_length=15000 \
    data.max_response_length=8192 \
    data.train_batch_size=8 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    azr.problem_type=harmony_minimal \
    azr.mas_r1.use_llm_judge=False \
    azr.mas_r1.execution_success_weight=0.0 \
    azr.mas_r1.final_answer_weight=1.0 \
    azr.mas_r1.multiply_processes=0 \
    azr.mas_r1.max_ray_workers=1 \
    azr.mas_r1.agent.model_name=gpt-oss-120b \
    azr.mas_r1.agent.reasoning_effort=low \
    "azr.mas_r1.agent.init_archive=['COT','COT_SC','Reflexion','LLM_debate']" \
    trainer.logger="${LOGGER}" \
    trainer.project_name=mas_orchestra_eval \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${SAVE_PATH}/ckpt" \
    trainer.validation_data_dir="${SAVE_PATH}" \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    trainer.total_training_steps=0 \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE:-1} \
    trainer.nnodes=1 \
    ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-32} \
    hydra.run.dir="${SAVE_PATH}/hydra" \
    "${SAMPLER_OVERRIDES[@]}" 2>&1 | tee "${SAVE_PATH}/run.log"
EVAL_STATUS=$?
if [ $EVAL_STATUS -ne 0 ]; then
    echo "ERROR: evaluation exited with status ${EVAL_STATUS}; see ${SAVE_PATH}/run.log"
    ray stop --force 2>/dev/null || true
    exit $EVAL_STATUS
fi

python summarize_math_eval.py "${SAVE_PATH}"

ray stop --force 2>/dev/null || true
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished"
