#!/bin/bash
#SBATCH --job-name=MAS_Orchestra_multiturn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_train_multiturn_%j.log

# Outputs: checkpoints/val dumps/logs in $CKPT_ROOT/$RUN_NAME, plots in W&B project $WANDB_PROJECT.
# [ToolOrchestra step 4] GRPO training of the multi-turn (dynamic plan-execute) orchestrator:
# Qwen2.5-7B-Instruct calls CoTAgent / SCAgent / ReflexionAgent / DebateAgent (gpt-oss-120b) as tools, up to 6 turns.
#
# Usage: sbatch train_multiturn.sh            (key read from ~/.openrouter_api_key or ~/.together_api_key)
# Extra Hydra overrides can be appended:  sbatch train_multiturn.sh trainer.total_training_steps=100

set -o pipefail
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started on $(hostname)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$PROJECT_DIR"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_FILES="${PROJECT_DIR}/data/deepscaler/train.parquet"
VAL_FILES=${VAL_FILES:-"['${PROJECT_DIR}/data/aime24/test.parquet']"}
TOOL_CONFIG=${TOOL_CONFIG:-${PROJECT_DIR}/mas_r1_reasoner/configs/tool_config/mas_agents.yaml}

# Sub-agents (gpt-oss-120b) are called through OpenRouter (OpenRouterCompletionSampler)
if [ -z "$OPENROUTER_API_KEY" ] && [ -f ~/.openrouter_api_key ]; then
    export OPENROUTER_API_KEY=$(cat ~/.openrouter_api_key)
fi
if [ -z "$OPENROUTER_API_KEY" ] && ! grep -q "mock_output: true" "$TOOL_CONFIG"; then
    echo "ERROR: set OPENROUTER_API_KEY or create ~/.openrouter_api_key (chmod 600)"
    exit 1
fi
# Other samplers in the base config's model_sampler_map are constructed at startup and need a key to exist
export TOGETHER_API_KEY=${TOGETHER_API_KEY:-unused}

ml CUDA/12.9.1
source /sw/eb/sw/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate mas-orchestra

# Fail fast on a bad key / no credits: otherwise every agent call would silently return an empty string
if ! grep -q "mock_output: true" "$TOOL_CONFIG"; then
    python scripts/check_openrouter.py || exit 1
fi

if [ ! -f "$TRAIN_FILES" ]; then
    echo "Preparing DeepScaleR -> $TRAIN_FILES"
    python scripts/prepare_deepscaler.py --out "$TRAIN_FILES" || exit 1
fi

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export RAY_memory_usage_threshold=0.9
export RAY_memory_monitor_refresh_ms=0
# Rollout TP ranks wait in a collective while their partner runs a whole multi-turn batch (up to 6 turns x 900 s agent
# timeout); the 30 min PyTorch default killed job 1115212 at step 132.
export VERL_DIST_TIMEOUT_S=${VERL_DIST_TIMEOUT_S:-10800}

ray stop --force 2>/dev/null || true
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-local}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS

RUN_NAME=${RUN_NAME:-harmony_multiturn_$(date +%Y%m%d_%H%M%S)}
# Checkpoints, validation dumps, logs and W&B files of a run all live in one project folder
CKPT_ROOT=${CKPT_ROOT:-/scratch/project/prj-02-llm-reasoning-shakkottai/Multi_agent_systems}
SAVE_PATH="${CKPT_ROOT}/${RUN_NAME}"
mkdir -p "$SAVE_PATH" logs

# W&B project "multi-agent-experiments" (login: ~/.netrc from `wandb login`, or WANDB_API_KEY)
WANDB_PROJECT=${WANDB_PROJECT:-multi-agent-experiments}
export WANDB_DIR="${SAVE_PATH}"
LOGGER="['console','wandb']"
if [ -z "$WANDB_API_KEY" ] && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    echo "WARNING: no W&B login found; logging to console only (run 'wandb login' to enable plots)"
    LOGGER="['console']"
fi

python -u -m mas_r1_reasoner.main_mas_r1 \
    --config-path=configs \
    --config-name=grpo_trainer_multiturn \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.checkpoint_root="${CKPT_ROOT}" \
    trainer.default_local_dir="${SAVE_PATH}/checkpoints" \
    trainer.validation_data_dir="${SAVE_PATH}/val" \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE:-8} \
    trainer.nnodes=1 \
    ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-64} \
    hydra.run.dir="${SAVE_PATH}/hydra" \
    "$@" 2>&1 | tee "${SAVE_PATH}/run.log"
STATUS=$?

ray stop --force 2>/dev/null || true
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished with status ${STATUS}"
exit $STATUS
