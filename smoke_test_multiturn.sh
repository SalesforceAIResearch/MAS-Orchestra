#!/bin/bash
#SBATCH --job-name=MAS_Orchestra_multiturn_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_smoke_multiturn_%j.log

# [ToolOrchestra step 4] End-to-end check of the multi-turn loop without API cost:
# mock agents (mas_agents_mock.yaml), AIME24 as tiny train set, 2 training steps, short sequences.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export TOOL_CONFIG="${PROJECT_DIR}/mas_r1_reasoner/configs/tool_config/mas_agents_mock.yaml"
export RUN_NAME="smoke_multiturn_$(date +%Y%m%d_%H%M%S)"
export SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-2}

bash "${PROJECT_DIR}/train_multiturn.sh" \
    data.train_files="${PROJECT_DIR}/data/aime24/test.parquet" \
    data.val_files="${PROJECT_DIR}/data/aime24/test.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=6144 \
    data.max_validation_prompt_length=6144 \
    data.max_response_length=4096 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.multi_turn.max_turns=3 \
    azr.mas_r1.mock_output=True \
    trainer.val_before_train=False \
    trainer.total_training_steps=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1
