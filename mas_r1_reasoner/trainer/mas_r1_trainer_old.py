
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Tuple, List, Any
from copy import deepcopy
from pathlib import Path

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager, _timer, apply_kl_penalty, compute_advantage, reduce_metrics, compute_data_metrics, compute_timing_metrics, AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import torch
from verl.utils.torch_functional import masked_mean
from mas_r1_reasoner.agents.common import get_prompt, main_rank_print
from mas_r1_reasoner.agents.logging_utils.stdout import PrettyPrinter as pp
from mas_r1_reasoner.data_precessor.MathDatasetProcessor import MathDatasetProcessor
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code

from mas_r1_reasoner.trainer.utils.helper import (
    get_safe_length,
    convert_numpy_types,
    create_mas_r1_dataloaders,
    initialize_mas_r1_agent_system,
    prepare_batch_for_generation,
    save_building_blocks_evaluation_results,
)

from mas_r1_reasoner.rewards.utils.execution import (
    execute_codes_and_store_results,
)
from mas_r1_reasoner.trainer.utils.rollout import (
    rollout_generation,
)
from mas_r1_reasoner.trainer.utils.log import (
    log_mas_r1_validation_scores,
    maybe_log_val_generations_to_wandb,
    maybe_log_train_generations_to_wandb,
    save_generated_code,
    save_responses_and_ground_truth,
    collect_generated_code_from_dict,
    save_mas_r1_checkpoint,
    collect_responses_and_ground_truth_from_validation_reward,
)
from mas_r1_reasoner.agents.shared_vars import get_global

class MASR1Trainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 device_name="cuda"):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # Set MAS-R1 specific attributes
        self.mas_r1_config = config.azr.get('mas_r1', {})

        self.processor = MathDatasetProcessor(self)


        # Set up global variables and initialize reference model sampler
        pp.status("Global Variables", "Setting up global variables from dataset processor...", "info")
        agent_config = self.mas_r1_config.get('agent', {})
        
        # Pass both agent_config and mas_r1_config to setup_global_variables
        self.processor.setup_global_variables(agent_config, self.mas_r1_config, config)

        # Call parent constructor first
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=device_name
        )

        
        # Tree architecture configuration
        # Set enable_tree_architecture=True to use the two-level tree architecture
        # This will call generate_and_extract_codes_structure instead of generate_and_extract_codes
        # Example config:
        # mas_r1:
        #   enable_tree_architecture: true  # Enable two-level tree architecture
        #   sub_agents_per_sub_task: 5     # Generate 5 sub-agents per sub-task
        
        # Get the enable_tree_architecture value from global variables (set by BaseDatasetProcessor)

        self.problem_type = get_global('global_problem_type')
       
        self.enable_tree_architecture = get_global("global_enable_tree_architecture")
       
        # Get the architecture_only_sequential value from global variables (set by BaseDatasetProcessor)
        self.architecture_only_sequential = get_global("global_architecture_only_sequential")
        
        self.sub_agents_per_sub_task = self.mas_r1_config.get('sub_agents_per_sub_task', 3)

        self.code_execution_timeout = self.mas_r1_config.get('code_execution_timeout')

        # Code saving configuration
        self.save_generated_code = self.mas_r1_config.get('code_saving', {}).get('save_generated_code', False)
        self.save_code_summary = self.mas_r1_config.get('code_saving', {}).get('save_code_summary', False)
        self.max_accumulated_steps = self.mas_r1_config.get('code_saving', {}).get('max_accumulated_steps', 10)
        self.save_responses_and_ground_truth = self.mas_r1_config.get('logging', {}).get('save_intermediate_results', False)
        self.dataset_type = get_global('global_dataset_name')

        # Initialize tracking attributes
        self._current_step_generated_code = None
        self._accumulated_generated_code = []
        self._current_step_responses = None
        self._accumulated_responses = []
        
        # Initialize agent system as None (will be set up later)
        self.agent_system = None
        
        # Initialize AgentSystem after global variables and reference model sampler are set up
        pp.status("AgentSystem", "Initializing AgentSystem for MAS code execution...", "info")
        initialize_mas_r1_agent_system(self, self.config)

        self.processor = MathDatasetProcessor(self)

        # import time
        # time.sleep(100000)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")


    def _validate(self):
        """
        Validation method using the same modular approach as training.
        Uses shared helper functions for consistency and maintainability.
        """
        main_rank_print(f"\n{'='*80}")
        main_rank_print("STARTING MAS-R1 VALIDATION")
        main_rank_print(f"{'='*80}")
        main_rank_print(f"Validation dataloader: {self.val_dataloader}")
        main_rank_print(f"Validation reward function: {self.val_reward_fn}")
        main_rank_print(f"Problem type: {self.problem_type}")
        main_rank_print(f"Dataset type: {self.dataset_type}")

        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []


        # MAS-R1 specific metrics for logging
        mas_r1_overall_scores = []
        mas_r1_code_execution_success = []
        mas_r1_final_answer_correctness = []

        # Store first batch data for wandb logging
        first_batch_test_batch = None
        first_batch_reward_extra_info = None

        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            main_rank_print(f"\n{'='*60}")
            main_rank_print(f"VALIDATION BATCH {batch_idx+1}/{len(self.val_dataloader)}")
            main_rank_print(f"{'='*60}")

            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # Repeat batch first for pass@k evaluation (like ray_trainer_new.py)
            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1)
            if val_n > 1:
                main_rank_print(f"Repeating batch {val_n} times for pass@k evaluation (before generation)...")
                batch = batch.repeat(repeat_times=val_n, interleave=True)
                main_rank_print(f"Batch size after repeat: {len(batch)}")

            test_gen_batch = prepare_batch_for_generation(self, batch)

            # Check for building blocks evaluation mode - handled in _prepare_raw_data_batch_for_generation
            eval_building_blocks = get_global("global_eval_building_blocks")
            if eval_building_blocks:
                # assert get_global("global_add_judge") # no need for this, already updatead the helper.py
                pp.status("Evaluation", "Building blocks evaluation mode enabled - results will be saved during validation", "info")
                return



            test_output_gen_batch = rollout_generation(self, test_gen_batch, is_validation=True)

            # Union the gen_batch (which has original data) with the final output
            test_batch = batch.union(test_output_gen_batch)
            # test_batch has tensor keys, unlike batch in fit()
            
            # Evaluate using reward function with return_dict=True to get extra info
            main_rank_print(f"\n{'='*60}")
            main_rank_print("COMPUTING VALIDATION REWARDS")
            main_rank_print(f"{'='*60}")
            
            reward_result = self.val_reward_fn(self, sample_outputs, is_validation=True, input_data=test_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_info = reward_result["reward_extra_info"]
            sample_outputs = reward_result["sample_outputs"]
            # execution_results_dataproto = reward_result["final_output"]

            # Update test_batch to use expanded 32-size data for proper validation
            expanded_data = reward_result.get("expanded_data")
            if expanded_data is not None:
                main_rank_print(f"Updating validation batch from {len(test_batch)} to {len(expanded_data)} responses")
                test_batch = expanded_data
            else:
                raise RuntimeError("No expanded_data returned from validation reward manager")

            main_rank_print(f"Validation rewards computed successfully!")
            main_rank_print(f"Reward tensor shape: {reward_tensor.shape}")
            main_rank_print(f"Validation batch size after update: {len(test_batch)}")
            main_rank_print(f"Average reward: {reward_tensor.mean().item():.4f}")


            # Store original inputs for wandb logging (use gen_batch which has the original data)
            # We want to see the last step input_ids, not the original one
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Collect responses and ground truth from validation reward for saving
            current_step_responses, accumulated_responses = collect_responses_and_ground_truth_from_validation_reward(
                batch=test_batch,
                reward_extra_info=reward_extra_info,
                global_steps=self.global_steps,
                save_responses_and_ground_truth=self.save_responses_and_ground_truth
            )
            
            # Store the results in the trainer instance for saving
            if current_step_responses is not None:
                self._current_step_responses = current_step_responses
            if accumulated_responses is not None:
                self._accumulated_responses = accumulated_responses

            # Extract MAS-R1 specific metrics from reward_extra_info
            if 'combined_reward' in reward_extra_info:
                mas_r1_overall_scores.extend(reward_extra_info['combined_reward'])
            if 'code_execution_success' in reward_extra_info:
                mas_r1_code_execution_success.extend(reward_extra_info['code_execution_success'])
            if 'final_answer_correctness' in reward_extra_info:
                mas_r1_final_answer_correctness.extend(reward_extra_info['final_answer_correctness'])

            # Store scores for wandb logging
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            # Get data_source from meta_info if available, otherwise use default
            data_source = test_batch.meta_info.get('mas_r1_data_source', 'math')
            data_source_lst.append([data_source] * reward_tensor.shape[0])

            # Store first batch data for wandb logging
            if first_batch_test_batch is None:
                first_batch_test_batch = test_batch
                first_batch_reward_extra_info = reward_extra_info

        # Log MAS-R1 specific metrics
        if mas_r1_overall_scores:
            avg_overall_score = sum(mas_r1_overall_scores) / len(mas_r1_overall_scores)
            avg_code_execution = sum(mas_r1_code_execution_success) / len(mas_r1_code_execution_success)
            avg_answer_correctness = sum(mas_r1_final_answer_correctness) / len(mas_r1_final_answer_correctness)
            
            self.logger.log(data={
                'val/mas_r1/overall_score': avg_overall_score,
                'val/mas_r1/code_execution_success': avg_code_execution,
                'val/mas_r1/final_answer_correctness': avg_answer_correctness,
            }, step=self.global_steps)
            
            main_rank_print(f"MAS-R1 Validation Metrics:")
            main_rank_print(f"  - Overall Score: {avg_overall_score:.4f}")
            main_rank_print(f"  - Code Execution Success: {avg_code_execution:.4f}")
            main_rank_print(f"  - Final Answer Correctness: {avg_answer_correctness:.4f}")
            main_rank_print(f"  - scores: {scores}")

        # Log validation samples to wandb
        self.validation_table = maybe_log_val_generations_to_wandb(
            inputs=sample_inputs, 
            outputs=sample_outputs, 
            scores=sample_scores,
            config=self.config,
            global_steps=self.global_steps,
            validation_table=getattr(self, 'validation_table', None),
            test_batch=first_batch_test_batch if first_batch_test_batch is not None else None,
            reward_extra_info=first_batch_reward_extra_info if first_batch_reward_extra_info is not None else None
        )

        # Aggregate results
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # Use VERL's process_validation_metrics to compute pass@k statistics
        from verl.trainer.ppo.metric_utils import process_validation_metrics
        
        # Prepare reward_extra_infos_dict for VERL metrics computation
        reward_extra_infos_dict = {"reward": sample_scores}
        if mas_r1_overall_scores:
            reward_extra_infos_dict["combined_reward"] = mas_r1_overall_scores
        if mas_r1_code_execution_success:
            reward_extra_infos_dict["code_execution_success"] = mas_r1_code_execution_success
        if mas_r1_final_answer_correctness:
            reward_extra_infos_dict["final_answer_correctness"] = mas_r1_final_answer_correctness
        
        # Compute VERL-style pass@k metrics
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        
        # Add VERL pass@k metrics to metric_dict
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        # Compute pass@k metrics for powers of 2 using our custom function
        val_n = self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1)
        if val_n > 1:
            from mas_r1_reasoner.trainer.mas_r1_trainer_utils import compute_pass_at_k_metrics
            
            # Compute pass@k metrics
            pass_at_k_metrics = compute_pass_at_k_metrics(
                data_sources=data_sources,
                sample_inputs=sample_inputs,
                infos_dict=reward_extra_infos_dict,
                val_n=val_n,
                seed=42
            )
            
            # Add pass@k metrics to metric_dict
            for data_source, var2metric2val in pass_at_k_metrics.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    for metric_name, metric_val in metric2val.items():
                        if (var_name == core_var) and "pass@1" in metric_name:
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                        metric_dict[pfx] = metric_val

        # Evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # Add test_score metrics to existing metric_dict (don't overwrite pass@k metrics)
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # main_rank_print(f" metric_dict: {metric_dict}")


        # Add MAS-R1 specific metrics to metric_dict for automatic logging
        if mas_r1_overall_scores:
            avg_overall_score = sum(mas_r1_overall_scores) / len(mas_r1_overall_scores)
            avg_code_execution = sum(mas_r1_code_execution_success) / len(mas_r1_code_execution_success)
            avg_answer_correctness = sum(mas_r1_final_answer_correctness) / len(mas_r1_final_answer_correctness)
            
            metric_dict.update({
                'val/mas_r1/overall_score': avg_overall_score,
                'val/mas_r1/code_execution_success': avg_code_execution,
                'val/mas_r1/final_answer_correctness': avg_answer_correctness,
            })
            
            main_rank_print(f"MAS-R1 Validation Metrics:")
            main_rank_print(f"  - Overall Score: {avg_overall_score:.4f}")
            main_rank_print(f"  - Code Execution Success: {avg_code_execution:.4f}")
            main_rank_print(f"  - Final Answer Correctness: {avg_answer_correctness:.4f}")

        # Log VERL pass@k metrics
        main_rank_print(f"VERL Pass@k Metrics:")
        for key, value in metric_dict.items():
            if "val-core" in key or "val-aux" in key:
                main_rank_print(f"  - {key}: {value:.4f}")

        main_rank_print(f"\n{'='*80}")
        main_rank_print("MAS-R1 VALIDATION COMPLETED")
        main_rank_print(f"{'='*80}")
        main_rank_print(f"Total validation samples: {len(sample_inputs)}")
        main_rank_print(f"Average reward: {reward_tensor.mean().item():.4f}")
        main_rank_print(f"Metrics: {metric_dict}")
        main_rank_print(f"{'='*80}\n")

        return metric_dict


    def _save_checkpoint(self):
        """Override _save_checkpoint to also save generated code and responses"""
        save_dir = Path(self.config.trainer.default_local_dir)
        save_mas_r1_checkpoint(self, save_dir)

    def _save_checkpoint_parent(self):
        """Call parent class save_checkpoint method"""
        super()._save_checkpoint()



    def _create_dataloader(self, train_dataset=None, val_dataset=None, collate_fn=None, train_sampler=None):
        """
        Override _create_dataloader to use PreprocessedRLDataset for both training and validation.
        This ensures that both training and validation use the preprocessed data (/export/xgen-finance/meta_agent/mas_r1/scripts/data_prepare).
        """

        self.train_dataloader, self.val_dataloader, self.total_training_steps = create_mas_r1_dataloaders(
            self.config, self.tokenizer
        )


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from mas_r1_reasoner.agents.tracking import ReasonRLTracking
        from omegaconf import OmegaConf

        logger = ReasonRLTracking(project_name=self.config.trainer.project_name,
                                  experiment_name=self.config.trainer.experiment_name,
                                  default_backend=self.config.trainer.logger,
                                  config=OmegaConf.to_container(self.config, resolve=True))

        self.logger = logger


        pp.status("Config", f"Project: {self.config.trainer.project_name}, Experiment: {self.config.trainer.experiment_name}", "info")
        pp.status("Algorithm", f"Using {self.config.algorithm.adv_estimator} advantage estimator with Code Generation + Execution", "info")
        pp.status("Setup", f"Critic enabled: {self.use_critic}, Reference policy: {self.use_reference_policy}", "info")
        pp.status("Problem Type", f"Training on problem type: {self.problem_type}", "info")
        pp.status("Tree Architecture", f"Tree architecture enabled: {self.enable_tree_architecture}", "info")

        self.global_steps = 0

        # load checkpoint before doing anything
        pp.status("Checkpoint", "Loading checkpoint if available...", "info")
        self._load_checkpoint()



        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            pp.section_header("Initial Validation")
            pp.status("Validation", "Running initial validation...", "info")

            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')

            if get_global("global_eval_building_blocks"):
                pp.status("Validation", "Validation only mode, exiting", "success")
                return

            # Use enhanced MAS-R1 validation logging (same as training validation)
            log_mas_r1_validation_scores(val_metrics, self.global_steps)

            # Convert metrics to table format
            metrics_table = []
            for k, v in val_metrics.items():
                metrics_table.append([k, f"{v:.4f}" if isinstance(v, float) else v])

            pp.table(["Metric", "Value"], metrics_table, "Initial Validation Results")
            logger.log(data=val_metrics, step=self.global_steps)

            # Save val metrics to model path
            if self.config.trainer.get('log_to_model_path', False):
                import json
                import os
                # Convert numpy types to native Python types for JSON serialization
                val_metrics_converted = convert_numpy_types(val_metrics)
                with open(os.path.join(self.config.actor_rollout_ref.model.path, f'{self.problem_type}_metrics.json'), 'w') as f:
                    json.dump(val_metrics_converted, f)
                    
        # we start from step 1
        self.global_steps += 1
        total_steps = self.total_training_steps

        pp.section_header(f"Starting Code Generation + Execution Training - {self.problem_type}")
        pp.status("Training", f"Starting training for {self.config.trainer.total_epochs} epochs ({total_steps} steps)", "info")

        # Use the standard train_dataloader that was created in _create_dataloader
        # The dataloader is already created and uses PreprocessedRLDataset
        main_rank_print(f"Using preprocessed training dataloader with {len(self.train_dataloader)} batches")

        if self.total_training_steps == 0:
            pp.status("Validation", "Validation only mode, exiting", "success")
            return

        for epoch in range(self.config.trainer.total_epochs):
            pp.status("Epoch", f"Starting epoch {epoch+1}/{self.config.trainer.total_epochs}", "info")

            for batch_idx, batch_dict in enumerate(self.train_dataloader):

                main_rank_print(f"\n{'='*100}")
                main_rank_print(f"TRAINING STEP {self.global_steps} - Epoch {epoch+1}, Batch {batch_idx+1} - {self.problem_type}")
                main_rank_print(f"{'='*100}")

                metrics = {}
                timing_raw = {}

                # Initialize training data lists for wandb logging
                train_inputs = []
                train_outputs = []
                train_scores = []

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Step 2: Prepare batch for generation using shared function
                gen_batch = prepare_batch_for_generation(self, batch)

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = rollout_generation(self, gen_batch, is_validation=False)


                    if self.config.algorithm.adv_estimator == 'remax':
                        raise NotImplementedError("ReMax is not supported for MAS-R1")


                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    
            
                    # First repeat batch to align with expanded responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                     
                    # Finally union with mas_code_generation_output (which already has the right size from generation)
                    batch = batch.union(gen_batch_output)
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    
                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            main_rank_print(f"Reference policy log prob computed and added to batch")
                            main_rank_print(f"Batch keys after ref_log_prob union: {list(batch.batch.keys())}")

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):

                        main_rank_print(f"\n{'='*60}")
                        main_rank_print("COMPUTING REWARDS AND ADVANTAGES")
                        main_rank_print(f"{'='*60}")

                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            raise NotImplementedError("Reward model is not supported for MAS-R1")

                        pp.status("REWARD", f"Computing rewards for {self.problem_type}...", "info")
                        main_rank_print(f"\n{'='*60}")
                        main_rank_print("COMPUTING REWARDS")
                        main_rank_print(f"{'='*60}")

                        # Call reward function with batch data
                        reward_result = self.reward_fn(self, train_outputs, is_validation=False, input_data=batch, return_dict=True)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_info = reward_result["reward_extra_info"]
                        train_outputs = reward_result["sample_outputs"]
                        question_results = reward_result["question_results"]
                        # execution_results_dataproto = reward_result["final_output"]
                        
                        # Update batch to use expanded 32-size data for proper advantage computation
                        expanded_data = reward_result.get("expanded_data")
                        if expanded_data is not None:
                            main_rank_print(f"Updating batch from {len(batch)} to {len(expanded_data)} responses for advantage computation")
                            batch = expanded_data
                        else:
                            raise RuntimeError("No expanded_data returned from reward manager")
                        
                        main_rank_print(f"Rewards computed successfully!")
                        main_rank_print(f"Reward tensor shape: {reward_tensor.shape}")
                        main_rank_print(f"Batch size after update: {len(batch)}")
                        main_rank_print(f"Average reward: {reward_tensor.mean().item():.4f}")
                        main_rank_print(f"{'='*60}\n")
                        pp.status("REWARD", f"Rewards computed for {self.problem_type}", "success")


                        # Collect training samples for logging (after all batch modifications)
                        if hasattr(self.config.trainer, 'train_generations_to_log_to_wandb') and self.config.trainer.train_generations_to_log_to_wandb > 0:
                            # Store execution results (final answers) for wandb logging
                            execution_results = batch.non_tensor_batch.get('execution_results', [])
                            for result in execution_results:
                                # Use the execution result (final answer) instead of generated code
                                execution_output = result.get('result', 'N/A')
                                train_outputs.append(execution_output)

                            # Store original inputs for wandb logging (after batch modifications)
                            input_ids = batch.batch['input_ids']
                            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                            train_inputs.extend(input_texts)


                        # Collect generated code for saving
                        current_step_code, accumulated_code = collect_generated_code_from_dict(
                            question_results=question_results,
                            global_steps=self.global_steps,
                            save_generated_code=self.save_generated_code,
                            max_accumulated_steps=self.max_accumulated_steps,
                            logger=self.logger if hasattr(self, 'logger') and self.logger is not None else None
                        )

                        # Store the results in the trainer instance
                        if current_step_code is not None:
                            self._current_step_generated_code = current_step_code
                        if accumulated_code is not None:
                            self._accumulated_generated_code = accumulated_code


                        # Store scores for training logging
                        if hasattr(self.config.trainer, 'train_generations_to_log_to_wandb') and self.config.trainer.train_generations_to_log_to_wandb > 0:
                            scores = reward_tensor.sum(-1).cpu().tolist()
                            train_scores.extend(scores)

                            # Log training samples to wandb (similar to validation)
                            self.training_table = maybe_log_train_generations_to_wandb(
                                inputs=train_inputs,
                                outputs=train_outputs,
                                scores=train_scores,
                                config=self.config,
                                global_steps=self.global_steps,
                                training_table=getattr(self, 'training_table', None),
                                train_batch=batch,
                                reward_extra_info=reward_extra_info  # Now we have reward_extra_info for training too
                            )

                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # Check if mean@k is enabled and we have the necessary configuration
                        if (self.mas_r1_config.get('full_sampling_mean_at_k', False) and 
                            hasattr(batch, 'non_tensor_batch') and 
                            'mean_at_k_enabled' in batch.non_tensor_batch and # Check for our flag
                            batch.non_tensor_batch.get('mean_at_k_enabled', False)): # Ensure it's true
                            
                            # Use our custom mean@k advantage estimator
                            from mas_r1_reasoner.trainer.utils.advantages import MeanAtKAdvantageEstimator
                            estimator = MeanAtKAdvantageEstimator()
                            
                            # Compute advantages using our custom estimator
                            advantages = estimator.compute_advantage(
                                batch.batch['token_level_rewards'], 
                                batch
                            )
                            
                            # CRITICAL: Store advantages and compute returns in the format VERL expects
                            batch.batch['advantages'] = advantages
                            
                            # Compute returns for proper VERL integration
                            # For GRPO compatibility: returns = advantages (same as VERL's GRPO)
                            returns = advantages # Note: GRPO specific
                            batch.batch['returns'] = returns
                            
                            main_rank_print(f"🎯 Using our custom mean@k advantage estimator!")
                            main_rank_print(f"   Advantage shape: {advantages.shape}")
                            main_rank_print(f"   Returns shape: {returns.shape}")
                            main_rank_print(f"   Advantages: {advantages[:, 0].tolist()}")  # Show first token of each response
                            main_rank_print(f"   Returns: {returns[:, 0].tolist()}")  # Show first token of each response
                            main_rank_print(f"   Mean@k enabled: {self.mas_r1_config.get('full_sampling_mean_at_k', False)}")
                            main_rank_print(f"   ✅ Advantages and returns properly formatted for VERL GRPO compatibility!")
                            main_rank_print(f"   📝 Note: Returns = Advantages (GRPO behavior)")
                            
                        else:
                            # Fall back to VERL's default advantage computation
                            batch = compute_advantage(batch,
                                                      adv_estimator=self.config.algorithm.adv_estimator,
                                                      gamma=self.config.algorithm.gamma,
                                                      lam=self.config.algorithm.lam,
                                                      num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return