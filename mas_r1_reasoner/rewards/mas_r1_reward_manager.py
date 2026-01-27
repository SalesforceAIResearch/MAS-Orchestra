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
MAS-R1 Reward Manager that inherits from NaiveRewardManager.
This provides a clean interface for MAS-R1 reward computation with minimal modifications.
"""
import re
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.rewards.utils.compute_score import create_mas_r1_compute_score
from mas_r1_reasoner.rewards.utils.hierarchical_reward import sub_task_sub_agent_to_hierarchical_reward_separate, sub_task_sub_agent_to_hierarchical_reward_unified
from mas_r1_reasoner.trainer.utils.helper import get_bool_config
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from mas_r1_reasoner.rewards.utils.execution import (
    execute_codes_and_store_results,
)
from mas_r1_reasoner.rewards.utils.extraction_tree import generate_and_extract_codes_tree
from mas_r1_reasoner.rewards.utils.extraction_sequential import generate_and_extract_codes_sequential
from mas_r1_reasoner.rewards.utils.extraction import (
    extract_questions_and_ground_truth,
    generate_and_extract_codes,
    generate_and_extract_codes_with_tree_validation,
    generate_and_extract_codes_with_sequential_validation
)
from mas_r1_reasoner.rewards.utils.merge_same_sub_task import merge_identical_sub_tasks
from mas_r1_reasoner.agents.shared_vars import get_global
from mas_r1_reasoner.rewards.utils.reformat import (
    from_question_results_to_execution_results,
)
from mas_r1_reasoner.rewards.utils.mean_at_k_reward import (
    apply_mean_at_k_rewards
)
async def _process_single_question_reward_async(args: Tuple[int, object, object, object, object, object, object]) -> Tuple[int, float, dict]:
    """
    Process a single question's reward computation in a thread-safe manner.
    
    Args:
        args: Tuple of (question_index, data_item, tokenizer, compute_score, reward_fn_key, num_examine)
        
    Returns:
        Tuple of (question_index, reward_score, reward_extra_info)
    """
    question_index, data_item, tokenizer, compute_score, reward_fn_key, num_examine = args
    
    try:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[reward_fn_key]
        
        # Get extra_info from the data item's non_tensor_batch
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        score = await compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        # Extract detailed scores from extra_info for wandb logging
        reward_extra_info = {"score": score}
        
        # Check if detailed scores were stored by compute_score
        if extra_info and 'mas_r1_scores' in extra_info:
            mas_r1_scores = extra_info['mas_r1_scores']
            # Map the detailed scores to the keys expected by wandb logging
            if 'code_execution_success' in mas_r1_scores:
                reward_extra_info['code_execution_success'] = mas_r1_scores['code_execution_success']
            if 'final_answer_correctness' in mas_r1_scores:
                reward_extra_info['final_answer_correctness'] = mas_r1_scores['final_answer_correctness']
            if 'combined_reward' in mas_r1_scores:
                reward_extra_info['combined_reward'] = mas_r1_scores['combined_reward']
            if 'predicted_answer' in mas_r1_scores:
                reward_extra_info['predicted_answer'] = mas_r1_scores['predicted_answer']
        
        return question_index, score, reward_extra_info
        
    except Exception as e:
        print(f"Error processing question {question_index}: {e}")
        return question_index, 0.0, {"score": 0.0, "error": str(e)}


def _process_single_hierarchical_reward(args: Tuple[int, object, object, object, object]) -> Tuple[int, float, dict]:
    """
    Process a single response's hierarchical reward computation in a thread-safe manner.
    
    Args:
        args: Tuple of (response_index, result, compute_score, mock_score)
        
    Returns:
        Tuple of (response_index, sub_agent_reward, reward_info)
    """
    response_index, result, compute_score, mock_score = args
    
    try:
        if not result:
            return response_index, 0.0, {'error': 'No result data'}
            
        # Compute sub-agent reward using compute_score_func with real execution data
        ground_truth = result.get('ground_truth', '')
        code = result.get('code', '')
        
        # Use mock_score configuration to decide whether to compute real score or use random
        if mock_score:
            # Use deterministic random scores for debugging/development
            import random
            random.seed(response_index)  # Deterministic for debugging
            sub_agent_reward = random.choice([0.0, 1.0])
        else:
            # Use compute_score_func directly with real execution data
            sub_agent_score = compute_score(
                data_source="math",
                solution_str=code,
                ground_truth=ground_truth,
                extra_info={'execution_result': result}
            )
            sub_agent_reward = float(sub_agent_score)
        
        # Prepare reward info
        reward_info = {
            'sub_agent_reward': sub_agent_reward,
            'level': result.get('level', 1),
            'ground_truth': ground_truth,
            'code': code
        }
        
        return response_index, sub_agent_reward, reward_info
        
    except Exception as e:
        print(f"Error processing hierarchical reward for response {response_index}: {e}")
        return response_index, 0.0, {'error': str(e), 'sub_agent_reward': 0.0, 'level': 1}


class MASR1RewardManager(NaiveRewardManager):
    """MAS-R1 Reward Manager that inherits from NaiveRewardManager."""
    
    def __init__(self, tokenizer, num_examine=5, config=None):
        # Get YAML config values from passed config
        if config and hasattr(config, 'azr') and hasattr(config.azr, 'mas_r1'):
            try:
                code_execution_weight = config.azr.mas_r1.get('execution_success_weight', 0.5)
                final_answer_weight = config.azr.mas_r1.get('final_answer_weight', 0.5)
                main_rank_print(f"✅ Using YAML config: execution_success_weight={code_execution_weight}, final_answer_weight={final_answer_weight}")
            except Exception as e:
                raise RuntimeError(f"Failed to read config values: {e}")
        else:
            raise RuntimeError("No config provided or missing azr.mas_r1 section")
        
        # Store config for later use
        self.config = config
        
        # Get mock_score from agent_config using helper function
        self.mock_score = get_bool_config(config, 'azr.mas_r1.mock_score', default_value=False, required=False)
        main_rank_print(f"✅ Mock score configuration: {self.mock_score}")
        
        # Get mock_sub_task_sub_agent from mas_r1 config using helper function
        self.mock_sub_task_sub_agent = get_bool_config(config, 'azr.mas_r1.mock_sub_task_sub_agent', default_value=False, required=False)
        main_rank_print(f"✅ Mock sub-task/sub-agent configuration: {self.mock_sub_task_sub_agent}")
        
        # Get diff_based_reward from mas_r1 config using helper function
        self.diff_based_reward = get_bool_config(config, 'azr.mas_r1.diff_based_reward', default_value=False, required=True)
        main_rank_print(f"✅ Diff-based reward configuration: {self.diff_based_reward}")
        
        # Get unified_group from mas_r1 config using helper function
        self.unified_group = get_bool_config(config, 'azr.mas_r1.unified_group', default_value=False, required=True)
        main_rank_print(f"✅ Unified group configuration: {self.unified_group}")

        self.expansive_group = get_bool_config(config, 'azr.mas_r1.expansive_group', default_value=False, required=True)
        main_rank_print(f"✅ expansive_group group configuration: {self.expansive_group}")

        self.disable_hierarchical_reward = get_bool_config(config, 'azr.mas_r1.disable_hierarchical_reward', default_value=False, required=True)
        main_rank_print(f"✅ disable_hierarchical_reward configuration: {self.disable_hierarchical_reward}")


        # Get mean@k reward configuration
        self.full_sampling_mean_at_k = get_bool_config(config, 'azr.mas_r1.full_sampling_mean_at_k', default_value=False, required=False)
        if self.full_sampling_mean_at_k:
            self.mean_at_k_size = config.azr.mas_r1.get('mean_at_k_size', 4)
            main_rank_print(f"✅ Mean@k reward configuration: enabled={self.full_sampling_mean_at_k}, size={self.mean_at_k_size}")
        else:
            main_rank_print(f"✅ Mean@k reward configuration: disabled")

        # Validate that only one UID planning strategy is enabled
        enabled_strategies = sum([self.diff_based_reward, self.unified_group, self.expansive_group])
        if enabled_strategies != 1:
            error_msg = f"Exactly one UID planning strategy should be enabled, but got {enabled_strategies} enabled strategies: diff_based_reward={self.diff_based_reward}, unified_group={self.unified_group}, expansive_group={self.expansive_group}"
            main_rank_print(f"❌ {error_msg}")
            # raise ValueError(error_msg)
        
        # Log which strategy is enabled
        if self.diff_based_reward:
            main_rank_print(f"🎯 UID Planning Strategy: DIFF_BASED_REWARD (Level 2 responses from same question share UID)")
        elif self.unified_group:
            main_rank_print(f"🎯 UID Planning Strategy: UNIFIED_GROUP (Level 2 responses inherit parent UIDs)")
        elif self.expansive_group:
            main_rank_print(f"🎯 UID Planning Strategy: EXPANSIVE_GROUP (Level 2 responses get new group UIDs)")


        # Create MAS-R1 compute_score function with configuration
        mas_r1_config = {
            'code_execution_weight': code_execution_weight,
            'final_answer_weight': final_answer_weight,
            'answer_pattern': r"(?i)\n\nAnswer\s*:\s*([^\n]+)",
            'default_answer_on_failure': "Unavailable"
        }
        compute_score_func = create_mas_r1_compute_score(mas_r1_config)
        
        # Initialize parent NaiveRewardManager with our compute_score function
        super().__init__(tokenizer=tokenizer, num_examine=num_examine, compute_score=compute_score_func)
              
        main_rank_print(f"MAS-R1 Reward Manager initialized with MAS-R1 compute_score (mock_score={self.mock_score}, mock_sub_task_sub_agent={self.mock_sub_task_sub_agent}, diff_based_reward={self.diff_based_reward}, unified_group={self.unified_group}, expansive_group={self.expansive_group}, full_sampling_mean_at_k={self.full_sampling_mean_at_k})")
    
    def __call__(self, trainer_instance, sample_outputs, is_validation, input_data: DataProto, return_dict: bool = False):
        """Override to handle our MAS-R1 data format from create_dataproto_from_question_results."""
        
        #--------------------------------

        # Step 1: Extract questions and ground truth using shared function
        question_results = extract_questions_and_ground_truth(trainer_instance, input_data)

        # Step 2: Generate and extract codes using shared function
        # Choose generation function based on tree architecture configuration
        
        # validation do not need to do structure
        if trainer_instance.architecture_only_sequential:
            main_rank_print("Generating and extracting codes with sequential architecture")
            if not is_validation:
                mas_code_generation_output, question_results = generate_and_extract_codes_sequential(
                    trainer_instance, self, question_results, input_data
                )
            else:
                mas_code_generation_output, question_results = generate_and_extract_codes_with_sequential_validation(
                    trainer_instance, self, question_results, input_data
                )

        elif trainer_instance.enable_tree_architecture:
            if not is_validation:
                main_rank_print("Generating and extracting codes with two-level tree architecture")
                mas_code_generation_output, question_results = generate_and_extract_codes_tree(
                    trainer_instance, self, question_results, input_data
                )
            else:
                main_rank_print("Generating and extracting codes with tree architecture validation")
                mas_code_generation_output, question_results = generate_and_extract_codes_with_tree_validation(
                    trainer_instance, question_results, input_data
                )
        else:
            main_rank_print("Generating and extracting codes with one-level tree architecture")
            mas_code_generation_output, question_results = generate_and_extract_codes(
                trainer_instance, question_results, input_data, is_validation=is_validation
            )

        # Step 4: Execute codes and store results using shared function
        question_results = execute_codes_and_store_results(trainer_instance, question_results, is_validation=is_validation)

        # Step 5: Convert dictionary back to DataProto format for VERL
        # Create final output with execution results
        final_output = from_question_results_to_execution_results(question_results, mas_code_generation_output)

        # Store execution results (final answers) for wandb logging instead of generated code
        execution_results = final_output.non_tensor_batch.get('execution_results', [])
        execution_outputs = []
        for result in execution_results:
            # Use the execution result (final answer) instead of generated code
            execution_output = result.get('result', 'N/A')
            execution_outputs.append(execution_output)
        sample_outputs.extend(execution_outputs)
        #--------------------------------
        # USE FINAL_OUTPUT DIRECTLY - IT NOW CONTAINS ALL TENSOR FIELDS
        # The from_question_results_to_execution_results function has been fixed to include
        # all necessary tensor fields (input_ids, attention_mask, responses) from mas_code_generation_output
        # No need for union operations anymore!
        
        main_rank_print(f"Using final_output directly (now contains all tensor fields)...")
        main_rank_print(f"  - input_data size: {len(input_data)} responses")
        main_rank_print(f"  - mas_code_generation_output size: {len(mas_code_generation_output)} responses")
        main_rank_print(f"  - final_output size: {len(final_output)} responses")
        
        # final_output already contains all the expanded data with tensor fields
        # No need for union operations that require same batch sizes
        data = final_output
        
        main_rank_print(f"Data structure simplified:")
        main_rank_print(f"  - Final batch size: {len(data)} responses")
        main_rank_print(f"  - Tensor fields: {list(data.batch.keys())}")
        main_rank_print(f"  - Non-tensor fields: {list(data.non_tensor_batch.keys())}")
        
        # Validate that we have the required tensor fields
        required_tensor_fields = ['input_ids', 'attention_mask', 'responses']
        for field in required_tensor_fields:
            if field not in data.batch:
                raise RuntimeError(f"Missing required tensor field '{field}' in final_output")
            main_rank_print(f"  - {field} shape: {data.batch[field].shape}")
        
        # Check that execution_results has the right size
        execution_results = data.non_tensor_batch.get('execution_results', [])
        if len(execution_results) != len(data):
            raise RuntimeError(f"Execution results size ({len(execution_results)}) doesn't match batch size ({len(data)})")
        main_rank_print(f"  - Execution results: {len(execution_results)} items")
        
        main_rank_print(f"✅ Data structure validation passed")

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']
        
        # Validate that we have the required data
        if 'responses' not in data.batch:
            raise RuntimeError("DataProto must contain 'responses' in batch")
        
        if not hasattr(data, 'non_tensor_batch') or not data.non_tensor_batch:
            raise RuntimeError("DataProto must have non_tensor_batch")
        
        if 'reward_model' not in data.non_tensor_batch:
            raise RuntimeError("non_tensor_batch must contain 'reward_model'")

        if not (hasattr(data, 'meta_info') and data.meta_info):
            # If no meta_info, this indicates a serious data flow issue
            # We should never reach this point in our implementation
            raise RuntimeError("DataProto missing meta_info")

        # PARALLEL REWARD COMPUTATION: Process questions in parallel instead of calling parent
        main_rank_print(f"\n{'='*60}")
        main_rank_print("PARALLEL REWARD COMPUTATION")
        main_rank_print(f"{'='*60}")
        
        # Extract execution results and stats directly from data
        execution_results = data.non_tensor_batch.get('execution_results', [])
        execution_stats = data.non_tensor_batch.get('execution_stats', [])
        
        main_rank_print(f"Reward Manager: Execution results size: {len(execution_results)}")
        main_rank_print(f"Reward Manager: Batch size: {len(data)}")
        
        # Validate ground truth alignment (optional validation)
        self._validate_ground_truth_alignment(data, execution_results, execution_stats)
        
        if trainer_instance.enable_tree_architecture and not is_validation:
            raise NotImplementedError("Hierarchical reward computation is not implemented yet")
            # main_rank_print("🌳 Tree architecture enabled - using hierarchical reward computation")
            # reward_tensor, reward_extra_info = self._compute_hierarchical_reward(data, execution_results)
        else:
            main_rank_print("📊 Standard architecture - using normal reward computation")
            # Call _compute_normal_reward directly - semaphore is now inside the method
            import asyncio
            reward_tensor, reward_extra_info = asyncio.run(self._compute_normal_reward(data, execution_results, execution_stats))
        
        main_rank_print(f"Reward computation completed for {len(data)} questions")
        main_rank_print(f"{'='*60}\n")
        
        # Validate reward_extra_info to ensure all keys have correct length
        batch_size = len(data)  # Total responses: dynamic (Level 1 + Level 2)
        
        # Extract actual Level 1 count from the data instead of hard-coding
        level_1_count = 0
        for i, result in enumerate(execution_results):
            if result and isinstance(result, dict) and result.get('level') == 1:
                level_1_count += 1
        
        main_rank_print(f"Validating reward_extra_info for {batch_size} total responses ({level_1_count} Level 1 + {batch_size - level_1_count} Level 2)...")
        
        # Only validate the fields that follow Level 1 structure
        required_fields = ['code_execution_success', 'combined_reward', 'final_answer_correctness', 'score']
        
        for key in required_fields:
            if key not in reward_extra_info:
                error_msg = f"ERROR: Missing required field '{key}' in reward_extra_info"
                main_rank_print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            current_length = len(reward_extra_info[key]) if reward_extra_info[key] else 0
            
            # All fields should have batch_size length (dynamic total responses)
            expected_length = batch_size
            # Remove any None values that might have occurred due to processing errors
            reward_extra_info[key] = [val for val in reward_extra_info[key] if val is not None]

            if len(reward_extra_info[key]) != expected_length:
                error_msg = f"ERROR: Key '{key}' has {len(reward_extra_info[key])} non-None values, expected {expected_length}"
                main_rank_print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            else:
                main_rank_print(f"✅ Key '{key}': {len(reward_extra_info[key])} values)")
    
        main_rank_print(f"✓ All required reward_extra_info keys validated successfully")
        
        # Pad the data to ensure it's divisible by world_size for subsequent training steps
        # This prevents chunking errors in update_actor, update_critic, etc.
        if not is_validation:  # Only pad for training data
            from verl.protocol import pad_dataproto_to_divisor
            world_size = trainer_instance.actor_rollout_wg.world_size
            data_padded, pad_size = pad_dataproto_to_divisor(data, world_size)
            
            # Pad reward tensor with zeros for the padding samples (at the END, consistent with VERL)
            reward_tensor_padded = torch.zeros((len(data_padded), reward_tensor.size(1)), dtype=reward_tensor.dtype)
            reward_tensor_padded[:len(data), :] = reward_tensor  # Original data at the beginning
            # Padding samples (zeros) are automatically at the end
            main_rank_print(f"   - Padded tensor size: {len(data_padded)} vs. {len(reward_tensor_padded)}")
            # main_rank_print(f"   - Padding strategy: Original data at beginning, zeros at end (consistent with VERL)")

        
            # Use the padded data for training
            data_for_training = data_padded
            reward_tensor_for_training = reward_tensor_padded
        else:
            # For validation, no padding needed
            data_for_training = data
            reward_tensor_for_training = reward_tensor
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor_for_training,
                "reward_extra_info": reward_extra_info,
                "sample_outputs": sample_outputs,
                "question_results": question_results,
                # "final_output": final_output,
                "expanded_data": data_for_training  # Return the padded data for trainer to update input_data
            }
        else:
            return reward_tensor

    def _validate_ground_truth_alignment(self, data: DataProto, execution_results: list, execution_stats: list):
        """
        Validate ground truth alignment between reward_model and execution_stats.
        This follows the exact pattern from mas_r1_reward_manager_old.py
        """
        main_rank_print(f"\n{'='*60}")
        main_rank_print("REWARD MANAGER: VALIDATING GROUND TRUTH ALIGNMENT")
        main_rank_print(f"{'='*60}")
        
        reward_model_ground_truths = []
        execution_stats_ground_truths = []
        
        # Extract ground truths from reward_model
        for i in range(len(data)):
            try:
                reward_ground_truth = data.non_tensor_batch['reward_model'][i]['ground_truth']
                reward_model_ground_truths.append(reward_ground_truth)
            except (IndexError, KeyError, TypeError) as e:
                main_rank_print(f"WARNING: Could not extract reward_model ground truth for sample {i}: {e}")
                reward_model_ground_truths.append('ERROR')
        
        # Extract ground truths from execution_stats
        for i in range(len(data)):
            try:
                if i < len(execution_results) and execution_results[i]:
                    execution_ground_truth = execution_results[i].get('ground_truth', '')
                else:
                    execution_ground_truth = ''
                execution_stats_ground_truths.append(execution_ground_truth)
            except (IndexError, KeyError, TypeError) as e:
                main_rank_print(f"WARNING: Could not extract execution_stats ground truth for sample {i}: {e}")
                execution_stats_ground_truths.append('ERROR')
        
        # Compare ground truths
        mismatches_found = []
        for i, (reward_gt, execution_gt) in enumerate(zip(reward_model_ground_truths, execution_stats_ground_truths)):
            # main_rank_print(f"Sample {i}:")
            # main_rank_print(f"  Reward model ground truth: '{reward_gt}'")
            # main_rank_print(f"  Execution stats ground truth: '{execution_gt}'")
            
            if reward_gt != execution_gt:
                raise RuntimeError(f"Ground truth mismatch found in MASR1RewardManager! Reward model ground truth: '{reward_gt}' Execution stats ground truth: '{execution_gt}'")
            # else:
            #     main_rank_print(f"  ✓ Ground truths match")
        
        if mismatches_found:
            error_msg = f"GROUND TRUTH MISMATCHES DETECTED in MASR1RewardManager!"
            error_msg += f"\nTotal mismatches: {len(mismatches_found)}"
            for mismatch in mismatches_found:
                error_msg += f"\nSample {mismatch['sample_index']}:"
                error_msg += f"\n  Reward model: '{mismatch['reward_ground_truth']}'"
                error_msg += f"\n  Execution stats: '{mismatch['execution_ground_truth']}'"
            main_rank_print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"✓ All ground truths aligned ({len(reward_model_ground_truths)} samples)")
        main_rank_print(f"{'='*60}\n")

    async def _compute_normal_reward(self, data: DataProto, execution_results: list, execution_stats: list) -> Tuple[torch.Tensor, Dict]:
        """
        Compute normal rewards for standard architecture (non-tree).
        Uses parallel processing to compute rewards for each response independently.
        """
        main_rank_print(f"Computing normal rewards for {len(data)} responses")
        
        # Create a new batch with the expected format
        adapted_batch = {}
        
        # Create dummy prompts and attention_mask since we only have responses
        batch_size = data.batch['responses'].shape[0]
        adapted_batch['prompts'] = torch.zeros((batch_size, 1), dtype=torch.long)
        adapted_batch['responses'] = data.batch['responses']
        adapted_batch['attention_mask'] = torch.ones((batch_size, 1), dtype=torch.long)
        
        # Create adapted DataProto with proper structure
        adapted_data = DataProto.from_single_dict(adapted_batch)
        
        # Prepare non_tensor_batch with required fields
        adapted_data.non_tensor_batch = {
            'reward_model': data.non_tensor_batch.get('reward_model', []),
            'data_source': ['math'] * batch_size,  # Required by NaiveRewardManager
            'extra_info': []  # Will be populated with execution results
        }
        
        # Copy execution results from meta_info to extra_info for NaiveRewardManager
        adapted_data.meta_info = data.meta_info
        # Populate extra_info with execution results for each sample
        execution_results = data.non_tensor_batch.get('execution_results', [])
        execution_stats = data.non_tensor_batch.get('execution_stats', [])
        
        # NEW: With {question}_{response} structure, we already have individual execution stats per response
        # No need for np.repeat since batch size already matches the number of responses
        main_rank_print(f"Reward Manager: Execution results size: {len(execution_results)}")
        main_rank_print(f"Reward Manager: Execution stats size: {len(execution_stats)}")
        main_rank_print(f"Reward Manager: Batch size: {len(adapted_data)}")
        
        # Validate that sizes match
        if len(execution_results) != len(adapted_data):
            raise RuntimeError(f"Execution results size ({len(execution_results)}) doesn't match batch size ({len(adapted_data)})")
        if len(execution_stats) != len(adapted_data):
            raise RuntimeError(f"Execution stats size ({len(execution_stats)}) doesn't match batch size ({len(adapted_data)})")
        
        # Populate extra_info with execution results and statistics (no repetition needed)
        for i, result in enumerate(execution_results):
            extra_info = {
                'execution_result': result
            }
            
            # Add execution statistics if available
            if i < len(execution_stats) and execution_stats[i]:
                extra_info['execution_stats'] = execution_stats[i]
            
            # Add validation flag from execution result
            if isinstance(result, dict) and 'is_validation' in result:
                extra_info['is_validation'] = result['is_validation']
            
            adapted_data.non_tensor_batch['extra_info'].append(extra_info)
        
        # PARALLEL REWARD COMPUTATION: Process questions in parallel instead of calling parent
        main_rank_print(f"\n{'='*60}")
        main_rank_print("PARALLEL REWARD COMPUTATION")
        main_rank_print(f"{'='*60}")
        
        batch_size = len(adapted_data)
        reward_tensor = torch.zeros_like(adapted_data.batch["responses"], dtype=torch.float32)
        reward_extra_info = {
            'code_execution_success': [None] * batch_size,
            'final_answer_correctness': [None] * batch_size,
            'combined_reward': [None] * batch_size,
            'predicted_answer': [None] * batch_size,
            'score': [None] * batch_size
        }
        
        # Prepare arguments for parallel processing
        args_list = []
        for i in range(batch_size):
            data_item = adapted_data[i]
            args = (i, data_item, self.tokenizer, self.compute_score, self.reward_fn_key, self.num_examine)
            args_list.append(args)
        
        # Use asyncio for parallel processing with semaphore control
        import asyncio
        from mas_r1_reasoner.agents.shared_vars import get_global
        
        # Create semaphore to limit concurrent question processing
        max_concurrent = get_global("global_max_concurrent")
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_question_with_semaphore(args):
            async with semaphore:
                return await _process_single_question_reward_async(args)
        
        # Process all questions concurrently with semaphore control
        tasks = [process_question_with_semaphore(args) for args in args_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results as they complete (true parallelism) and store them immediately
        for i, result in enumerate(results):
            try:
                
                question_index, reward, question_extra_info = result
                
                # Store reward in tensor immediately
                data_item = adapted_data[question_index]
                
                # Get prompt length from prompts tensor
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                
                # Calculate valid_response_length correctly: only sum the response portion of attention_mask
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                
                reward_tensor[question_index, valid_response_length - 1] = reward
                
                # Store extra info immediately at the correct index (like reward_tensor)
                reward_extra_info['code_execution_success'][question_index] = question_extra_info['code_execution_success']
                reward_extra_info['final_answer_correctness'][question_index] = question_extra_info['final_answer_correctness']
                reward_extra_info['combined_reward'][question_index] = question_extra_info['combined_reward']
                reward_extra_info['predicted_answer'][question_index] = question_extra_info['predicted_answer']
                
                # Store the score - this was missing!
                reward_extra_info['score'][question_index] = question_extra_info.get('score', reward)
                
            except Exception as e:
                raise RuntimeError(f"Error processing question result: {e}")
        
        main_rank_print(f"Normal reward computation completed for {batch_size} questions")
        return reward_tensor, reward_extra_info


def setup_reward_manager(tokenizer, num_examine, config):
    """Set up the MAS-R1 reward manager."""
    return MASR1RewardManager(tokenizer=tokenizer, num_examine=num_examine, config=config)