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
Direct Reward Manager that inherits from NaiveRewardManager.
This provides a simplified interface for direct reward computation where input_data
already contains the final answer, eliminating the need for extraction/execution steps.
"""
import re
import torch
from verl import DataProto
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.rewards.mas_r1_reward_manager import MASR1RewardManager


class DirectRewardManager(MASR1RewardManager):
    """Direct Reward Manager that inherits from NaiveRewardManager.
    
    This manager works with input_data that already contains the final answer,
    eliminating the need for extraction and execution steps.
    """
    
    def __init__(self, tokenizer, num_examine=5, config=None):

        # Initialize parent NaiveRewardManager with our compute_score function
        super().__init__(tokenizer=tokenizer, num_examine=num_examine, config=config)
        main_rank_print(f"Direct Reward Manager initialized with MAS-R1 compute_score)")
    
    def __call__(self, trainer_instance, sample_outputs, is_validation, input_data: DataProto, return_dict: bool = False):
        """Override to handle direct reward computation from input_data containing final answers."""
        
        main_rank_print(f"Direct Reward Manager: Processing {len(input_data)} responses with final answers")
        
        # Use input_data directly since it already contains the final answers
        data = input_data
        
        # Store final answers for wandb logging
        if hasattr(data, 'non_tensor_batch') and 'final_answers' in data.non_tensor_batch:
            final_answers = data.non_tensor_batch.get('final_answers', [])
            sample_outputs.extend(final_answers)
        else:
            # If no final_answers field, use responses as sample outputs
            for i in range(len(data)):
                response_str = self.tokenizer.decode(data.batch['responses'][i], skip_special_tokens=True)
                sample_outputs.append(response_str)
        
        main_rank_print(f"Data structure:")
        main_rank_print(f"  - Batch size: {len(data)} responses")
        main_rank_print(f"  - Tensor fields: {list(data.batch.keys())}")
        main_rank_print(f"  - Non-tensor fields: {list(data.non_tensor_batch.keys())}")
        
        # Validate that we have the required tensor fields
        required_tensor_fields = ['input_ids', 'attention_mask', 'responses']
        for field in required_tensor_fields:
            if field not in data.batch:
                raise RuntimeError(f"Missing required tensor field '{field}' in input_data")
            main_rank_print(f"  - {field} shape: {data.batch[field].shape}")
        
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

        # DIRECT REWARD COMPUTATION: Process responses directly without extraction/execution
        main_rank_print(f"\n{'='*60}")
        main_rank_print("DIRECT REWARD COMPUTATION")
        main_rank_print(f"{'='*60}")
        
        # Check if source_file is present in input data (required for per-file validation metrics)
        if 'source_file' not in data.non_tensor_batch:
            raise RuntimeError(
                "ERROR: No 'source_file' information found in input data.\n"
                "This is required for per-file validation metrics and tables.\n"
                "Make sure your validation data includes source_file metadata from the dataset processing."
            )
        
        source_files = data.non_tensor_batch['source_file']
        unique_files = set(source_files[i] for i in range(len(source_files)) if source_files[i] is not None)
        main_rank_print(f"📁 Found source_file information: {len(unique_files)} unique files")
        for f in list(unique_files)[:3]:  # Show first 3
            import os
            main_rank_print(f"  - {os.path.basename(f)}")
        
        # Create execution results from input_data for compatibility with existing reward computation
        execution_results = []
        execution_stats = []
        reward_model_list = []
        
        for i in range(len(data)):
            # Extract question using the processor
            try:
                question = trainer_instance.processor.extract_math_question(data, i)
            except Exception as e:
                raise RuntimeError(f"Warning: Could not extract question for sample {i}: {e}")
                
            
            # Extract final answer from input_data
            if hasattr(data, 'non_tensor_batch') and 'final_answers' in data.non_tensor_batch:
                final_answer = data.non_tensor_batch['final_answers'][i] if i < len(data.non_tensor_batch['final_answers']) else ""
            else:
                # Decode response to get final answer
                response_str = self.tokenizer.decode(data.batch['responses'][i], skip_special_tokens=True)
                final_answer = response_str
            
            # Get ground truth
            ground_truth = data.non_tensor_batch['reward_model'][i]['ground_truth']
            
            # Get source_file if available (for tracking which validation file)
            source_file = data.non_tensor_batch.get('source_file', [None] * len(data))[i]
            
            # Create execution result structure
            execution_result = {
                'result': final_answer,
                'error': final_answer, # use error as final_answer so that we can use math_scorer.extract_solution in compute_score.py
                'ground_truth': ground_truth,
                'question': question,  # Add question to execution result
                'level': 1,  # Direct reward manager treats all as level 1
                'response_idx': i,
                'is_validation': is_validation,
                'source_file': source_file  # Track which file this sample came from
            }
            execution_results.append(execution_result)
            
            # Create execution stats
            execution_stats.append({
                'ground_truth': ground_truth,
                'final_answer': final_answer,
                'question': question  # Add question to execution stats
            })
            
            # Create reward_model entry for this response (required by MAS-R1 infrastructure)
            reward_model_list.append({
                'ground_truth': ground_truth
            })
        
        main_rank_print(f"Direct Reward Manager: Execution results size: {len(execution_results)}")
        main_rank_print(f"Direct Reward Manager: Batch size: {len(data)}")
        
        # Store execution results in data.non_tensor_batch for _compute_normal_reward to access
        if not hasattr(data, 'non_tensor_batch'):
            data.non_tensor_batch = {}
        
        # Preserve source_file before overwriting non_tensor_batch arrays
        source_files_array = data.non_tensor_batch.get('source_file', None)
        
        # Convert to numpy arrays as expected by VERL protocol
        import numpy as np
        data.non_tensor_batch['execution_results'] = np.array(execution_results, dtype=object)
        data.non_tensor_batch['execution_stats'] = np.array(execution_stats, dtype=object)
        data.non_tensor_batch['reward_model'] = np.array(reward_model_list, dtype=object)
        
        # Restore source_file array (needed for per-file validation metrics/tables)
        if source_files_array is not None:
            data.non_tensor_batch['source_file'] = source_files_array
            main_rank_print(f"✓ Preserved source_file information for {len(execution_results)} samples")
        
        # Validate ground truth alignment
        self._validate_ground_truth_alignment(data, execution_results, execution_stats)
        
        # Use normal reward computation (no hierarchical structure for direct rewards)
        main_rank_print("📊 Direct architecture - using normal reward computation")
        # Call _compute_normal_reward directly - semaphore is now inside the method
        import asyncio
        reward_tensor, reward_extra_info = asyncio.run(self._compute_normal_reward(data, execution_results, execution_stats))
        
        # Create question_results for compatibility with trainer
        # Since DirectRewardManager doesn't use complex question extraction,
        # we create a simple structure that matches what the trainer expects
        question_results = {}
        for i, execution_result in enumerate(execution_results):
            # Use a simple key format for direct reward manager
            question_key = f"direct_question<<MySep>>{i}"
            question_results[question_key] = {
                'original_index': i,
                'ground_truth': execution_result.get('ground_truth', ''),
                'execution_result': execution_result
            }
        
        main_rank_print(f"Direct reward computation completed for {len(data)} responses")
        main_rank_print(f"{'='*60}\n")
        
        # Validate reward_extra_info to ensure all keys have correct length
        batch_size = len(data)
        main_rank_print(f"Validating reward_extra_info for {batch_size} responses...")
        
        # Only validate the fields that follow Level 1 structure
        required_fields = ['code_execution_success', 'combined_reward', 'final_answer_correctness', 'score']
        
        for key in required_fields:
            if key not in reward_extra_info:
                error_msg = f"ERROR: Missing required field '{key}' in reward_extra_info"
                main_rank_print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            current_length = len(reward_extra_info[key]) if reward_extra_info[key] else 0
            
            # All fields should have batch_size length
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
        if not is_validation:  # Only pad for training data
            from verl.protocol import pad_dataproto_to_divisor
            world_size = trainer_instance.actor_rollout_wg.world_size
            data_padded, pad_size = pad_dataproto_to_divisor(data, world_size)
            
            # Pad reward tensor with zeros for the padding samples
            reward_tensor_padded = torch.zeros((len(data_padded), reward_tensor.size(1)), dtype=reward_tensor.dtype)
            reward_tensor_padded[:len(data), :] = reward_tensor
            main_rank_print(f"   - Padded tensor size: {len(data_padded)} vs. {len(reward_tensor_padded)}")
            
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
                "expanded_data": data_for_training
            }
        else:
            return reward_tensor


def setup_reward_manager(tokenizer, num_examine, config):
    """Set up the Direct reward manager."""
    return DirectRewardManager(tokenizer=tokenizer, num_examine=num_examine, config=config)