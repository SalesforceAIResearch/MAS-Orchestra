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
Harmony Reward Manager that inherits from NaiveRewardManager.
This provides a simplified interface for harmony reward computation where input_data
already contains the final answer, eliminating the need for extraction/execution steps.
"""
import torch
from typing import Dict, Tuple
from verl import DataProto
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.rewards.utils.execution import execute_codes_and_store_results
from mas_r1_reasoner.rewards.utils.extraction import (
    extract_questions_and_ground_truth,
    generate_and_extract_codes,
)
from mas_r1_reasoner.rewards.utils.reformat import from_question_results_to_execution_results
from mas_r1_reasoner.rewards.mas_r1_reward_manager import MASR1RewardManager


def handle_direct_answers_after_execution(question_results: Dict, is_validation: bool) -> Dict:
    """
    Handle direct answers by replacing entries in execution_results and execution_stats after normal execution flow.
    
    Args:
        question_results: Dictionary containing question results with extracted_code_data
        is_validation: Whether this is validation mode
        
    Returns:
        Updated question_results with direct answers replaced
    """
    main_rank_print("Checking for direct answers and replacing execution results")
    
    # Initialize counters for logging
    direct_answer_count = 0
    code_execution_count = 0
    total_questions = 0
    
    for key, value in question_results.items():
        if isinstance(value, dict) and 'extracted_code_data' in value:
            total_questions += 1
            
            # Get fields from extracted_code_data
            extracted_code_data = value['extracted_code_data']
            answer = extracted_code_data.get('extracted_thought')
            name = extracted_code_data.get('extracted_name')
            extracted_code = extracted_code_data.get('extracted_code')
            
            # If code is "direct_answer", this is a direct answer - replace execution results
            if name == "direct_answer":
                direct_answer_count += 1
                main_rank_print(f"Direct answer found for {key}, replacing execution results")
                
                # Use empty string as fallback if no answer found
                if not answer:
                    answer = ''
                
                # Calculate success once
                success = bool(answer)  # True if answer is not empty, False if empty
                
                # Replace execution_result with direct answer data
                value['execution_result'] = {
                    'code': '',  # No code for direct answers
                    'result': answer,
                    'success': success,
                    'error': answer,  # Use answer as error for compatibility
                    'question': value.get('question', ''),
                    'response_idx': value.get('response_idx', 0),
                    'ground_truth': value.get('ground_truth', ''),
                    'original_index': value.get('original_index', 0),
                    'is_validation': is_validation,
                    'level': 1,
                    'parent_response_idx': value.get('parent_response_idx', None),
                    'parent_sub_task': value.get('parent_sub_task', None),
                    'response_text': value.get('response_text', '')
                }
                
                # Replace execution_stats with direct answer data
                value['execution_stats'] = {
                    'total_executions': 1,  # One "execution" (direct answer)
                    'successful_executions': 1 if success else 0,
                    'failed_executions': 0 if success else 1,
                    'execution_results': [(answer, success, answer)],  # Put answer as execution result
                    'success_rate': 1.0 if success else 0.0,  # Success based on whether answer exists
                    'question': value.get('question', ''),
                    'response_idx': value.get('response_idx', 0),
                    'ground_truth': value.get('ground_truth', ''),
                    'code': '',  # No code for direct answers
                    'is_validation': is_validation
                }
                
                if answer:
                    main_rank_print(f"  Replaced with direct answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                else:
                    main_rank_print(f"  No answer found, using empty string as fallback")
            else:
                code_execution_count += 1
                main_rank_print(f"Code execution result for {key}: {extracted_code[:100]}{'...' if len(extracted_code) > 100 else ''}")
    
    # Log the statistics
    main_rank_print(f"\n{'='*60}")
    main_rank_print(f"HARMONY DIRECT ANSWER PROCESSING STATISTICS")
    main_rank_print(f"{'='*60}")
    main_rank_print(f"Total questions processed: {total_questions}")
    main_rank_print(f"Direct answers (replaced execution results): {direct_answer_count}")
    main_rank_print(f"Code execution results (kept as-is): {code_execution_count}")
    main_rank_print(f"Direct answer percentage: {(direct_answer_count/total_questions*100):.1f}%" if total_questions > 0 else "N/A")
    main_rank_print(f"Code execution percentage: {(code_execution_count/total_questions*100):.1f}%" if total_questions > 0 else "N/A")
    main_rank_print(f"{'='*60}\n")
    
    return question_results



class HarmonyRewardManager(MASR1RewardManager):
    """Harmony Reward Manager that inherits from NaiveRewardManager.
    
    This manager works with input_data that already contains the final answer,
    eliminating the need for extraction and execution steps.
    """
    
    def __init__(self, tokenizer, num_examine=5, config=None):
        # Initialize parent NaiveRewardManager with our compute_score function
        super().__init__(tokenizer=tokenizer, num_examine=num_examine, config=config)
              
        main_rank_print(f"Harmony Reward Manager initialized with MAS-R1 compute_score)")
    
    def __call__(self, trainer_instance, sample_outputs, is_validation, input_data: DataProto, return_dict: bool = False):
        """Override to handle harmony reward computation with code generation and execution."""

        #--------------------------------

        # Step 1: Extract questions and ground truth using shared function
        question_results = extract_questions_and_ground_truth(trainer_instance, input_data)

        # Step 2: Generate and extract codes using shared function
        # Choose generation function based on tree architecture configuration
        
        # validation do not need to do structure
        main_rank_print("Generating and extracting codes")
        mas_code_generation_output, question_results = generate_and_extract_codes(
            trainer_instance, question_results, input_data, is_validation=is_validation
        )

        # Step 4: Execute codes and store results using shared function
        main_rank_print("execute_codes_and_store_results")
        question_results = execute_codes_and_store_results(trainer_instance, question_results, is_validation=is_validation)

        # Step 5: Handle direct answers after executio
        main_rank_print("handle_direct_answers_after_execution")
        question_results = handle_direct_answers_after_execution(question_results, is_validation)


        # Step 5: Convert dictionary back to DataProto format for VERL
        # Create final output with execution results
        main_rank_print("from_question_results_to_execution_results")
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

def setup_reward_manager(tokenizer, num_examine, config):
    """Set up the Harmony reward manager."""
    return HarmonyRewardManager(tokenizer=tokenizer, num_examine=num_examine, config=config)