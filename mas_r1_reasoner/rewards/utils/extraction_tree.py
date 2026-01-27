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
Extraction structure utilities for MAS-R1 Trainer with two-level tree architecture.
"""

from typing import Dict, Tuple, Any, List
from verl.protocol import DataProto
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code
from mas_r1_reasoner.trainer.utils.helper import get_safe_length
import torch
import numpy as np
from mas_r1_reasoner.trainer.utils.rollout import rollout_generation
from mas_r1_reasoner.rewards.utils.execution import (
    execute_codes_and_store_results,
)
from mas_r1_reasoner.rewards.utils.uid_planning import get_uid_planning_function, validate_uid_planning
from mas_r1_reasoner.rewards.utils.extraction_strategy import (extract_codes)
from mas_r1_reasoner.agents.common import get_prompt, get_known_prompt
from mas_r1_reasoner.agents.shared_vars import get_global
import copy


class SubAgentBatchManager:
    """
    Manages batch processing of sub-agent generation for tree architecture.
    Processes all sub-tasks and sub-agents synchronously in batches.
    """
    
    def __init__(self, trainer_instance, new_sub_agents_per_sub_task: int = None):
        self.trainer_instance = trainer_instance
        # Use trainer's configurable value if not explicitly provided
        if new_sub_agents_per_sub_task is None:
            new_sub_agents_per_sub_task = trainer_instance.sub_agents_per_sub_task
        self.new_sub_agents_per_sub_task = new_sub_agents_per_sub_task
    
    def generate_all_sub_agents_batch(self, sub_tasks: List[str], question_results: Dict = None, is_validation: bool = False) -> List[Tuple[DataProto, str, str]]:
        """
        Generate all sub-agents for all sub-tasks in a single batch operation.
        Each sub-task generates multiple new sub-tasks using the same prompt.
        """
        main_rank_print(f"🚀 Starting batch sub-agent generation for {len(sub_tasks)} sub-tasks...")
        
        if not sub_tasks:
            main_rank_print("📝 No sub-tasks to process")
            return []
        
        # Create all sub-agent batches for all sub-tasks
        all_sub_agent_batches = []
        sub_task_batch_coordination = []
        
        for sub_task_idx, sub_task in enumerate(sub_tasks):
            # Generate multiple new sub-tasks for each sub-task using the same prompt
            for new_sub_agent_idx in range(self.new_sub_agents_per_sub_task):
                # Create batch for this specific sub-task with the standard prompt
                sub_agent_batch = create_sub_agent_batch_from_sub_task_and_prompt(
                    self.trainer_instance, 
                    sub_task, 
                    new_sub_agent_idx,  # This will be used to generate different new sub-tasks
                    question_results,  # Pass the question_results for context
                    sub_task_idx  # Pass the sub-task index for debug printing
                )
                
                all_sub_agent_batches.append(sub_agent_batch)
                # Use sub_task_idx which is globally unique across all sub-tasks to prevent duplication
                sub_task_batch_coordination.append((sub_task, f"sub_task_{sub_task_idx}_new_sub_agent_{new_sub_agent_idx}", new_sub_agent_idx))
        
        main_rank_print(f"📝 Created {len(all_sub_agent_batches)} sub-agent batches ({len(sub_tasks)} sub-tasks × {self.new_sub_agents_per_sub_task} new sub-tasks each)")
        
        # Process all sub-agent batches in a single generation call
        if all_sub_agent_batches:
            # Combine all batches into one large batch
            combined_batch = self._combine_sub_agent_batches(all_sub_agent_batches)
            
            # DEBUG: Print the first input from the combined batch to see Level 2 input
            main_rank_print(f"\n{'='*60}")
            main_rank_print("DEBUG: LEVEL 2 INPUT - FIRST SAMPLE FROM COMBINED BATCH")
            main_rank_print(f"{'='*60}")
            if combined_batch and len(combined_batch) > 0:
                # first_sample = combined_batch[0]
                # main_rank_print(f"First sample batch keys: {list(first_sample.batch.keys())}")
                # if 'input_ids' in first_sample.batch:
                #     input_ids = first_sample.batch['input_ids']
                #     main_rank_print(f"First sample input_ids shape: {input_ids.shape}")
                #     # Decode the first input to see the actual prompt
                #     try:
                #         tokenizer = self.trainer_instance.tokenizer
                #         decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
                #         main_rank_print(f"First sample decoded input (first 500 chars):")
                #         main_rank_print(f"'{decoded_input}...'")
                #     except Exception as e:
                #         main_rank_print(f"Failed to decode first sample: {e}")
                
                # if hasattr(first_sample, 'non_tensor_batch') and first_sample.non_tensor_batch:
                #     main_rank_print(f"First sample non_tensor_batch keys: {list(first_sample.non_tensor_batch.keys())}")
                
                main_rank_print(f"Combined batch total size: {len(combined_batch)}")
                main_rank_print(f"{'='*60}\n")
            else:
                main_rank_print(f"❌ Combined batch is empty or None!")
                main_rank_print(f"{'='*60}\n")
            
            # Always pad in Level 2 since batch size is unpredictable
            # (depends on number of sub-tasks extracted from Level 1 responses)
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            combined_batch_padded, pad_size = pad_dataproto_to_divisor(combined_batch, self.trainer_instance.actor_rollout_wg.world_size)
            
            # Use validation pattern to force n=1 for Level 2 generation
            # This leverages VERL's built-in validation logic that automatically sets n=1
            # main_rank_print(f"   - Set validate=True to force VERL to use n=1 (same as validation mode)")
            # main_rank_print(f"   - This ensures Level 2 generates exactly 1 response per sub-task")
           
            combined_batch_padded.meta_info = {
                'eos_token_id': self.trainer_instance.tokenizer.eos_token_id,
                'pad_token_id': self.trainer_instance.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                "do_sample": self.trainer_instance.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            } 

            # side affect: It will use val_kwargs. like tempreture=0.5
            # Generate sequences - VERL will automatically use n=1 when validate=True
            sub_agent_output_padded = self.trainer_instance.actor_rollout_wg.generate_sequences(combined_batch_padded)            
            

            if not is_validation: # this only for training
                # Apply the same processing that the trainer does to the batch before reward_fn
                # This ensures Level 2 responses have the same tensor fields as Level 1 responses
                # Apply trainer-level processing BEFORE unpadding to ensure proper chunking
                main_rank_print(f"🔧 Applying trainer-level processing to Level 2 responses (padded)...")
                
                # Apply trainer-level processing in the exact same order as the trainer
                sub_agent_output_padded = self._apply_trainer_level_processing(sub_agent_output_padded)
                
                main_rank_print(f"✅ Successfully applied trainer-level processing to Level 2 responses")
                main_rank_print(f"   Final Level 2 tensor fields: {list(sub_agent_output_padded.batch.keys())}")
            
            # Unpad the results AFTER processing to maintain proper chunking during compute_log_prob
            sub_agent_output = unpad_dataproto(sub_agent_output_padded, pad_size=pad_size)
            
            # Split the output back into individual sub-agent results
            results = self._split_sub_agent_results(sub_agent_output, sub_task_batch_coordination)
            
            successful_count = len([r for r, _, _ in results if r is not None])
            main_rank_print(f"🎯 Batch sub-agent generation completed: {successful_count}/{len(results)} successful")
            
            return results
        else:
            main_rank_print("📝 No sub-agent batches to process")
            return []
    
    def _combine_sub_agent_batches(self, sub_agent_batches: List[DataProto]) -> DataProto:
        """
        Combine multiple sub-agent batches into one large batch for efficient processing.
        Uses DataProto.from_single_dict following the raw_data mode pattern.
        """
        if not sub_agent_batches:
            return None
        
        if len(sub_agent_batches) == 1:
            return sub_agent_batches[0]
        
        main_rank_print(f"🔗 Combining {len(sub_agent_batches)} sub-agent batches into single batch")
        
        # Get the first batch as reference for structure and meta_info
        reference_batch = sub_agent_batches[0]
        
        # Prepare batch_dict following raw_data mode pattern
        batch_dict = {}
        
        # Combine tensors for each key (following raw_data mode tensor handling)
        for key in reference_batch.batch.keys():
            try:
                # Collect all tensors for this key
                tensors_to_combine = [batch.batch[key] for batch in sub_agent_batches if key in batch.batch and batch.batch[key] is not None]
                if tensors_to_combine:
                    combined_tensor = torch.cat(tensors_to_combine, dim=0)
                    batch_dict[key] = combined_tensor
                    main_rank_print(f"   ✓ Combined tensor '{key}': {combined_tensor.shape}")
                else:
                    # This should never happen - all batches should have the same structure
                    error_msg = f"Failed to combine tensor '{key}': No valid tensors found in any batch. This indicates a fundamental problem with batch structure consistency."
                    main_rank_print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
            except Exception as e:
                # This should never happen - raise error instead of fallback
                error_msg = f"Failed to combine tensor '{key}': {e}. This indicates a fundamental problem with tensor combination that must be fixed."
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        # Combine non-tensor data (following raw_data mode non_tensor_batch handling)
        non_tensor_data = {}
        if hasattr(reference_batch, 'non_tensor_batch'):
            for key in reference_batch.non_tensor_batch.keys():
                try:
                    combined_data = []
                    for batch in sub_agent_batches:
                        if hasattr(batch, 'non_tensor_batch') and key in batch.non_tensor_batch:
                            batch_data = batch.non_tensor_batch[key]
                            if isinstance(batch_data, list):
                                combined_data.extend(batch_data)
                            else:
                                combined_data.append(batch_data)
                        else:
                            combined_data.append(None)
                    
                    # Convert to numpy array following raw_data mode pattern
                    import numpy as np
                    non_tensor_data[key] = np.array(combined_data, dtype=object)
                    main_rank_print(f"   ✓ Combined non-tensor '{key}': {len(combined_data)} items")
                except Exception as e:
                    # This should never happen - raise error instead of fallback
                    error_msg = f"Failed to combine non-tensor '{key}': {e}. This indicates a fundamental problem with non-tensor data combination that must be fixed."
                    main_rank_print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
        
        # Create DataProto using from_single_dict (same as raw_data mode)
        combined_batch = DataProto.from_single_dict(batch_dict)
        
        # Set non_tensor_batch manually since from_single_dict doesn't handle it automatically
        if non_tensor_data:
            combined_batch.non_tensor_batch = non_tensor_data
        
        # Inherit meta_info from reference batch (following raw_data mode pattern)
        if hasattr(reference_batch, 'meta_info') and reference_batch.meta_info:
            combined_batch.meta_info = reference_batch.meta_info.copy()
            main_rank_print(f"   ✓ Inherited meta_info: {list(combined_batch.meta_info.keys())}")
        
        # Calculate total batch size
        total_batch_size = len(combined_batch.batch.get('input_ids', []))
        main_rank_print(f"🎯 Successfully combined {len(sub_agent_batches)} batches into single batch with size {total_batch_size}")
        
        return combined_batch
    
    def _split_sub_agent_results(self, combined_output: DataProto, sub_task_batch_coordination: List[Tuple[str, str, int]]) -> List[Tuple[DataProto, str, str]]:
        """
        Split the combined sub-agent output back into individual results.
        Uses DataProto.from_single_dict following the raw_data mode pattern.
        """
        if not combined_output or not sub_task_batch_coordination:
            return []
        
        main_rank_print(f"✂️ Splitting combined output back into {len(sub_task_batch_coordination)} individual results")
        
        # Get total size and calculate batch size
        total_size = len(combined_output.batch.get('input_ids', []))
        batch_size = total_size // len(sub_task_batch_coordination)
        
        main_rank_print(f"   📊 Total size: {total_size}, Batch size: {batch_size}")
        
        results = []
        for i, (sub_task, new_sub_agent_id, new_sub_agent_idx) in enumerate(sub_task_batch_coordination):
            try:
                # Calculate slice indices
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < len(sub_task_batch_coordination) - 1 else total_size
                
                # Prepare batch_dict following raw_data mode pattern
                batch_dict = {}
                
                # Slice tensor data
                for key, tensor in combined_output.batch.items():
                    if tensor is not None:
                        batch_dict[key] = tensor[start_idx:end_idx]
                
                # Create DataProto using from_single_dict (same as raw_data mode)
                individual_result = DataProto.from_single_dict(batch_dict)
                
                # Inherit meta_info from combined output (following raw_data mode pattern)
                if hasattr(combined_output, 'meta_info') and combined_output.meta_info:
                    individual_result.meta_info = {}
                    for key, value in combined_output.meta_info.items():
                        if isinstance(value, list):
                            # Split global_token_num list according to the slice indices
                            individual_result.meta_info[key] = value[start_idx:end_idx]

                
                # Set non_tensor_batch manually since from_single_dict doesn't handle it automatically
                individual_result.non_tensor_batch = {}
                if hasattr(combined_output, 'non_tensor_batch'):
                    for key, data in combined_output.non_tensor_batch.items():
                        if isinstance(data, list):
                            individual_result.non_tensor_batch[key] = data[start_idx:end_idx]
                        else:
                            individual_result.non_tensor_batch[key] = data
                
                results.append((individual_result, sub_task, new_sub_agent_id))
                # main_rank_print(f"   ✓ Split result {i+1}: {sub_task} -> {new_sub_agent_id} (indices {start_idx}:{end_idx})")
                
            except Exception as e:
                # This should never happen - raise error instead of fallback
                error_msg = f"Failed to split result for {sub_task}: {e}. This indicates a fundamental problem with result splitting that must be fixed."
                main_rank_print(f"   ❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        main_rank_print(f"🎯 Successfully split combined output into {len(results)} individual results")
        return results

    def _apply_trainer_level_processing(self, sub_agent_output):
        """
        Apply the same processing that the trainer does to the batch before reward_fn.
        
        This function replicates the trainer's code exactly:
        - Same comments
        - Same variable names  
        - Same logic flow
        - Same error handling
        """

        #TODO: we do not do _balance_batch here. The combine and split heavily depend on order
        # compute global_valid tokens
        sub_agent_output.meta_info['global_token_num'] = torch.sum(sub_agent_output.batch['attention_mask'], dim=-1).tolist()

        # recompute old_log_probs
        old_log_prob = self.trainer_instance.actor_rollout_wg.compute_log_prob(sub_agent_output)
        sub_agent_output = sub_agent_output.union(old_log_prob)

        if self.trainer_instance.use_reference_policy:
            # compute reference log_prob
            if not self.trainer_instance.ref_in_actor:
                ref_log_prob = self.trainer_instance.ref_policy_wg.compute_ref_log_prob(sub_agent_output)
            else:
                ref_log_prob = self.trainer_instance.actor_rollout_wg.compute_ref_log_prob(sub_agent_output)
            sub_agent_output = sub_agent_output.union(ref_log_prob)
            main_rank_print(f"Reference policy log prob computed and added to batch")
            main_rank_print(f"Batch keys after ref_log_prob union: {list(sub_agent_output.batch.keys())}")

        # compute values
        if self.trainer_instance.use_critic:
            values = self.trainer_instance.critic_wg.compute_values(sub_agent_output)
            sub_agent_output = sub_agent_output.union(values)

        return sub_agent_output


def create_sub_agent_batch_from_sub_task_and_prompt(trainer_instance, sub_task: str, prompt_idx: int, question_results: Dict, sub_task_idx: int = None) -> DataProto:
    """
    Create a batch for sub-agent generation from a sub-task and prompt index.
    This function creates a new batch that will be used to generate new sub-tasks.
    Follows VERL's exact dataset preparation pattern and includes the specific question for context.
    """
    from verl.protocol import DataProto
    from verl.utils.torch_functional import tokenize_and_postprocess_data
    from verl.utils.model import compute_position_id_with_mask
    
    # Get the specific question for this sub-agent batch
    # We can now use the sub_tasks_mapping we stored earlier
    specific_question = None
    if question_results:
        try:
            # Find the question that contains this sub-task using the mapping
            for key, value in question_results.items():
                if isinstance(value, dict) and 'sub_tasks_mapping' in value:
                    # Check if this sub-task exists in the mapping
                    if sub_task in value['sub_tasks_mapping']:
                        specific_question = key
                        break
            
            # If we can't find a specific question, raise an error
            if specific_question is None:
                error_msg = f"Could not find a specific question for sub-task: {sub_task}"
                main_rank_print(f"❌ {error_msg}")
                raise ValueError(error_msg)
                
            # main_rank_print(f"📝 Using specific question for sub-agent batch: {specific_question[:100]}...")
        except Exception as e:
            error_msg = f"Failed to extract specific question from question_results: {e}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    else:
        error_msg = "question_results is required but was not provided"
        main_rank_print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    if not specific_question:
        error_msg = "No specific question found for sub-agent batch"
        main_rank_print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Load the actual prompt from the file and use proper placeholders
    
    try:
        # Check if known_prompt is configured
        known_prompt = get_global("global_known_prompt")
        if known_prompt is not None:
            # Use known prompt instead of dynamic prompt
            system_prompt, mas_prompt = get_known_prompt(
                question=specific_question, 
                indicator=known_prompt, 
                level=2
            )
        else:
            # Use dynamic prompt as before
            system_prompt, mas_prompt = get_prompt(
                question=specific_question, 
                sub_task=sub_task, 
                level=2
            )
        
        # Create MAS prompt using template (same pattern as helper.py)
        mas_template = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mas_prompt},
        ]
        
        # Apply chat template
        if hasattr(trainer_instance.tokenizer, 'apply_chat_template') and getattr(trainer_instance.tokenizer, 'chat_template', None):
            full_prompt = trainer_instance.tokenizer.apply_chat_template(
                mas_template, tokenize=False, add_generation_prompt=True
            )
        else:
            raise RuntimeError("Tokenizer does not support chat template")
        
        # main_rank_print(f"📝 Generated Level 2 prompt using get_prompt with sub_task: {sub_task[:50]}...")
        
        # Use VERL's exact tokenization and post-processing pattern (same as helper.py raw_data mode)
        # Get max_length from trainer config
        max_length = trainer_instance.config.data.max_prompt_length
        pad_token_id = trainer_instance.tokenizer.pad_token_id
        
        # Import the exact functions used in helper.py raw_data mode
        from verl.utils.model import compute_position_id_with_mask
        from mas_r1_reasoner.trainer.utils.helper import tokenize_and_left_pad
        
        # Tokenize and left-pad using the same function as helper.py
        input_ids, attention_mask = tokenize_and_left_pad(
            full_prompt, 
            max_length, 
            trainer_instance.tokenizer, 
            pad_token_id
        )
        
        # Decode and print first input_text for debugging (only for first sub-task)
        if sub_task_idx == 0 and prompt_idx == 0:  # Only print for the first sub-task
            first_input_text = trainer_instance.tokenizer.decode(input_ids, skip_special_tokens=False)
            main_rank_print(f"🔍 First input text in the second calls: {first_input_text} [CUT OFF]")
        
        # Compute position IDs using VERL's function (same as in helper.py)
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
        
        # Add batch dimension since we're in raw data mode (same as helper.py)
        input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        attention_mask = attention_mask.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        position_ids = position_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        
        # Prepare batch_dict following raw_data mode pattern
        batch_dict = {
            'input_ids': input_ids,  # Keep batch dimension
            'attention_mask': attention_mask,  # Keep batch dimension
            'position_ids': position_ids,  # Keep batch dimension
        }
        
        # Create DataProto using from_single_dict (same as raw_data mode)
        sub_agent_batch = DataProto.from_single_dict(batch_dict)
        
        # Set non_tensor_batch manually since from_single_dict doesn't handle it automatically
        sub_agent_batch.non_tensor_batch = {
            'full_prompt': full_prompt,
            'system_prompt': system_prompt,
            'mas_prompt': mas_prompt,
            'sub_task': sub_task,
            'prompt_idx': prompt_idx,
            'specific_question': specific_question,
        }
        
        sub_agent_batch.meta_info = {
            'eos_token_id': trainer_instance.tokenizer.eos_token_id,
            'pad_token_id': trainer_instance.tokenizer.pad_token_id,
        }
        # main_rank_print(f"   ✓ Created meta_info: {list(sub_agent_batch.meta_info.keys())}")
    
        # main_rank_print(f"✅ Loaded and processed prompt using VERL pattern")
        # main_rank_print(f"   - Sub-task: {sub_task}...")
        # main_rank_print(f"   - Specific question: {specific_question}...")
        # main_rank_print(f"   - Tokenized length: {input_ids.shape[-1]} tokens")
        # main_rank_print(f"   - Max length: {max_length}")
        
    except Exception as e:
        error_msg = f"Failed to load/process prompt from: {e}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    return sub_agent_batch



def generate_and_extract_codes_tree(trainer_instance, reward_manager, question_results: Dict, mas_code_generation_output: DataProto) -> Tuple[DataProto, Dict]:
    """
    Generate codes and extract them into question_results dictionary with two-level tree architecture.
    
    This function implements a hierarchical generation process:
    - Level 0: Original questions
    - Level 1: First-level responses
    - Level 2: Second-level new sub-tasks
    
    Total expected rollouts: Level 1 + Level 2 responses
    
    The function:
    1. Generates first-level responses using the original batch
    2. Extracts sub-tasks from first-level responses
    3. Generates second-level new sub-tasks using batch processing
    4. Extracts codes from both levels and stores them with clear tree structure
    5. Maintains hierarchical relationships through level metadata
    6. Stores level_1_keys and level_2_keys in question_results for downstream functions
    
    Args:
        trainer_instance: The trainer instance for model generation
        reward_manager: The reward manager instance for accessing UID planning strategy
        question_results: Dictionary to store results with tree structure
        is_validation: Whether this is validation mode
    
    Returns:
        Tuple of (mas_code_generation_output, question_results) where question_results
        contains the complete tree structure with all levels and the following additional keys:
        - _level_1_keys: List of Level 1 response keys
        - _level_2_keys: List of Level 2 response keys  
        - _total_response_keys: Combined list of all response keys
    """
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("GENERATING AND EXTRACTING CODES WITH TWO-LEVEL TREE ARCHITECTURE")
    main_rank_print(f"{'='*60}")

    
    # Generate sequences for first level
    main_rank_print("🚀 FIRST LEVEL: Generating initial responses...")
    
    # Extract sub-tasks and sub-agents from first-level generation
    main_rank_print("🔍 FIRST LEVEL: Extracting sub-tasks and sub-agents...")
    # sub_tasks, sub_agents = extract_sub_tasks_and_sub_agents(trainer_instance, mas_code_generation_output, question_results)
    # main_rank_print(f"📊 FIRST LEVEL COMPLETE: {len(sub_tasks)} sub-tasks extracted from {len(sub_agents)} sub-agents")

    original_questions_list = list(question_results.keys()) # before we do anything to question_results


    sub_tasks = extract_codes(trainer_instance, mas_code_generation_output, question_results)
    main_rank_print(f"📊 FIRST LEVEL COMPLETE: {len(sub_tasks)} sub-tasks extracted")

    # SECOND LEVEL: Generate new sub-tasks using batch processing
    sub_agent_results = []
    if sub_tasks:
        main_rank_print(f"\n{'='*60}")
        main_rank_print("SECOND LEVEL: BATCH GENERATION OF NEW SUB-TASKS")
        main_rank_print(f"{'='*60}")
        
        try:
            # Initialize batch manager for second-level generation
            manager = SubAgentBatchManager(trainer_instance)
            main_rank_print(f"🔄 New sub-agents per sub-task: {manager.new_sub_agents_per_sub_task}")
            
            # Generate all sub-agents in a single batch operation
            sub_agent_results = manager.generate_all_sub_agents_batch(sub_tasks, question_results)
            
            successful_count = len([r for r, _, _ in sub_agent_results if r is not None])
            total_attempts = len(sub_tasks) * manager.new_sub_agents_per_sub_task
            
            main_rank_print(f"🚀 SECOND LEVEL COMPLETE: {successful_count}/{total_attempts} sub-task rollouts successful")
            
            # Log performance metrics
            if successful_count > 0:
                success_rate = (successful_count / total_attempts) * 100
                main_rank_print(f"📈 Tree architecture efficiency: {success_rate:.1f}% ({successful_count}/{total_attempts})")
                
                # Show breakdown by sub-task
                # for sub_task in sub_tasks:
                #     sub_task_results = [r for r, st, _ in sub_agent_results if st == sub_task and r is not None]
                #     main_rank_print(f"   - {sub_task}: {len(sub_task_results)}/{manager.new_sub_agents_per_sub_task} new sub-tasks generated")
            
        except Exception as e:
            error_msg = f"Batch sub-agent generation failed: {e}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    # Extract codes and store in dictionary with clear tree structure
    main_rank_print(f"\n{'='*60}")
    main_rank_print("EXTRACTING CODES AND STORING IN TREE STRUCTURE")
    main_rank_print(f"{'='*60}")
    
    mas_code_generation_length = get_safe_length(mas_code_generation_output, "mas_code_generation_output")
    
    # Get only original questions (keys without '_') from question_results
    
    # Determine expected generation length based on mode
    # Training: rollout.n responses per question
    rollout_n = trainer_instance.config.actor_rollout_ref.rollout.n
    expected_length = len(original_questions_list) * rollout_n
    main_rank_print(f"TRAINING MODE: Expecting {rollout_n} responses per question")

    main_rank_print(f"Questions: {len(original_questions_list)}")
    main_rank_print(f"Expected generation length: {expected_length}")
    main_rank_print(f"Actual generation length: {mas_code_generation_length}")
    
    if mas_code_generation_length != expected_length:
        error_msg = f"Generation length mismatch! Expected {expected_length}, got {mas_code_generation_length}"
        raise RuntimeError(error_msg)
    
    # Process each question and its responses (First Level Results - Level 1)
    # Training: process rollout.n responses per question (First Level Results - Level 1)    
    # Track response index for each question
    question_response_count = {}
    
    for generation_idx in range(mas_code_generation_length):
        response_text = trainer_instance.tokenizer.decode(mas_code_generation_output.batch['responses'][generation_idx], skip_special_tokens=True)
        
        # Extract question directly from mas_code_generation_output
        question = trainer_instance.processor.extract_math_question(mas_code_generation_output, generation_idx)
        
        # Increment response index for this question
        if question not in question_response_count:
            question_response_count[question] = 0
        response_idx = question_response_count[question]
        question_response_count[question] += 1
        
        # Try to extract code
        try:
            code, name, thought = extract_code_from_response(
                response_text,
                validate_python_code,
                trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
            )
            
            # Store with correct keys for {question}<<MySep>>{response} structure
            extracted_code_data = {
                'mas_code_generation_response': response_text,
                'extracted_code': code,
                'extracted_name': name,
                'extracted_thought': thought,
                'code_extraction_success': code is not None,
            }
            
            # Store directly in {question}<<MySep>>{response} key
            question_response_key = f"{question}<<MySep>>{response_idx}"
            question_results[question_response_key] = {
                'response_text': response_text,
                'extracted_code_data': extracted_code_data,
                'question': question,
                'response_idx': response_idx,
                'ground_truth': question_results[question].get('ground_truth') if question in question_results else None,
                'original_index': question_results[question].get('original_index', generation_idx) if question in question_results else generation_idx,
                'level': 1,  # Level 1: First-level responses
            }
            
            if code is not None:
                main_rank_print(f"✓ Response {generation_idx+1}: Valid code extracted for question: {question[:100]}{'...' if len(question) > 100 else ''}")
            else:
                main_rank_print(f"✗ Response {generation_idx+1}: No valid code extracted for question: {question[:100]}{'...' if len(question) > 100 else ''}")
        except Exception as e:
            main_rank_print(f"✗ Response {generation_idx+1}: Code extraction failed: {e}")
            
            # Store failed extraction
            failed_extracted_code_data = {
                'mas_code_generation_response': response_text,
                'extracted_code': "",
                'extracted_name': "Unknown",
                'extracted_thought': "",
                'code_extraction_success': False,
                'extraction_error': str(e)
            }
            
            # Store directly in {question}<<MySep>>{response} key
            question_response_key = f"{question}<<MySep>>{response_idx}"
            question_results[question_response_key] = {
                'response_text': response_text,
                'extracted_code_data': failed_extracted_code_data,
                'question': question,
                'response_idx': response_idx,
                'ground_truth': question_results[question].get('ground_truth') if question in question_results else None,
                'original_index': question_results[question].get('original_index', generation_idx) if question in question_results else generation_idx,
                'level': 1,  # Level 1: First-level responses
            }
    # Process Second Level Results (Sub-agent rollouts - Level 2)
    if sub_agent_results:
        
        main_rank_print(f"\n{'='*60}")
        main_rank_print("PROCESSING SECOND LEVEL RESULTS (NEW SUB-TASK GENERATION - LEVEL 2)")
        main_rank_print(f"{'='*60}")
        
        # Process each sub-agent result from second-level rollouts
        for sub_agent_output, sub_task, new_sub_agent_id in sub_agent_results:
            if sub_agent_output is not None:
                # main_rank_print(f"🔄 Processing second-level rollout for '{sub_task}' with '{new_sub_agent_id}'...")
                
                # Extract codes from sub-agent output (similar to generate_and_extract_codes)
                sub_agent_length = get_safe_length(sub_agent_output, "sub_agent_output")
                main_rank_print(f"📊 Second-level rollout '{new_sub_agent_id}' output shape: {sub_agent_output.batch['responses'].shape}; sub_agent_length: {sub_agent_length}")
                
                # Process each response in the sub-agent output
                for response_idx in range(sub_agent_length):
                    try:
                        # Decode the response text
                        response_text = trainer_instance.tokenizer.decode(
                            sub_agent_output.batch['responses'][response_idx], 
                            skip_special_tokens=True
                        )
                        
                        # Extract code from response (same logic as generate_and_extract_codes)
                        code, name, thought = extract_code_from_response(
                            response_text,
                            validate_python_code,
                            trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
                        )
                        
                        # Create extracted code data
                        extracted_code_data = {
                            'mas_code_generation_response': response_text,
                            'extracted_code': code,
                            'extracted_name': name,
                            'extracted_thought': thought,
                            'code_extraction_success': code is not None,
                        }
                        
                        # Find the original question and response that generated this sub-task
                        # We can now use the sub_tasks_mapping we stored earlier
                        original_question = None

                        # Search through question_results to find the original question that contains this sub_task
                        for key, value in question_results.items():
                            if isinstance(value, dict) and 'sub_tasks_mapping' in value:
                                # Check if this sub-task exists in the mapping
                                if sub_task in value['sub_tasks_mapping']:
                                    # Get the first occurrence (we can enhance this later if needed)
                                    #TODO: current way is similar to always take the first occurrence
                                    mapping_data = value['sub_tasks_mapping'][sub_task][0]
                                    original_question = key
                                    original_question_idx = value.get('original_index', 0)
                                    original_response_idx = mapping_data['response_idx']
                                    break
                        
                        # If we can't find the original, raise error instead of fallback
                        if original_question is None:
                            error_msg = f"Could not find original question for sub-task '{sub_task}'. This indicates a fundamental problem with the tree structure mapping."
                            main_rank_print(f"❌ {error_msg}")
                            # import time
                            # time.sleep(100000)
                            raise RuntimeError(error_msg)
                        
                        # Create unique key for this second-level result
                        sub_agent_key = f"second_level<<MySep>>{new_sub_agent_id}<<MySep>>{response_idx}"
                        
                        # Store sub-agent results in question_results with proper tree structure
                        question_results[sub_agent_key] = {
                            'response_text': response_text,
                            'extracted_code_data': extracted_code_data,
                            'question': original_question,
                            'response_idx': response_idx,
                            'ground_truth': question_results.get(original_question, {}).get('ground_truth'),
                            'original_index': original_question_idx,
                            'is_validation': False,
                            'level': 2,  # Level 2: Second-level new sub-task responses
                            'parent_sub_task': sub_task,
                            'parent_response_idx': original_response_idx,
                            'sub_agent_output': sub_agent_output,  # Keep reference to original output
                            'new_sub_agent_id': new_sub_agent_id
                        }
                        
                        if code is not None:
                            main_rank_print(f"✓ Second-level '{new_sub_agent_id[:50]}' Response {response_idx+1}: Valid code extracted")
                        else:
                            main_rank_print(f"✗ Second-level '{new_sub_agent_id[:50]}' Response {response_idx+1}: No valid code extracted")
                            
                    except Exception as e:
                        main_rank_print(f"✗ Second-level '{new_sub_agent_id}' Response {response_idx+1}: Code extraction failed: {e}")
                        
                        # Find the original question for this sub-task (same logic as above)
                        original_question = None
                        
                        # Search through question_results to find the original question that contains this sub_task
                        for key, value in question_results.items():
                            if isinstance(value, dict) and 'sub_tasks_mapping' in value:
                                # Check if this sub-task exists in the mapping
                                if sub_task in value['sub_tasks_mapping']:
                                    # Get the first occurrence (we can enhance this later if needed)
                                    mapping_data = value['sub_tasks_mapping'][sub_task][0]
                                    original_question = key
                                    original_question_idx = value.get('original_index', 0)
                                    original_response_idx = mapping_data['response_idx']
                                    break
                        
                        # If we can't find the original, raise error instead of fallback
                        if original_question is None:
                            error_msg = f"Could not find original question for sub-task '{sub_task}' in exception handler. This indicates a fundamental problem with the tree structure mapping."
                            main_rank_print(f"❌ {error_msg}")
                            raise RuntimeError(error_msg)
                        
                        # Store failed extraction
                        failed_extracted_code_data = {
                            'mas_code_generation_response': response_text if 'response_text' in locals() else "Unknown",
                            'extracted_code': "",
                            'extracted_name': "Unknown",
                            'extracted_thought': "",
                            'code_extraction_success': False,
                            'extraction_error': str(e)
                        }
                        
                        # Create unique key for this failed second-level result
                        sub_agent_key = f"second_level<<MySep>>{new_sub_agent_id}<<MySep>>{response_idx}"
                        
                        # Store failed sub-agent result
                        question_results[sub_agent_key] = {
                            'response_text': "Unknown",
                            'extracted_code_data': failed_extracted_code_data,
                            'question': original_question,  # Use the found original question
                            'response_idx': response_idx,
                            'ground_truth': question_results[original_question].get('ground_truth'),
                            'original_index': original_question_idx,
                            'is_validation': False,
                            'level': 2,
                            'parent_sub_task': sub_task,
                            'parent_response_idx': original_response_idx,
                            'sub_agent_output': sub_agent_output,
                            'new_sub_agent_id': new_sub_agent_id
                        }
            else:
                main_rank_print(f"❌ Skipping failed second-level rollout for '{new_sub_agent_id}'")
    
    # Summary of two-level tree architecture
    if sub_agent_results:
        main_rank_print(f"\n{'='*60}")
        main_rank_print("TWO-LEVEL TREE ARCHITECTURE SUMMARY")
        main_rank_print(f"{'='*60}")
        
        # Count actual second-level results that were processed
        second_level_count = 0
        second_level_success_count = 0
        
        for key, value in question_results.items():
            if isinstance(value, dict) and value.get('level') == 2:
                second_level_count += 1
                if value.get('extracted_code_data', {}).get('code_extraction_success', False):
                    second_level_success_count += 1
        
        main_rank_print(f"✅ Level 0 (Original Questions): {len(original_questions_list)} questions")
        main_rank_print(f"✅ Level 1 (First-Level Responses): {mas_code_generation_length} responses")
        main_rank_print(f"✅ Level 2 (Second-Level New Sub-Tasks): {second_level_count} responses processed ({second_level_success_count} successful code extractions)")
        
        # Calculate total tree output
        total_level_1 = mas_code_generation_length
        total_level_2 = second_level_count
        total_tree_output = total_level_1 + total_level_2
        
        main_rank_print(f"🌳 Total Tree Output: {total_tree_output} responses")
        main_rank_print(f"   - Level 1: {total_level_1} responses")
        main_rank_print(f"   - Level 2: {total_level_2} new sub-task responses")
        
    # Collect level_1_keys and level_2_keys for downstream functions
    level_1_keys = []
    level_2_keys = []
    
    for key, value in question_results.items():
        if isinstance(value, dict):
            if value.get('level') == 1:
                level_1_keys.append(key)
            elif value.get('level') == 2:
                level_2_keys.append(key)
    
    # Store the keys in question_results for easy access
    question_results['_level_1_keys'] = level_1_keys
    question_results['_level_2_keys'] = level_2_keys
    question_results['_total_response_keys'] = level_1_keys + level_2_keys
    
    main_rank_print(f"🔑 Collected keys for downstream functions:")
    main_rank_print(f"   Level 1 keys: {len(level_1_keys)}")
    main_rank_print(f"   Level 2 keys: {len(level_2_keys)}")
    main_rank_print(f"   Total response keys: {len(level_1_keys) + len(level_2_keys)}")
    
    # Create expanded DataProto with dynamic total responses (Level 1 + Level 2)
    if level_2_keys:
        # Calculate expected total dynamically instead of hardcoded 32
        rollout_n = trainer_instance.config.actor_rollout_ref.rollout.n
        new_sub_agents_per_sub_task = trainer_instance.sub_agents_per_sub_task  # This should match the value in SubAgentBatchManager
        expected_level_1 = len(original_questions_list) * rollout_n
        expected_level_2 = expected_level_1 * new_sub_agents_per_sub_task
        expected_total = expected_level_1 + expected_level_2
        
        main_rank_print(f"🌳 Dynamic calculation for tree architecture:")
        main_rank_print(f"   - Questions: {len(original_questions_list)}")
        main_rank_print(f"   - Rollout.n: {rollout_n}")
        main_rank_print(f"   - Sub-agents per sub-task: {new_sub_agents_per_sub_task}")
        main_rank_print(f"   - Expected Level 1: {expected_level_1} responses")
        main_rank_print(f"   - Expected Level 2: {expected_level_2} responses")
        main_rank_print(f"   - Expected Total: {expected_total} responses")
        
        expanded_mas_code_generation_output = _create_expanded_dataproto(
            trainer_instance, reward_manager, mas_code_generation_output, sub_agent_results, level_1_keys, level_2_keys
        )
    else:
        # This should never happen in tree architecture - raise error instead of fallback
        error_msg = f"Tree architecture enabled but no Level 2 responses found. Expected {expected_level_2} Level 2 responses but got 0. This indicates a fundamental failure in the tree architecture generation process."
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    return expanded_mas_code_generation_output, question_results


def _create_expanded_dataproto(trainer_instance, reward_manager, original_mas_code_generation_output: DataProto, sub_agent_results: List[Tuple[DataProto, str, str]], 
                              level_1_keys: List[str], level_2_keys: List[str]) -> DataProto:
    """
    Create an expanded DataProto by combining original Level 1 responses with Level 2 responses.
    
    Args:
        trainer_instance: The trainer instance for accessing configuration
        reward_manager: The reward manager instance for accessing UID planning strategy
        original_mas_code_generation_output: Original DataProto with Level 1 responses
        sub_agent_results: List of (DataProto, sub_task, new_sub_agent_idx) tuples for Level 2
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        
    Returns:
        Expanded DataProto with total responses (Level 1 + Level 2)
    """
    main_rank_print(f"\n{'='*60}")
    main_rank_print(f"CREATING EXPANDED DATAPROTO WITH {len(level_1_keys) + len(level_2_keys)} RESPONSES")
    main_rank_print(f"{'='*60}")
    
    # Start with original Level 1 responses
    expanded_batch = {}
    
    # Copy all tensor fields from original mas_code_generation_output (Level 1)
    for key, tensor in original_mas_code_generation_output.batch.items():
        if tensor is not None:
            expanded_batch[key] = tensor
            main_rank_print(f"✓ Copied Level 1 tensor '{key}': {tensor.shape}")
    
    # Extract actual Level 2 tensors from sub_agent_results
    level_2_tensors = {}
    
    if not sub_agent_results:
        error_msg = "No sub_agent_results provided for Level 2 tensor extraction. This indicates a fundamental failure in Level 2 generation."
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Get the first sub_agent_result to understand the tensor structure
    first_sub_agent_result = sub_agent_results[0][0]  # (DataProto, sub_task, new_sub_agent_idx)
    
    if first_sub_agent_result is None:
        error_msg = "First sub_agent_result is None. This indicates a fundamental failure in Level 2 generation."
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Extract tensor fields from Level 2 responses
    for key in first_sub_agent_result.batch.keys():
        if key not in expanded_batch:
            continue  # Skip fields that don't exist in Level 1
            
        # Collect all Level 2 tensors for this field
        level_2_tensor_list = []
        
        for sub_agent_output, sub_task, new_sub_agent_idx in sub_agent_results:
            if sub_agent_output is None:
                error_msg = f"Sub_agent_output is None for sub_task '{sub_task}'. This indicates a fundamental failure in Level 2 generation."
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            if key in sub_agent_output.batch and sub_agent_output.batch[key] is not None:
                level_2_tensor_list.append(sub_agent_output.batch[key])
            else:
                error_msg = f"Missing tensor field '{key}' in sub_agent_output for sub_task '{sub_task}'. This indicates a fundamental failure in Level 2 tensor structure."
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        # Concatenate all Level 2 tensors for this field
        if level_2_tensor_list:
            combined_level_2_tensor = torch.cat(level_2_tensor_list, dim=0)
            level_2_tensors[key] = combined_level_2_tensor
            main_rank_print(f"✓ Extracted actual Level 2 tensor '{key}': {combined_level_2_tensor.shape}")
        else:
            error_msg = f"No valid Level 2 tensors found for field '{key}'. This indicates a fundamental failure in Level 2 tensor extraction."
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    # Combine Level 1 and Level 2 tensors
    for key in expanded_batch.keys():
        if key in level_2_tensors:
            # Concatenate Level 1 + Level 2 = total responses
            combined_tensor = torch.cat([expanded_batch[key], level_2_tensors[key]], dim=0)
            expanded_batch[key] = combined_tensor
            main_rank_print(f"✓ Combined tensor '{key}': {combined_tensor.shape} ({len(level_1_keys)} + {len(level_2_keys)} = {len(level_1_keys) + len(level_2_keys)})")
        else:
            error_msg = f"Missing Level 2 tensor for field '{key}'. Cannot create expanded DataProto without all tensor fields."
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    # Create the expanded DataProto
    expanded_dataproto = DataProto.from_single_dict(expanded_batch)
    
    # Copy non-tensor data and meta_info
    if hasattr(original_mas_code_generation_output, 'non_tensor_batch'):
        expanded_dataproto.non_tensor_batch = original_mas_code_generation_output.non_tensor_batch.copy()
    
    # CRITICAL: Expand existing fields to match the total responses (Level 1 + Level 2)
    # We don't create new fields - we expand the existing ones that the trainer expects
    
    total_responses = len(level_1_keys) + len(level_2_keys)
    main_rank_print(f"📊 Expanding fields to match {total_responses} total responses:")
    main_rank_print(f"   - Level 1: {len(level_1_keys)} responses")
    main_rank_print(f"   - Level 2: {len(level_2_keys)} responses")
    
    # 1. Expand 'uid' field if it exists, otherwise create it
    if 'uid' in expanded_dataproto.non_tensor_batch:
        # Expand existing uid field
        original_uids = expanded_dataproto.non_tensor_batch['uid']
        if len(original_uids) < total_responses:
            # Determine UID planning strategy based on trainer configuration
            try:
                # Get the UID planning strategy from trainer config
                if reward_manager.unified_group:
                    strategy = 'unified_group'
                elif reward_manager.expansive_group:
                    strategy = 'expansive_group'
                elif reward_manager.diff_based_reward:
                    strategy = 'diff_based_reward'
                else:
                    raise ValueError(f"Invalid UID planning strategy: {reward_manager.unified_group} or {reward_manager.expansive_group} or {reward_manager.diff_based_reward}")
                
                main_rank_print(f"🎯 Using UID planning strategy: {strategy}")
                
                # Get the appropriate UID planning function
                uid_planning_func = get_uid_planning_function(strategy)
                
                # Plan UIDs using the selected strategy
                expanded_uids, uid_mapping_info = uid_planning_func(
                    original_uids, level_1_keys, level_2_keys, trainer_instance
                )
                
                # Update the DataProto with expanded UIDs
                expanded_dataproto.non_tensor_batch['uid'] = np.array(expanded_uids, dtype=object)
                
                # Validate the UID planning results
                validate_uid_planning(expanded_uids, level_1_keys, level_2_keys, uid_mapping_info, trainer_instance)
                
                main_rank_print(f"✅ UID expansion completed using {strategy} strategy")
                main_rank_print(f"   - Total UIDs: {len(expanded_uids)} ({len(original_uids)} original + {len(expanded_uids) - len(original_uids)} new/inherited)")
                
            except Exception as e:
                error_msg = f"Failed to plan UIDs using {strategy} strategy: {e}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
    else:
        raise RuntimeError("Missing 'uid' field in expanded DataProto")


    # Handle meta_info carefully - some fields need to be combined, others just copied
    if hasattr(original_mas_code_generation_output, 'meta_info'):
        expanded_dataproto.meta_info = original_mas_code_generation_output.meta_info.copy()
        
        # Special handling for global_token_num - combine from Level 1 and Level 2
        if 'global_token_num' in expanded_dataproto.meta_info:
            level_1_global_token_num = expanded_dataproto.meta_info['global_token_num']
            
            # Collect Level 2 global_token_num from all sub_agent_outputs
            # Now that _split_sub_agent_results properly splits meta_info, each has the correct values
            level_2_global_token_num = []
            for sub_agent_output, _, _ in sub_agent_results:
                if sub_agent_output is not None and hasattr(sub_agent_output, 'meta_info') and 'global_token_num' in sub_agent_output.meta_info:
                    # Each sub_agent_output now has the correct split global_token_num
                    level_2_global_token_num.extend(sub_agent_output.meta_info['global_token_num'])
                else:
                    raise RuntimeError(f"No global_token_num found in Level 2 meta_info for sub_agent_output: {sub_agent_output}")
            
            # Combine Level 1 + Level 2 global_token_num
            combined_global_token_num = level_1_global_token_num + level_2_global_token_num
            expanded_dataproto.meta_info['global_token_num'] = combined_global_token_num
            
            main_rank_print(f"✓ Combined global_token_num: Level 1 ({len(level_1_global_token_num)}) + Level 2 ({len(level_2_global_token_num)}) = {len(combined_global_token_num)}")
            
            # Validate that the combined length matches our expected total
            expected_total = len(level_1_keys) + len(level_2_keys)
            if len(combined_global_token_num) != expected_total:
                error_msg = f"global_token_num length mismatch: expected {expected_total}, got {len(combined_global_token_num)}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        else:
            main_rank_print(f"⚠️  No global_token_num found in Level 1 meta_info")
    
    # Validate the expanded structure
    total_responses = len(expanded_dataproto)
    expected_responses = len(level_1_keys) + len(level_2_keys)
    
    main_rank_print(f"🎯 Expanded DataProto created successfully:")
    main_rank_print(f"   Expected responses: {expected_responses} ({len(level_1_keys)} Level 1 + {len(level_2_keys)} Level 2)")
    main_rank_print(f"   Actual responses: {total_responses}")
    main_rank_print(f"   Tensor fields: {list(expanded_batch.keys())}")
    
    if total_responses != expected_responses:
        error_msg = f"Expanded DataProto size mismatch: expected {expected_responses}, got {total_responses}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Final validation: ensure all required fields are present and have correct sizes
    main_rank_print(f"\n🔍 Final validation of expanded DataProto...")
    
    # Required tensor fields (these are the core fields the trainer expects)
    required_tensor_fields = ['input_ids', 'attention_mask', 'responses']
    for field in required_tensor_fields:
        if field not in expanded_dataproto.batch:
            error_msg = f"Missing required tensor field '{field}' in expanded DataProto"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        field_size = expanded_dataproto.batch[field].size(0)
        if field_size != total_responses:
            error_msg = f"Tensor field '{field}' size mismatch: expected {total_responses}, got {field_size}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✓ Tensor field '{field}': {expanded_dataproto.batch[field].shape}")

    # Required non-tensor fields (these are expected by the trainer and reward manager)
    required_non_tensor_fields = ['uid']
    for field in required_non_tensor_fields:
        if field not in expanded_dataproto.non_tensor_batch:
            error_msg = f"Missing required non-tensor field '{field}' in expanded DataProto"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        field_size = len(expanded_dataproto.non_tensor_batch[field])
        if field_size != total_responses:
            error_msg = f"Non-tensor field '{field}' size mismatch: expected {total_responses}, got {field_size}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✓ Required non-tensor field '{field}': {field_size} entries")

    main_rank_print(f"✅ Successfully created expanded DataProto with {total_responses} responses")
    main_rank_print(f"{'='*60}")
    
    return expanded_dataproto
