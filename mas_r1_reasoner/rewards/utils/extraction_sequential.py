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

import torch
import numpy as np
from mas_r1_reasoner.rewards.utils.extraction_tree import (
    SubAgentBatchManager,
)
from mas_r1_reasoner.rewards.utils.extraction_strategy import (extract_codes)

def generate_and_extract_codes_sequential(trainer_instance, reward_manager, question_results: Dict, mas_code_generation_output: DataProto) -> Tuple[DataProto, Dict]:
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
        mas_code_generation_output: Output from first-level generation
    
    Returns:
        Tuple of (mas_code_generation_output, question_results) where question_results
        contains the complete tree structure with all levels and the following additional keys:
        - _total_response_keys: Combined list of all response keys
    """
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("GENERATING AND EXTRACTING CODES WITH TWO-LEVEL TREE ARCHITECTURE")
    main_rank_print(f"{'='*60}")

    
    # Generate sequences for first level
    main_rank_print("🚀 FIRST LEVEL: Generating initial responses...")

    original_questions_list = list(question_results.keys()) # before we do anything to question_results

    # Extract sub-tasks and sub-agents from first-level generation
    main_rank_print("🔍 FIRST LEVEL: Extracting sub-tasks and sub-agents...")
    codes = extract_codes(trainer_instance, mas_code_generation_output, question_results)
    
    main_rank_print(f"📊 FIRST LEVEL COMPLETE: {len(codes)} codes extracted")
    
    # SECOND LEVEL: Generate new sub-tasks using batch processing
    sub_agent_results = []
    if codes:
        main_rank_print(f"\n{'='*60}")
        main_rank_print("SECOND LEVEL: BATCH GENERATION OF NEW SUB-TASKS")
        main_rank_print(f"{'='*60}")
        
        try:
            # Initialize batch manager for second-level generation
            manager = SubAgentBatchManager(trainer_instance)
            main_rank_print(f"🔄 New sub-agents per sub-task: {manager.new_sub_agents_per_sub_task}")
            
            # Generate all sub-agents in a single batch operation
            sub_agent_results = manager.generate_all_sub_agents_batch(codes, question_results)
            
            successful_count = len([r for r, _, _ in sub_agent_results if r is not None])
            total_attempts = len(codes) * manager.new_sub_agents_per_sub_task
            
            main_rank_print(f"🚀 SECOND LEVEL COMPLETE: {successful_count}/{total_attempts} sub-task rollouts successful")
            
            # Log performance metrics
            if successful_count > 0:
                success_rate = (successful_count / total_attempts) * 100
                main_rank_print(f"📈 Tree architecture efficiency: {success_rate:.1f}% ({successful_count}/{total_attempts})")
                
                # Show breakdown by sub-task
                # for sub_task in codes:
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
        
    # For sequential architecture, Level 1 responses already exist and codes are extracted by extract_codes function
    # No need to process Level 1 responses here since we only care about Level 2 generation
    # Process Second Level Results (Sub-agent rollouts - Level 2)
    if sub_agent_results:
        
        main_rank_print(f"\n{'='*60}")
        main_rank_print("PROCESSING SECOND LEVEL RESULTS (NEW SUB-TASK GENERATION - LEVEL 2)")
        main_rank_print(f"{'='*60}")
        
        # Process each sub-agent result from second-level rollouts
        # Track responses per question (like extraction.py does)
        question_response_counts = {}  # Track response count per question
        
        for sub_agent_output, sub_task, _ in sub_agent_results:
            if sub_agent_output is not None:
                # main_rank_print(f"🔄 Processing sequential rollout for '{sub_task}'...")
                
                # Extract codes from rollout output (similar to generate_and_extract_codes)
                sub_agent_length = get_safe_length(sub_agent_output, "sub_agent_output")
                main_rank_print(f"📊 Sequential rollout output shape: {sub_agent_output.batch['responses'].shape}; response_count: {sub_agent_length}")
                
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
                            raise RuntimeError(error_msg)
                        
                        # Create unique key for this second-level result
                        # Use format that GRPO mode expects: {question}<<MySep>>{response_idx} (like extraction.py)
                        # Track response count per question
                        if original_question not in question_response_counts:
                            question_response_counts[original_question] = 0
                        question_response_counts[original_question] += 1
                        
                        sub_agent_key = f"{original_question}<<MySep>>{question_response_counts[original_question] - 1}"
                        
                        # Store sub-agent results in question_results following extraction.py pattern
                        # No level distinction - just store as regular responses
                        question_results[sub_agent_key] = {
                            'response_text': response_text,
                            'extracted_code_data': extracted_code_data,
                            'question': original_question,
                            'response_idx': question_response_counts[original_question] - 1,
                            'ground_truth': question_results[original_question].get('ground_truth', ''),
                            'original_index': original_question_idx,
                            'is_validation': False
                            # No level, parent_sub_task, etc. - keep it simple like extraction.py
                        }
                        
                        if code is not None:
                            main_rank_print(f"✓ Sequential Response {question_response_counts[original_question]}: Valid code extracted")
                        else:
                            main_rank_print(f"✗ Sequential Response {question_response_counts[original_question]}: No valid code extracted")
                            
                    except Exception as e:
                        main_rank_print(f"✗ Sequential Response: Code extraction failed: {e}")
                        
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
                        # Use format that GRPO mode expects: {question}<<MySep>>{response_idx} (like extraction.py)
                        # Track response count per question
                        if original_question not in question_response_counts:
                            question_response_counts[original_question] = 0
                        question_response_counts[original_question] += 1
                        
                        sub_agent_key = f"{original_question}<<MySep>>{question_response_counts[original_question] - 1}"
                        
                        # Store failed sub-agent result following extraction.py pattern
                        question_results[sub_agent_key] = {
                            'response_text': "Unknown",
                            'extracted_code_data': failed_extracted_code_data,
                            'question': original_question,  # Use the found original question
                            'response_idx': question_response_counts[original_question] - 1,
                            'ground_truth': question_results[original_question].get('ground_truth', ''),
                            'original_index': original_question_idx,
                            'is_validation': False
                            # No level, parent_sub_task, etc. - keep it simple like extraction.py
                        }
            else:
                main_rank_print(f"❌ Skipping failed sequential rollout")
    
    # Summary of sequential architecture
    if sub_agent_results:
        main_rank_print(f"\n{'='*60}")
        main_rank_print("SEQUENTIAL ARCHITECTURE SUMMARY")
        main_rank_print(f"{'='*60}")
        
        # Count actual responses that were processed
        response_count = 0
        success_count = 0
        
        for key, value in question_results.items():
            if isinstance(value, dict) and 'extracted_code_data' in value:
                response_count += 1
                if value.get('extracted_code_data', {}).get('code_extraction_success', False):
                    success_count += 1
        
        main_rank_print(f"✅ Sequential Architecture Summary:")
        main_rank_print(f"   - Original questions: {len(original_questions_list)}")
        main_rank_print(f"   - Generated responses: {response_count} responses processed ({success_count} successful code extractions)")
        
        main_rank_print(f"📊 Total responses: {response_count}")
        
    # For sequential architecture, we only return Level 2 responses (not combined Level 1 + Level 2)
    # Keys follow {question}<<MySep>>{response_idx} format like extraction.py for Standard GRPO Mode
    
    # Collect response keys from the processed sub-agent results (needed for function call)
    response_keys = []
    for key, value in question_results.items():
        if isinstance(value, dict) and 'extracted_code_data' in value:
            response_keys.append(key)
    
    main_rank_print(f"🔑 Sequential architecture: Collected {len(response_keys)} response keys")
    
    if sub_agent_results:
        # Create a DataProto containing the generated responses
        response_dataproto = _create_level_2_only_dataproto(
            trainer_instance, reward_manager, sub_agent_results, response_keys, mas_code_generation_output
        )
        
        main_rank_print(f"✅ Created sequential response DataProto with {len(response_keys)} responses")
        main_rank_print(f"   - Using unified_group UID strategy")
        main_rank_print(f"   - Responses inherit parent UIDs")
        
        return response_dataproto, question_results
    else:
        # This should never happen in sequential architecture - raise error instead of fallback
        error_msg = f"Sequential architecture enabled but no responses found. This indicates a fundamental failure in the sequential generation process."
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


def _create_level_2_only_dataproto(trainer_instance, reward_manager, sub_agent_results: List[Tuple[DataProto, str, str]], 
                                  level_2_keys: List[str], mas_code_generation_output: DataProto) -> DataProto:
    """
    Create a DataProto containing only sequential responses using unified_group UID strategy.
    
    In sequential architecture, we only return Level 2 responses (not combined Level 1 + Level 2).
    The UID planning follows the unified_group strategy where Level 2 responses inherit their
    Level 1 parent UIDs.
    
    Args:
        trainer_instance: The trainer instance for accessing configuration
        reward_manager: The reward manager instance for accessing UID planning strategy
        sub_agent_results: List of (DataProto, sub_task, _) tuples for sequential responses
        level_2_keys: List of sequential response keys
        mas_code_generation_output: DataProto from first-level generation containing parent UIDs
        
    Returns:
        DataProto containing only sequential responses with proper UID planning
    """
    main_rank_print(f"\n{'='*60}")
    main_rank_print(f"CREATING SEQUENTIAL DATAPROTO WITH {len(level_2_keys)} RESPONSES")
    main_rank_print(f"Using UNIFIED_GROUP UID strategy")
    main_rank_print(f"{'='*60}")
    
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
    level_2_batch = {}
    for key in first_sub_agent_result.batch.keys():
        # Collect all Level 2 tensors for this field
        level_2_tensor_list = []
        
        for sub_agent_output, sub_task, _ in sub_agent_results:
            if sub_agent_output is None:
                error_msg = f"Sequential rollout output is None for sub_task '{sub_task}'. This indicates a fundamental failure in sequential generation."
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
            level_2_batch[key] = combined_level_2_tensor
            main_rank_print(f"✓ Extracted Level 2 tensor '{key}': {combined_level_2_tensor.shape}")
        else:
            error_msg = f"No valid Level 2 tensors found for field '{key}'. This indicates a fundamental failure in Level 2 tensor extraction."
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    # Create the Level 2 only DataProto
    level_2_dataproto = DataProto.from_single_dict(level_2_batch)
    
    # Copy non_tensor_batch from the first sub_agent_result as a template
    if hasattr(first_sub_agent_result, 'non_tensor_batch'):
        level_2_dataproto.non_tensor_batch = {}
        # We'll populate this with the actual Level 2 data
    
    # Define total_responses early so it's available for both training and validation modes
    total_responses = len(level_2_dataproto)
    expected_responses = len(level_2_keys)
    
    main_rank_print(f"🎯 Level 2 only DataProto created successfully:")
    main_rank_print(f"   Expected responses: {expected_responses}")
    main_rank_print(f"   Actual responses: {total_responses}")
    main_rank_print(f"   Tensor fields: {list(level_2_batch.keys())}")
    
    if total_responses != expected_responses:
        error_msg = f"Level 2 DataProto size mismatch: expected {expected_responses}, got {total_responses}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # CRITICAL: Create UIDs for Level 2 responses using unified_group strategy
    # In unified_group, Level 2 responses inherit their Level 1 parent UIDs
    main_rank_print(f"🎯 Planning UIDs for Level 2 responses using unified_group strategy...")
    
    # Get configuration for UID planning
    # Training: rollout.n responses per question
    rollout_n = trainer_instance.config.actor_rollout_ref.rollout.n
    new_sub_agents_per_sub_task = trainer_instance.sub_agents_per_sub_task
    
    # Calculate how many Level 2 responses we need per Level 1 parent
    total_level_2_responses = len(level_2_keys)
    expected_level_1_parents = total_level_2_responses // new_sub_agents_per_sub_task
    
    main_rank_print(f"   - Total Level 2 responses: {total_level_2_responses}")
    main_rank_print(f"   - Expected Level 1 parents: {expected_level_1_parents}")
    main_rank_print(f"   - Sub-agents per sub-task: {new_sub_agents_per_sub_task}")
    
    # For sequential architecture with unified_group strategy, Level 2 responses inherit their Level 1 parent UIDs
    # We need to get the actual UIDs from the Level 1 responses that these Level 2 responses are based on
    
    # TRAINING MODE: Do UID planning to create proper UIDs for Level 2 responses
    main_rank_print(f"🎯 TRAINING MODE: Planning UIDs for Level 2 responses...")
    # Extract parent UIDs from Level 1 responses for UID planning
    if hasattr(mas_code_generation_output, 'non_tensor_batch') and 'uid' in mas_code_generation_output.non_tensor_batch:
        parent_uids = mas_code_generation_output.non_tensor_batch['uid']
        main_rank_print(f"✓ Extracted {len(parent_uids)} parent UIDs from Level 1 responses")
    else:
        error_msg = "No UIDs found in Level 1 output. Cannot create Level 2 responses without parent UIDs."
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
        
    
    # Use the existing UID planning functions from uid_planning.py
    try:
        # Determine UID planning strategy - for sequential architecture, we use unified_group
        strategy = 'unified_group'
        main_rank_print(f"🎯 Using UID planning strategy: {strategy}")
        
        # Get the appropriate UID planning function
        uid_planning_func = get_uid_planning_function(strategy)
        
        # For sequential architecture, we need to create a mapping that represents the Level 1 structure
        # The uid_planning_func expects level_1_keys and level_2_keys, but we only have Level 2
        # So we create dummy level_1_keys that represent the parent structure
        
        # Create dummy level_1_keys that represent the parent structure
        dummy_level_1_keys = [f"parent_{i:04d}" for i in range(expected_level_1_parents)]
        
        # Plan UIDs using the selected strategy
        # This will create UIDs where Level 2 responses inherit their Level 1 parent UIDs
        expanded_uids, uid_mapping_info = uid_planning_func(
            parent_uids, dummy_level_1_keys, level_2_keys, trainer_instance
        )
        
        # Extract only the Level 2 UIDs (the inherited ones)
        # The expanded_uids contains [parent_uids + inherited_uids], we want only the inherited part
        level_2_uids = expanded_uids[len(parent_uids):]
        
        # Validate that we got the right number of Level 2 UIDs
        if len(level_2_uids) != total_level_2_responses:
            error_msg = f"UID planning returned {len(level_2_uids)} Level 2 UIDs, expected {total_level_2_responses}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Set the UIDs in the DataProto
        level_2_dataproto.non_tensor_batch['uid'] = np.array(level_2_uids, dtype=object)
        
        # Validate the UID planning results
        validate_uid_planning(expanded_uids, dummy_level_1_keys, level_2_keys, uid_mapping_info, trainer_instance)
        
        main_rank_print(f"✅ UID planning completed using {strategy} strategy")
        main_rank_print(f"   - Parent UIDs: {len(parent_uids)}")
        main_rank_print(f"   - Level 2 UIDs: {len(level_2_uids)} (inherited from parents)")
        main_rank_print(f"   - Each Level 2 response inherits its Level 1 parent UID")
        
    except Exception as e:
        error_msg = f"Failed to plan UIDs using {strategy} strategy: {e}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


        # Required non-tensor fields (these are expected by the trainer and reward manager)
        required_non_tensor_fields = ['uid']
        for field in required_non_tensor_fields:
            if field not in level_2_dataproto.non_tensor_batch:
                error_msg = f"Missing required non-tensor field '{field}' in Level 2 DataProto"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            field_size = len(level_2_dataproto.non_tensor_batch[field])
            if field_size != total_responses:
                error_msg = f"Non-tensor field '{field}' size mismatch: expected {total_responses}, got {field_size}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            main_rank_print(f"   ✓ Required non-tensor field '{field}': {field_size} entries")


    # Copy meta_info from original Level 1 output (same as extraction_tree.py)
    # This should happen after UID planning to ensure proper order
    if hasattr(mas_code_generation_output, 'meta_info') and mas_code_generation_output.meta_info:
        level_2_dataproto.meta_info = mas_code_generation_output.meta_info.copy()
        main_rank_print(f"✓ Copied meta_info from Level 1 output: {list(level_2_dataproto.meta_info.keys())}")
    else:
        main_rank_print(f"⚠️  No meta_info found in Level 1 output")
        # Initialize with basic meta_info
        level_2_dataproto.meta_info = {
            'eos_token_id': trainer_instance.tokenizer.eos_token_id,
            'pad_token_id': trainer_instance.tokenizer.pad_token_id,
        }
        main_rank_print(f"✓ Initialized basic meta_info with tokenizer settings")


    # Handle meta_info - combine from all Level 2 responses (similar to extraction_tree.py)
    # Note: meta_info is now copied from Level 1 output above
    
    # Special handling for global_token_num - combine from all Level 2 responses
    if hasattr(first_sub_agent_result, 'meta_info') and 'global_token_num' in first_sub_agent_result.meta_info:
        combined_global_token_num = []
        
        for sub_agent_output, _, _ in sub_agent_results:
            if sub_agent_output is not None and hasattr(sub_agent_output, 'meta_info') and 'global_token_num' in sub_agent_output.meta_info:
                combined_global_token_num.extend(sub_agent_output.meta_info['global_token_num'])
            else:
                raise RuntimeError(f"No global_token_num found in Level 2 meta_info for sub_agent_output: {sub_agent_output}")
        
        # Replace Level 1 global_token_num with Level 2 combined values
        level_2_dataproto.meta_info['global_token_num'] = combined_global_token_num
        main_rank_print(f"✓ Combined global_token_num from {len(sub_agent_results)} Level 2 responses: {len(combined_global_token_num)} total tokens")
        
        # Validate that the combined length matches our expected total
        if len(combined_global_token_num) != total_responses:
            error_msg = f"global_token_num length mismatch: expected {total_responses}, got {len(combined_global_token_num)}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    else:
        main_rank_print(f"⚠️  No global_token_num found in Level 2 meta_info")
    
    # Ensure essential meta_info fields are present
    if 'eos_token_id' not in level_2_dataproto.meta_info:
        level_2_dataproto.meta_info['eos_token_id'] = trainer_instance.tokenizer.eos_token_id
    if 'pad_token_id' not in level_2_dataproto.meta_info:
        level_2_dataproto.meta_info['pad_token_id'] = trainer_instance.tokenizer.pad_token_id
    main_rank_print(f"✓ Ensured essential tokenizer meta_info fields are present")
    

    
    # Final validation: ensure all required fields are present and have correct sizes
    main_rank_print(f"\n🔍 Final validation of Level 2 only DataProto...")
    
    # Required tensor fields (these are the core fields the trainer expects)
    required_tensor_fields = ['input_ids', 'attention_mask', 'responses']
    for field in required_tensor_fields:
        if field not in level_2_dataproto.batch:
            error_msg = f"Missing required tensor field '{field}' in Level 2 DataProto"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        field_size = level_2_dataproto.batch[field].size(0)
        if field_size != total_responses:
            error_msg = f"Tensor field '{field}' size mismatch: expected {total_responses}, got {field_size}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✓ Tensor field '{field}': {level_2_dataproto.batch[field].shape}")
    

    
    main_rank_print(f"✅ Successfully created Level 2 only DataProto with {total_responses} responses")
    main_rank_print(f"✅ Using unified_group UID strategy - Level 2 responses inherit parent UIDs")
    main_rank_print(f"{'='*60}")
    
    return level_2_dataproto
