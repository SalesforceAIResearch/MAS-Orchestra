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
Utility functions and helper methods for MAS-R1 Trainer.
This file contains helper methods that can be extracted from the main trainer
to keep the main trainer file focused on core PPO logic.
"""

import os
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from verl import DataProto
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from mas_r1_reasoner.agents.common import main_rank_print
import re
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code

def from_question_results_to_execution_results(question_results: Dict, mas_code_generation_output: DataProto) -> DataProto:
    """
    Create DataProto from question_results dictionary.
    This function converts the dictionary back to DataProto format for VERL compatibility.
    Supports both original question structure and new {question}<<MySep>>{response} structure for GRPO.
    From question_results to tensor with execution results and reward model.
    
    Args:
        question_results: Dictionary containing question data with execution results
        mas_code_generation_output: DataProto containing generation outputs
        
    Returns:
        DataProto with execution results and reward model
    """
    main_rank_print(f"\n{'='*60}")
    main_rank_print("CREATING DATAPROTO FROM QUESTION RESULTS")
    # main_rank_print(f"question_results: {question_results}")
    main_rank_print(f"{'='*60}")
    
    
    # Determine the processing mode and extract keys
    mode_info = _determine_processing_mode(question_results)
    
    # Use questions_set directly as questions_list since we'll reorder everything anyway
    # No need to maintain any specific order from mas_code_generation_output

    main_rank_print(f"Training mode: Using questions_set directly as questions_list")
    main_rank_print(f"  - questions_set size: {len(mode_info['questions_set'])}")
        
    questions_list = list(mode_info['questions_set'])
    main_rank_print(f"  - questions_list: {questions_list}")
    # input_ids = input_data.batch['input_ids']
    # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    # main_rank_print(f"  - input_texts: {input_texts}")

    # Determine if this is training (multiple responses) or validation (single response)
    has_multiple_responses = _determine_multiple_response_mode(
        mode_info, questions_list, question_results
    )
    
    # Process responses based on the detected mode using existing helper functions
    if mode_info['enable_tree_architecture']:
        main_rank_print(f"🌳 TREE ARCHITECTURE MODE: Processing Level 1 + Level 2 responses")
        main_rank_print(f"Total questions: {len(questions_list)}")
        main_rank_print(f"Total execution results: {mode_info['total_execution_results']}")
        main_rank_print(f"Level 1 responses: {len(mode_info['level_1_keys'])}")
        main_rank_print(f"Level 2 responses: {len(mode_info['level_2_keys'])}")
        main_rank_print(f"Average responses per question: {mode_info['total_execution_results'] / len(questions_list):.1f}")
        
        responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size = _process_tree_architecture_responses(
            question_results, mode_info['level_1_keys'], mode_info['level_2_keys'], questions_list, mas_code_generation_output
        )
        
    elif has_multiple_responses:
        main_rank_print(f"📊 STANDARD GRPO MODE: Processing multiple responses per question")
        responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size = _process_standard_grpo_responses(
            question_results, questions_list, mas_code_generation_output
        )
    
    else:
        main_rank_print(f"SINGLE RESPONSE MODE: Processing one response per question")
        main_rank_print(f"Total questions: {len(questions_list)}")
        main_rank_print(f"Total execution results: {mode_info['total_execution_results']}")
        
        batch_size = len(questions_list)
        main_rank_print(f"📊 Simple mode: Batch size based on original questions: {batch_size}")
        
        responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size = _process_simple_mode_responses(
            question_results, questions_list, mas_code_generation_output
        )
    
    # import time
    # time.sleep(1000)
    # Log the final batch size being processed
    _log_batch_processing_summary(batch_size, mode_info)
    
    # Create the final DataProto with all tensor fields
    final_output = _create_final_dataproto(
        responses_tensor, execution_results, execution_stats_list, reward_model_list, 
        batch_size, mas_code_generation_output
    )
    
    # Call appropriate diversity check based on mode
    if mode_info['enable_tree_architecture']:
        _check_tree_architecture_diversity(
            question_results, mode_info['level_1_keys'], mode_info['level_2_keys'], 
            execution_results, execution_stats_list
        )
    elif has_multiple_responses:
        _check_standard_grpo_diversity(question_results, execution_results, execution_stats_list)
    else:
        main_rank_print(f"SINGLE RESPONSE MODE: One response per question - no diversity check needed")
        main_rank_print(f"Total questions: {len(questions_list)}")
        main_rank_print(f"✓ Single response mode with one response per question")
    
    main_rank_print(f"{'='*40}")
    main_rank_print(f"{'='*60}\n")
    
    return final_output


def _determine_processing_mode(question_results: Dict) -> Dict:
    """
    Determine the processing mode and extract relevant keys from question_results.
    
    Args:
        question_results: Dictionary containing question data
        
    Returns:
        Dictionary with mode information and extracted keys
    """
    # Check if tree architecture mode is enabled
    if '_level_1_keys' in question_results and '_level_2_keys' in question_results:
        # Tree architecture mode: Use pre-collected keys
        level_1_keys = question_results['_level_1_keys']
        level_2_keys = question_results['_level_2_keys']
        all_response_keys = level_1_keys + level_2_keys
        enable_tree_architecture = True
        
        main_rank_print(f"🔑 Using pre-collected keys from extraction_tree.py (tree architecture mode)")
        main_rank_print(f"   Level 1 keys: {len(level_1_keys)}")
        main_rank_print(f"   Level 2 keys: {len(level_2_keys)}")
        main_rank_print(f"   Total response keys: {len(all_response_keys)}")
        
        # Extract questions from all response keys
        questions_set = _extract_questions_from_tree_keys(all_response_keys)
        
    else:
        # Simple mode: only Level 1 responses
        level_1_keys = []
        level_2_keys = []
        enable_tree_architecture = False
        
        for key in question_results.keys():
            # Check if this is a Level 1 response key (format: {question}<<MySep>>{idx})
            if '<<MySep>>' in key and key.split('<<MySep>>')[-1].isdigit():
                level_1_keys.append(key)
        
        questions_set = set()
        for key in level_1_keys:
            # Split using the robust separator
            question = key.rsplit('<<MySep>>', 1)[0]
            questions_set.add(question)
        
        main_rank_print(f"📊 Simple mode - Level 1 responses only")
        main_rank_print(f"📊 Code collection:")
        main_rank_print(f"   Level 1 responses: {len(level_1_keys)}")
        main_rank_print(f"   Total responses: {len(level_1_keys)}")
    
    # Log the mode being used for batch formatting
    main_rank_print(f"\n{'='*40}")
    main_rank_print("BATCH FORMATTING MODE")
    main_rank_print(f"{'='*40}")
    if enable_tree_architecture:
        main_rank_print(f"🌳 TREE ARCHITECTURE MODE: Processing Level 1 + Level 2 responses")
        main_rank_print(f"   Questions/sub-tasks: {len(questions_set)}")
        main_rank_print(f"   Level 1 keys: {len(level_1_keys)}")
        main_rank_print(f"   Level 2 keys: {len(level_2_keys)}")
    else:
        main_rank_print(f"📊 SIMPLE MODE: Processing Level 1 responses only")
        main_rank_print(f"   Questions: {len(questions_set)}")
        main_rank_print(f"   Level 1 keys: {len(level_1_keys)}")
    main_rank_print(f"{'='*40}\n")
    
    # Calculate total execution results
    if enable_tree_architecture:
        total_execution_results = len(level_1_keys) + len(level_2_keys)
    else:
        total_execution_results = len(level_1_keys)
    
    return {
        'enable_tree_architecture': enable_tree_architecture,
        'level_1_keys': level_1_keys,
        'level_2_keys': level_2_keys,
        'questions_set': questions_set,
        'total_execution_results': total_execution_results
    }


def _determine_multiple_response_mode(mode_info: Dict, questions_list: List[str], question_results: Dict) -> bool:
    """
    Determine if this is training (multiple responses) or validation (single response).
    
    Args:
        mode_info: Dictionary with mode information
        questions_list: List of questions in original batch order
        question_results: Dictionary containing question data
        
    Returns:
        True if multiple responses per question, False otherwise
    """
    if mode_info['enable_tree_architecture']:
        total_execution_results = mode_info['total_execution_results']
    else:
        # Count total execution results
        total_execution_results = 0
        for question in questions_list:
            # Count how many {question}<<MySep>>{response} keys exist for this question
            for key in question_results.keys():
                if key.startswith(f"{question}<<MySep>>") and key.split('<<MySep>>')[-1].isdigit():
                    # main_rank_print(f"key: {key}")
                    total_execution_results += 1

    main_rank_print(f"total_execution_results: {total_execution_results} vs. len(questions_list): {len(questions_list)}")

    # If total execution results equals number of questions, it's single response mode
    # If total execution results > number of questions, it's multiple response mode
    has_multiple_responses = total_execution_results > len(questions_list)
    return has_multiple_responses


def _extract_questions_from_tree_keys(all_response_keys: List[str]) -> set:
    """
    Extract unique questions from tree architecture response keys.
    
    Args:
        all_response_keys: List of response keys from tree architecture
        
    Returns:
        Set of unique question identifiers
    """
    questions_set = set()
    
    for key in all_response_keys:
        if '<<MySep>>' in key and key.split('<<MySep>>')[-1].isdigit():
            if key.startswith('second_level<<MySep>>'):
                # Level 2 response: second_level<<MySep>>{sub_task_id}<<MySep>>{response_idx}
                parts = key.split('<<MySep>>')
                if len(parts) >= 3:
                    sub_task_id = parts[1]  # Middle part is sub_task_id
                    questions_set.add(sub_task_id)
            else:
                # Level 1 response: {question}<<MySep>>{response_idx}
                question = key.rsplit('<<MySep>>', 1)[0]
                questions_set.add(question)
    
    main_rank_print(f"🌳 Tree architecture: {len(questions_set)} unique questions/sub-tasks identified")
    return questions_set


def _log_batch_processing_summary(batch_size: int, mode_info: Dict):
    """
    Log the final batch size being processed.
    
    Args:
        batch_size: Final batch size
        mode_info: Dictionary with mode information
    """
    main_rank_print(f"\n{'='*40}")
    main_rank_print("FINAL BATCH PROCESSING SUMMARY")
    main_rank_print(f"{'='*40}")
    if mode_info['enable_tree_architecture']:
        main_rank_print(f"🌳 Tree Architecture Mode:")
        main_rank_print(f"   Batch size: {batch_size} (all responses)")
        main_rank_print(f"   Level 1 responses: {len(mode_info['level_1_keys'])}")
        main_rank_print(f"   Level 2 responses: {len(mode_info['level_2_keys'])}")
        main_rank_print(f"   Total responses: {len(mode_info['level_1_keys']) + len(mode_info['level_2_keys'])}")
    else:
        main_rank_print(f"📊 Simple Mode:")
        main_rank_print(f"   Batch size: {batch_size} (original questions)")
        main_rank_print(f"   Level 1 responses: {len(mode_info['level_1_keys'])}")
    
    main_rank_print(f"Converting {batch_size} responses to DataProto format for validation")
    main_rank_print(f"{'='*40}\n")


def _create_final_dataproto(responses_tensor, execution_results, execution_stats_list, reward_model_list, 
                           batch_size, mas_code_generation_output):
    """
    Create the final DataProto with all necessary tensor fields and execution results.
    
    Args:
        responses_tensor: List of response tensors
        execution_results: List of execution results
        execution_stats_list: List of execution statistics
        reward_model_list: List of reward model data
        batch_size: Expected batch size
        mas_code_generation_output: DataProto containing generation outputs
        
    Returns:
        DataProto with all tensor fields and execution results
    """
    # Stack responses tensor
    if responses_tensor:
        responses_tensor = torch.stack(responses_tensor, dim=0)
    else:
        raise RuntimeError("No responses tensor found")
    
    # Create batch dictionary with all necessary tensor fields
    batch_dict = _create_batch_dict_with_tensor_fields(
        responses_tensor, batch_size, mas_code_generation_output, execution_results
    )
    
    # Create DataProto
    final_output = DataProto.from_single_dict(batch_dict)
    
    # Log what tensor fields were included
    main_rank_print(f"✓ DataProto tensor fields included:")
    for key, tensor in batch_dict.items():
        main_rank_print(f"  - {key}: {tensor.shape}")
    
    # Add non-tensor data - REORDER to match execution_results order
    if mas_code_generation_output and hasattr(mas_code_generation_output, 'non_tensor_batch'):
        # Reorder all non-tensor fields from mas_code_generation_output to match execution_results order, including UID
        reordered_non_tensor_fields = _reorder_non_tensor_fields_to_match_execution_results(
            mas_code_generation_output.non_tensor_batch, execution_results, mas_code_generation_output
        )
        
        # Add reordered non-tensor fields
        for key, value in reordered_non_tensor_fields.items():
            final_output.non_tensor_batch[key] = value
            main_rank_print(f"✓ Reordered non-tensor field '{key}' to match execution_results order: {len(value) if hasattr(value, '__len__') else 'scalar'} entries")
    else:
        main_rank_print("non_tensor_batch missing from mas_code_generation_output - this should never happen in training, or it is validation")
    
    final_output.non_tensor_batch.update({
        'reward_model': np.array(reward_model_list, dtype=object),
        'execution_results': np.array(execution_results, dtype=object),
        'execution_stats': np.array(execution_stats_list, dtype=object),
        'data_source': np.array(['math'] * len(execution_results), dtype=object),  # Add data_source field for reward manager
    })
    
    # Set meta_info - preserve all existing fields from mas_code_generation_output
    if mas_code_generation_output and hasattr(mas_code_generation_output, 'meta_info'):
        # Copy all existing meta_info fields
        final_output.meta_info = mas_code_generation_output.meta_info.copy()
        main_rank_print(f"✓ Preserved all meta_info fields from mas_code_generation_output: {list(final_output.meta_info.keys())}")
    else:
        raise RuntimeError("meta_info missing from mas_code_generation_output - this should never happen")
    
    # Add/update MAS-R1 specific fields
    final_output.meta_info['mas_r1_data_source'] = 'math'
    main_rank_print(f"✓ Added mas_r1_data_source to meta_info")
    
    # Validate and log the final structure
    _validate_and_log_final_dataproto(final_output, execution_results)
    
    return final_output


def _create_batch_dict_with_tensor_fields(responses_tensor, batch_size, mas_code_generation_output, execution_results):
    """
    Create batch dictionary with all necessary tensor fields from mas_code_generation_output.
    Reorder tensors to match execution_results order.
    
    Args:
        responses_tensor: Stacked responses tensor
        batch_size: Expected batch size
        mas_code_generation_output: DataProto containing generation outputs
        execution_results: List of execution results for reordering
        
    Returns:
        Dictionary with all tensor fields
    """
    batch_dict = {
        'responses': responses_tensor
    }
    
    # Add other tensor fields from mas_code_generation_output if available
    if mas_code_generation_output and hasattr(mas_code_generation_output, 'batch'):
        for key, tensor in mas_code_generation_output.batch.items():
            if key not in batch_dict and tensor is not None:
                # Always reorder tensors to match execution_results order when available
                if execution_results and len(execution_results) == batch_size:
                    # Create reordered tensor based on execution_results order
                    reordered_tensor = _reorder_tensor_to_match_execution_results(
                        tensor, execution_results, mas_code_generation_output
                    )
                    if reordered_tensor is not None:
                        batch_dict[key] = reordered_tensor
                        main_rank_print(f"✓ Reordered {key} tensor to match execution_results order: {reordered_tensor.shape}")
                    else:
                        error_msg = f"Failed to reorder {key} tensor to match execution_results order"
                        raise RuntimeError(error_msg)
                else:
                    # Fallback: use tensor as-is only if no execution_results available
                    batch_dict[key] = tensor
                    main_rank_print(f"✓ Using {key} tensor as-is (no execution_results for reordering): {tensor.shape}")
    
    return batch_dict


def _reorder_non_tensor_fields_to_match_execution_results(non_tensor_batch, execution_results, mas_code_generation_output):
    """
    Reorder non-tensor fields from mas_code_generation_output to match execution_results order.
    
    Args:
        non_tensor_batch: Dictionary of non-tensor fields from mas_code_generation_output
        execution_results: List of execution results in the desired order
        mas_code_generation_output: Original DataProto for reference
        
    Returns:
        Dictionary of reordered non-tensor fields
    """
    try:
        # Create a mapping from question to ALL original indices in mas_code_generation_output
        question_to_original_indices = {}
        if hasattr(mas_code_generation_output, 'non_tensor_batch') and 'question' in mas_code_generation_output.non_tensor_batch:
            for idx, question in enumerate(mas_code_generation_output.non_tensor_batch['question']):
                if question not in question_to_original_indices:
                    question_to_original_indices[question] = []
                question_to_original_indices[question].append(idx)
        
        # Create reordered indices based on execution_results
        reordered_indices = []
        question_usage_count = {}  # Track how many times we've used each question
        
        for exec_result in execution_results:
            question = exec_result.get('question', '')
            if question in question_to_original_indices:
                # Get the next available index for this question
                available_indices = question_to_original_indices[question]
                usage_count = question_usage_count.get(question, 0)
                
                if usage_count < len(available_indices):
                    # Use the next available index for this question
                    reordered_indices.append(available_indices[usage_count])
                    question_usage_count[question] = usage_count + 1
                else:
                    raise RuntimeError(f"Warning: Used all occurrences of question '{question}', reusing last occurrence")
            else:
                raise RuntimeError(f"Warning: Question '{question}' not found in original batch, using index 0")
        
        # Reorder all non-tensor fields
        reordered_fields = {}
        for key, value in non_tensor_batch.items():
            if isinstance(value, (list, np.ndarray)) and len(value) == len(mas_code_generation_output.non_tensor_batch.get('question')):
                # This field needs reordering (those not in the same legngth as quesiton, like meta-info, do not need to reorder)
                if isinstance(value, np.ndarray):
                    reordered_fields[key] = np.array([value[i] for i in reordered_indices], dtype=value.dtype)
                else:
                    reordered_fields[key] = [value[i] for i in reordered_indices]
                main_rank_print(f"✓ Reordered non-tensor field '{key}': {len(reordered_fields[key])} entries")
            else:
                # This field doesn't need reordering (scalar or different length)
                reordered_fields[key] = value
                main_rank_print(f"✓ Kept non-tensor field '{key}' as-is (no reordering needed)")
        
        return reordered_fields
        
    except Exception as e:
        main_rank_print(f"Error reordering non-tensor fields: {e}")
        # Fallback: return original fields
        return non_tensor_batch


def _reorder_tensor_to_match_execution_results(tensor, execution_results, mas_code_generation_output):
    """
    Reorder tensor from mas_code_generation_output to match execution_results order.
    
    Args:
        tensor: Original tensor from mas_code_generation_output
        execution_results: List of execution results in the desired order
        mas_code_generation_output: Original DataProto for reference
        
    Returns:
        Reordered tensor matching execution_results order, or None if reordering fails
    """
    try:
        # Create a mapping from question to ALL original indices in mas_code_generation_output
        question_to_original_indices = {}
        if hasattr(mas_code_generation_output, 'non_tensor_batch') and 'question' in mas_code_generation_output.non_tensor_batch:
            for idx, question in enumerate(mas_code_generation_output.non_tensor_batch['question']):
                if question not in question_to_original_indices:
                    question_to_original_indices[question] = []
                question_to_original_indices[question].append(idx)
        
        # Create reordered indices based on execution_results
        reordered_indices = []
        question_usage_count = {}  # Track how many times we've used each question
        
        for exec_result in execution_results:
            question = exec_result.get('question', '')
            if question in question_to_original_indices:
                # Get the next available index for this question
                available_indices = question_to_original_indices[question]
                usage_count = question_usage_count.get(question, 0)
                
                if usage_count < len(available_indices):
                    # Use the next available index for this question
                    reordered_indices.append(available_indices[usage_count])
                    question_usage_count[question] = usage_count + 1
                else:
                    raise RuntimeError(f"Warning: Used all occurrences of question '{question}', reusing last occurrence")
            else:
                raise RuntimeError(f"Warning: Question '{question}' not found in original batch, using index 0")
        
        # Reorder the tensor
        if reordered_indices:
            reordered_tensor = tensor[reordered_indices]
            return reordered_tensor
        else:
            return None
            
    except Exception as e:
        main_rank_print(f"Error reordering tensor: {e}")
        return None


def _validate_and_log_final_dataproto(final_output, execution_results):
    """
    Validate the final DataProto and log its structure.
    
    Args:
        final_output: The created DataProto
        execution_results: List of execution results
    """
    # Validate required fields
    if not hasattr(final_output, 'meta_info') or not final_output.meta_info:
        raise RuntimeError("Failed to set meta_info on DataProto - this should never happen")
    
    if not hasattr(final_output, 'non_tensor_batch') or not final_output.non_tensor_batch:
        raise RuntimeError("Failed to set non_tensor_batch on DataProto - this should never happen")
    
    if 'execution_results' not in final_output.non_tensor_batch:
        raise RuntimeError("non_tensor_batch missing 'execution_results' - this should never happen")
    
    if 'uid' not in final_output.non_tensor_batch:
        main_rank_print("non_tensor_batch missing 'uid' - this should never happen, or it is validation")
    
    # Log the final structure
    main_rank_print(f"✓ DataProto created successfully")
    main_rank_print(f"  - Batch size: {len(final_output)}")
    main_rank_print(f"  - Responses shape: {final_output.batch['responses'].shape if 'responses' in final_output.batch else 'N/A'}")
    main_rank_print(f"  - Execution results: {len(execution_results)}")
    main_rank_print(f"  - Execution statistics: {len(execution_results)}")
    main_rank_print(f"  - Data source: math (stored in meta_info)")
    main_rank_print(f"  - meta_info keys: {list(final_output.meta_info.keys())}")
    main_rank_print(f"  - meta_info values: {[(k, type(v)) for k, v in final_output.meta_info.items()]}")
    main_rank_print(f"  - non_tensor_batch keys: {list(final_output.non_tensor_batch.keys())}")
    main_rank_print(f"  - non_tensor_batch types: {[(k, type(v)) for k, v in final_output.non_tensor_batch.items()]}")


def _process_tree_architecture_responses(question_results, level_1_keys, level_2_keys, questions_list, mas_code_generation_output):
    """
    Process all responses for tree architecture mode (Level 1 + Level 2).
    
    Args:
        question_results: Dictionary containing all response data
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        questions_list: List of original questions for tensor mapping
        mas_code_generation_output: DataProto containing generation outputs
        
    Returns:
        tuple: (responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size)
    """
    main_rank_print(f"🌳 Tree Architecture Mode: Processing ALL responses, not just original questions")
    main_rank_print(f"   Original questions: {len(questions_list)}")
    main_rank_print(f"   Level 1 keys: {len(level_1_keys)}")
    main_rank_print(f"   Level 2 keys: {len(level_2_keys)}")
    main_rank_print(f"   Total responses to process: {len(level_1_keys) + len(level_2_keys)}")
    
    # For tree architecture, process ALL responses (Level 1 + Level 2)
    batch_size = len(level_1_keys) + len(level_2_keys)
    main_rank_print(f"🌳 Updated batch size for tree architecture: {batch_size} (all responses)")
    
    # Prepare tensors for DataProto
    responses_tensor = []
    execution_results = []
    execution_stats_list = []
    reward_model_list = []
    
    # Process all Level 1 responses first
    for i, level_1_key in enumerate(level_1_keys):
        if level_1_key not in question_results:
            raise RuntimeError(f"Level 1 key '{level_1_key}' not found in question_results")
        
        response_data = question_results[level_1_key]
        
        # Get the response text
        response_text = response_data.get('response_text', '')
        if not response_text:
            main_rank_print(f"No response_text found for {level_1_key}; response_data: {response_data}; question_results: {question_results}")
        
        # For Level 1, try to get the corresponding tensor from mas_code_generation_output
        # Extract question from key (e.g., "question1<<MySep>>0" -> "question1")
        question = level_1_key.rsplit('<<MySep>>', 1)[0]
        try:
            original_index = questions_list.index(question)
            if original_index < len(mas_code_generation_output.batch['responses']):
                tensor = mas_code_generation_output.batch['responses'][original_index]
                # main_rank_print(f"🌳 Level 1 response {i}: Using real tensor for '{question}' with size {tensor.size()}")
                responses_tensor.append(tensor)
            else:
                error_msg = f"Level 1 response {i}: Original index {original_index} exceeds available responses ({len(mas_code_generation_output.batch['responses'])})"
                raise RuntimeError(error_msg)
        except ValueError:
            error_msg = f"Level 1 response {i}: Question '{question}' not found in original questions list"
            raise RuntimeError(error_msg)
        
        # Get execution stats for this response
        execution_stats = response_data.get('execution_stats', {})
        if not execution_stats:
            raise RuntimeError(f"No execution_stats found for {level_1_key}")
        
        # Create execution result for this response
        if execution_stats.get('total_executions', 1) == 1:
            if execution_stats.get('execution_results'):
                single_execution_results = execution_stats.get('execution_results', [])
                if single_execution_results and len(single_execution_results) > 0:
                    result, success, error = single_execution_results[0]
                    execution_result = {
                        'code': execution_stats.get('code', ''),
                        'result': result,
                        'success': success,
                        'error': error,
                        'question': question,
                        'response_idx': i,
                        'ground_truth': response_data.get('ground_truth', ''),
                        'original_index': i,
                        'is_validation': True,
                        'level': 1,
                        'parent_response_idx': response_data.get('parent_response_idx', None),
                        'parent_sub_task': response_data.get('parent_sub_task', None),
                        'response_text': response_data.get('response_text', '')  # Add response_text for sub-task extraction
                    }
                else:
                    raise RuntimeError(f"No execution results found in execution_stats for {level_1_key}")
            else:
                raise RuntimeError(f"No execution results found in execution_stats for {level_1_key}")
        else:
            raise RuntimeError(f"Not implemented: Multiple executions: use placeholder values in execution_result")

        execution_results.append(execution_result)
        execution_stats_list.append(execution_stats)
        
        # Create reward_model entry for this response
        ground_truth = response_data.get('ground_truth', '')
        if not ground_truth:
            raise RuntimeError(f"Ground truth not found for {level_1_key}")
        
        reward_model_list.append({
            'ground_truth': ground_truth
        })
    
    # Process all Level 2 responses
    for i, level_2_key in enumerate(level_2_keys):
        if level_2_key not in question_results:
            raise RuntimeError(f"Level 2 key '{level_2_key}' not found in question_results")
        
        response_data = question_results[level_2_key]
        
        # Get the response text
        response_text = response_data.get('response_text', '')
        if not response_text:
            main_rank_print(f"No response_text found for {level_2_key}; response_data: {response_data}; question_results: {question_results}")
        
        # For Level 2, get the real tensor from sub_agent_output
        sub_agent_output = response_data.get('sub_agent_output', None)
        if sub_agent_output is None:
            error_msg = f"Level 2 response {len(level_1_keys) + i}: No sub_agent_output found in response data for {level_2_key}"
            raise RuntimeError(error_msg)
        
        if 'responses' not in sub_agent_output.batch:
            error_msg = f"Level 2 response {len(level_1_keys) + i}: No 'responses' field in sub_agent_output.batch for {level_2_key}"
            raise RuntimeError(error_msg)
        
        # Get the response_idx from the response data (not the loop index)
        response_idx = response_data.get('response_idx', 0)
        if response_idx >= len(sub_agent_output.batch['responses']):
            error_msg = f"Level 2 response {len(level_1_keys) + i}: Response index {response_idx} exceeds available responses ({len(sub_agent_output.batch['responses'])}) in sub_agent_output for {level_2_key}"
            raise RuntimeError(error_msg)
        
        # Use the real tensor from sub_agent_output with the correct response_idx
        real_tensor = sub_agent_output.batch['responses'][response_idx]
        # main_rank_print(f"🌳 Level 2 response {len(level_1_keys) + i}: Using real tensor from sub_agent_output with size {real_tensor.size()} (response_idx: {response_idx})")
        responses_tensor.append(real_tensor)
        
        # Get execution stats for this response
        execution_stats = response_data.get('execution_stats', {})
        if not execution_stats:
            raise RuntimeError(f"No execution_stats found for {level_2_key}")
        
        # Create execution result for this response
        if execution_stats.get('total_executions', 1) == 1:
            if execution_stats.get('execution_results'):
                single_execution_results = execution_stats.get('execution_results', [])
                if single_execution_results and len(single_execution_results) > 0:
                    result, success, error = single_execution_results[0]
                    execution_result = {
                        'code': execution_stats.get('code', ''),
                        'result': result,
                        'success': success,
                        'error': error,
                        'question': response_data.get('question', level_2_key),
                        'response_idx': len(level_1_keys) + i,  # Continue indexing from Level 1
                        'ground_truth': response_data.get('ground_truth', ''),
                        'original_index': len(level_1_keys) + i,
                        'is_validation': True,
                        'level': 2,
                        'parent_response_idx': response_data.get('parent_response_idx', None),
                        'parent_sub_task': response_data.get('parent_sub_task', None)
                    }
                    
                    # Debug: Log tree structure information for Level 2 responses
                    # main_rank_print(f"🌳 Creating execution_result for Level 2 response {len(level_1_keys) + i}:")
                    # main_rank_print(f"  Level: {execution_result['level']}")
                    # main_rank_print(f"  Parent Response IDX: {execution_result['parent_response_idx']}")
                    # main_rank_print(f"  Parent Sub-Task: {execution_result['parent_sub_task']}")
                else:
                    raise RuntimeError(f"No execution results found in execution_stats for {level_2_key}")
            else:
                raise RuntimeError(f"No execution results found in execution_stats for {level_2_key}")
        else:
            raise RuntimeError(f"Not implemented: Multiple executions: use placeholder values in execution_result")

        
        execution_results.append(execution_result)
        execution_stats_list.append(execution_stats)
        
        # Create reward_model entry for this response
        ground_truth = response_data.get('ground_truth', '')
        if not ground_truth:
            raise RuntimeError(f"Ground truth not found for {level_2_key}")
        
        reward_model_list.append({
            'ground_truth': ground_truth
        })
    
    main_rank_print(f"🌳 Successfully processed {len(execution_results)} responses for tree architecture")
    main_rank_print(f"   Level 1 responses: {len([r for r in execution_results if r.get('level') == 1])}")
    main_rank_print(f"   Level 2 responses: {len([r for r in execution_results if r.get('level') == 2])}")
    
    # Debug: Show tensor sizes collected
    main_rank_print(f"\n🔍 Tensor Size Analysis:")
    for i, tensor in enumerate(responses_tensor):
        response_level = "Level 1" if i < len(level_1_keys) else "Level 2"
        main_rank_print(f"   Response {i}: {response_level} - Tensor size: {tensor.size()}")
    
    return responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size


def _check_tree_architecture_diversity(question_results, level_1_keys, level_2_keys, execution_results, execution_stats_list):
    """
    Check diversity across tree architecture hierarchical responses.
    
    Args:
        question_results: Dictionary containing all response data
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        execution_results: List of execution results
        execution_stats_list: List of execution statistics
    """
    main_rank_print(f"\n{'='*40}")
    main_rank_print("TREE ARCHITECTURE DIVERSITY CHECK")
    main_rank_print(f"{'='*40}")
    main_rank_print(f"🌳 TREE ARCHITECTURE MODE: Checking diversity across hierarchical responses")
    main_rank_print(f"   Level 1 responses: {len(level_1_keys)}")
    main_rank_print(f"   Level 2 responses: {len(level_2_keys)}")
    main_rank_print(f"   Total responses: {len(level_1_keys) + len(level_2_keys)}")
    
    # For tree architecture, we check diversity across the hierarchical structure
    # Each Level 1 response can generate multiple Level 2 responses
    # This creates natural diversity through the tree structure
    
    # Group by original questions to check tree diversity
    question_groups = {}
    for i, level_1_key in enumerate(level_1_keys):
        question = level_1_key.rsplit('<<MySep>>', 1)[0]
        if question not in question_groups:
            question_groups[question] = {'level_1': [], 'level_2': []}
        
        # Add Level 1 response
        question_groups[question]['level_1'].append({
            'key': level_1_key,
            'execution_stats': execution_stats_list[i],
            'execution_result': execution_results[i]
        })
    
    # Add Level 2 responses to their parent questions
    for i, level_2_key in enumerate(level_2_keys):
        response_data = question_results[level_2_key]
        parent_sub_task = response_data.get('parent_sub_task', '')
        
        # Find the original question that generated this sub-task
        original_question = None
        for key, value in question_results.items():
            if isinstance(value, dict) and 'sub_tasks_mapping' in value:
                # Check if this sub-task exists in the mapping
                if parent_sub_task in value['sub_tasks_mapping']:
                    original_question = key.rsplit('<<MySep>>', 1)[0]
                    break
        
        if original_question and original_question in question_groups:
            question_groups[original_question]['level_2'].append({
                'key': level_2_key,
                'execution_stats': execution_stats_list[len(level_1_keys) + i],
                'execution_result': execution_results[len(level_1_keys) + i]
            })
    
    # Check diversity for each question's tree structure
    total_questions = len(question_groups)
    questions_with_tree_diversity = 0
    
    for question, responses in question_groups.items():
        level_1_count = len(responses['level_1'])
        level_2_count = len(responses['level_2'])
        
        main_rank_print(f"\nQuestion: {question[:100]}{'...' if len(question) > 100 else ''}")
        main_rank_print(f"  Level 1 responses: {level_1_count}")
        main_rank_print(f"  Level 2 responses: {level_2_count}")
        main_rank_print(f"  Total responses: {level_1_count + level_2_count}")
        
        # Check if there's diversity in the tree structure
        if level_2_count > 0:
            questions_with_tree_diversity += 1
            main_rank_print(f"  ✓ Tree diversity: Generates {level_2_count} sub-tasks")
            
            # Check code diversity across Level 2 responses
            codes = []
            for resp in responses['level_2']:
                code = resp['execution_stats'].get('code', '')
                codes.append(code)
            
            unique_codes = set(codes)
            if len(unique_codes) > 1:
                main_rank_print(f"  ✓ Code diversity: {len(unique_codes)} unique codes in sub-tasks")
            else:
                main_rank_print(f"  ⚠️  Same code across sub-tasks")
        else:
            main_rank_print(f"  ⚠️  No sub-tasks generated")
    
    # Summary for tree architecture
    main_rank_print(f"\n{'='*40}")
    main_rank_print("TREE ARCHITECTURE DIVERSITY SUMMARY")
    main_rank_print(f"{'='*40}")
    main_rank_print(f"Total questions: {total_questions}")
    main_rank_print(f"Questions with tree diversity: {questions_with_tree_diversity}")
    main_rank_print(f"Tree diversity rate: {questions_with_tree_diversity / total_questions * 100:.1f}%")
    
    if questions_with_tree_diversity > 0:
        main_rank_print(f"✓ Tree architecture provides natural diversity through sub-task generation")
    else:
        main_rank_print(f"⚠️  No tree diversity detected - all questions generate same sub-tasks")


def _process_standard_grpo_responses(question_results, questions_list, mas_code_generation_output):
    """
    Process responses for standard GRPO mode (multiple responses per question).
    
    Args:
        question_results: Dictionary containing all response data
        questions_list: List of original questions
        mas_code_generation_output: DataProto containing generation outputs
        
    Returns:
        tuple: (responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size)
    """
    main_rank_print(f"📊 STANDARD GRPO MODE: Processing multiple responses per question")
    main_rank_print(f"Total questions: {len(questions_list)}")
    
    # Now collect all {question}<<MySep>>{response} pairs in the order they should appear
    # This maintains original batch order while including all responses for GRPO
    question_response_pairs = []

    for question in questions_list:
        # Get all responses for this question from the {question}<<MySep>>{response} keys
        responses_for_question = []
        for key in question_results.keys():
            if key.startswith(f"{question}<<MySep>>") and key.split('<<MySep>>')[-1].isdigit():
                response_idx_str = key.split('<<MySep>>')[-1]
                response_idx = int(response_idx_str)
                responses_for_question.append((response_idx, key))
        
        # Sort by response_idx to maintain order
        responses_for_question.sort(key=lambda x: x[0])
        
        # Add all responses for this question
        for response_idx, key in responses_for_question:
            question_response_pairs.append((question, response_idx, key))
    
    batch_size = len(question_response_pairs)
    main_rank_print(f"Converting {batch_size} question-response pairs to DataProto format for GRPO")
    main_rank_print(f"Average responses per question: {batch_size / len(questions_list):.1f}")
    # main_rank_print(f"Question-response pairs: {[(q, r) for q, r, k in question_response_pairs]}")
    
    # Prepare tensors for DataProto
    responses_tensor = []
    execution_results = []
    execution_stats_list = []
    reward_model_list = []
    
    # Process each question-response pair in order
    for i, (question, response_idx, question_response_key) in enumerate(question_response_pairs):
        # Get the data for this specific response
        response_data = question_results[question_response_key]
        execution_stats = response_data.get('execution_stats', {})
        
        if not execution_stats:
            raise RuntimeError(f"No execution_stats found for {question_response_key}")
        
        # Get the response text from the {question}<<MySep>>{response} key
        response_text = response_data.get('response_text', '')
        if not response_text:
            main_rank_print(f"No response_text found for {question_response_key}; response_data: {response_data}; question_results: {question_results}")
        
        # Get the real response tensor from mas_code_generation_output
        # Use the response_idx stored in question_results to map to the correct response
        # This handles cases where question_response_pairs order differs from mas_code_generation_output order
        response_idx_in_generation = response_data.get('response_idx', 0)
        
        if response_idx_in_generation < len(mas_code_generation_output.batch['responses']):
            responses_tensor.append(mas_code_generation_output.batch['responses'][response_idx_in_generation])
        else:
            # This should never happen - raise error instead of fallback
            error_msg = f"Response index {response_idx_in_generation} exceeds available responses ({len(mas_code_generation_output.batch['responses'])}; question_response_pairs: {question_response_pairs})"
            raise RuntimeError(error_msg)

        # Create execution result for this specific response
        if execution_stats.get('total_executions', 1) == 1:
            # Single execution: populate real execution_result from execution_stats
            if execution_stats.get('execution_results'):
                # Get the single execution result from execution_stats
                single_execution_results = execution_stats.get('execution_results', [])
                if single_execution_results and len(single_execution_results) > 0:
                    # Extract the single execution result (result, success, error)
                    result, success, error = single_execution_results[0]
                    execution_result = {
                        'code': execution_stats.get('code', ''),
                        'result': result,
                        'success': success,
                        'error': error,
                        'question': question,
                        'response_idx': response_idx,
                        'ground_truth': response_data.get('ground_truth', ''),
                        'original_index': response_data.get('original_index', i),
                        'is_validation': response_data.get('is_validation', False)
                    }
                else:
                    raise RuntimeError(f"No execution results found in execution_stats for {question_response_key}")
            else:
                raise RuntimeError(f"No execution results found in execution_stats for {question_response_key}")
        else:
            raise RuntimeError(f"Not implemented: Multiple executions: use placeholder values in execution_result")

        
        execution_results.append(execution_result)
        
        # Process execution_stats for reward function
        if execution_stats:
            # Convert execution_results to the format expected by reward function
            raw_results = execution_stats.get('execution_results', [])
            # The reward function expects (result, success, error) tuples
            execution_stats['execution_results'] = raw_results
        execution_stats_list.append(execution_stats)
        
        # Create reward_model entry for this specific response
        ground_truth = response_data.get('ground_truth', '')
        if not ground_truth:
            raise RuntimeError(f"Ground truth not found for {question_response_key}")
        
        reward_model_list.append({
            'ground_truth': ground_truth
        })
    
    return responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size


def _check_standard_grpo_diversity(question_results, execution_results, execution_stats_list):
    """
    Check diversity across standard GRPO multiple responses per question.
    
    Args:
        question_results: Dictionary containing all response data
        execution_results: List of execution results
        execution_stats_list: List of execution statistics
    """
    main_rank_print(f"\n{'='*40}")
    main_rank_print("STANDARD GRPO DIVERSITY CHECK")
    main_rank_print(f"{'='*40}")
    main_rank_print(f"📊 STANDARD GRPO MODE: Checking diversity across multiple responses per question")
    
    # Create question_response_pairs for GRPO diversity checking
    question_response_pairs = []
    for key in question_results.keys():
        if '<<MySep>>' in key and key.split('<<MySep>>')[-1].isdigit():
            question = key.rsplit('<<MySep>>', 1)[0]
            response_idx_str = key.split('<<MySep>>')[-1]
            response_idx = int(response_idx_str)
            question_response_pairs.append((question, response_idx, key))
    
    # Group by question to check diversity
    question_groups = {}
    for i, (question, response_idx, question_response_key) in enumerate(question_response_pairs):
        if question not in question_groups:
            question_groups[question] = []
        question_groups[question].append({
            'response_idx': response_idx,
            'question_response_key': question_response_key,
            'execution_stats': execution_stats_list[i],
            'execution_result': execution_results[i]
        })
    
    # Check diversity for each question
    total_questions = len(question_groups)
    questions_with_diverse_responses = 0
    questions_with_diverse_codes = 0
    questions_with_diverse_results = 0
    
    for question, responses in question_groups.items():
        if len(responses) <= 1:
            continue  # Skip questions with only one response
            
        main_rank_print(f"\nQuestion: {question[:100]}{'...' if len(question) > 100 else ''}")
        main_rank_print(f"  Number of responses: {len(responses)}")
        
        # Check response diversity (extracted codes)
        codes = []
        for resp in responses:
            code = resp['execution_stats'].get('code', '')
            codes.append(code)
        
        unique_codes = set(codes)
        if len(unique_codes) > 1:
            questions_with_diverse_codes += 1
            main_rank_print(f"  ✓ Diverse codes: {len(unique_codes)} unique codes")
            for i, code in enumerate(codes):
                # Handle None codes safely
                if code is None:
                    code_display = "None"
                else:
                    code_display = f"{code[:50]}{'...' if len(code) > 50 else ''}"
                main_rank_print(f"    Response {i+1}: {code_display}")
        else:
            main_rank_print(f"  ✗ Same code across all responses")
            # Check if any responses are validation responses
            is_validation = any(resp['execution_result'].get('is_validation', False) for resp in responses)
            
            # Handle None codes safely in error message
            if codes and codes[0] is not None:
                code_display = f"{codes[0][:100]}{'...' if len(codes[0]) > 100 else ''}"
            else:
                code_display = "None"
                
            error_msg = f"GRPO ERROR: Same code across all responses for question '{question[:100]}{'...' if len(question) > 100 else ''}'. Code: {code_display}"
            
            if is_validation:
                # Give warning instead of error for validation mode
                main_rank_print(f"  ⚠️  WARNING (validation mode): {error_msg}")
            else:
                # Raise error if training - GRPO should generate diverse responses
                main_rank_print(f"  ⚠️  WARNING (training mode): {error_msg}")
                # it is possible, e.g. prompt_type=harmony, where there is "" code for direct answer
                # raise RuntimeError(error_msg)
        
        # Check execution result diversity
        results = []
        for resp in responses:
            if resp['execution_stats'].get('total_executions', 1) == 1:
                # Single execution: get the result
                execution_results_list = resp['execution_stats'].get('execution_results', [])
                if execution_results_list:
                    result, success, error = execution_results_list[0]
                    results.append(f"{result} (success={success})")
                else:
                    results.append("No execution result")
            else:
                # Multiple executions: count unique results
                execution_results_list = resp['execution_stats'].get('execution_results', [])
                unique_results = set()
                for result, success, error in execution_results_list:
                    unique_results.add(f"{result} (success={success})")
                results.append(f"Multiple: {len(unique_results)} unique results")
        
        unique_results = set(results)
        if len(unique_results) > 1:
            questions_with_diverse_results += 1
            main_rank_print(f"  ✓ Diverse execution results: {len(unique_results)} unique results")
            for i, result in enumerate(results):
                main_rank_print(f"    Response {i+1}: {result}")
        else:
            main_rank_print(f"  ✗ Same execution results across all responses")
        
        # Check if any diversity exists
        if len(unique_codes) > 1 or len(unique_results) > 1:
            questions_with_diverse_responses += 1
    
    # Summary statistics for standard GRPO
    main_rank_print(f"\n{'='*40}")
    main_rank_print("STANDARD GRPO DIVERSITY SUMMARY")
    main_rank_print(f"{'='*40}")
    main_rank_print(f"Total questions: {total_questions}")
    main_rank_print(f"Questions with multiple responses: {sum(1 for q, r in question_groups.items() if len(r) > 1)}")
    main_rank_print(f"Questions with diverse codes: {questions_with_diverse_codes}")
    main_rank_print(f"Questions with diverse execution results: {questions_with_diverse_results}")
    main_rank_print(f"Questions with any diversity: {questions_with_diverse_responses}")
    
    if total_questions > 0:
        diversity_rate = questions_with_diverse_responses / total_questions * 100
        main_rank_print(f"Diversity rate: {diversity_rate:.1f}%")
        
        if diversity_rate < 50:
            main_rank_print(f"⚠️  WARNING: Low diversity rate! GRPO may not be working effectively.")
        else:
            main_rank_print(f"✓ Good diversity rate for GRPO training.")


def _process_simple_mode_responses(question_results, questions_list, mas_code_generation_output):
    """
    Process responses for simple mode (only Level 1 responses for original questions).
    
    Args:
        question_results: Dictionary containing all response data
        questions_list: List of original questions
        mas_code_generation_output: DataProto containing generation outputs
        
    Returns:
        tuple: (responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size)
    """
    # Simple mode: only Level 1 responses for original questions
    batch_size = len(questions_list)
    main_rank_print(f"📊 Simple mode: Batch size based on original questions: {batch_size}")
    
    # Prepare tensors for DataProto
    responses_tensor = []
    execution_results = []
    execution_stats_list = []
    reward_model_list = []
    
    # Process each question (single response per question)
    for i, question in enumerate(questions_list):
        # Simple mode: only Level 1 responses
        question_response_key = f"{question}<<MySep>>0"
        response_level = 1
        
        if question_response_key not in question_results:
            raise RuntimeError(f"Question response key '{question_response_key}' not found in question_results. Available keys: {list(question_results.keys())}")
        
        question_data = question_results[question_response_key]
        
        # Get the response text from the {question}<<MySep>>{response} key
        response_text = question_data.get('response_text', '')
        if not response_text: # '' will be judge as False
            main_rank_print(f"No response_text found for {question_response_key}; question_data: {question_data}; question_results: {question_results}")
        
        # Use the single response tensor from mas_code_generation_output
        if i < len(mas_code_generation_output.batch['responses']):
            responses_tensor.append(mas_code_generation_output.batch['responses'][i])
        else:
            error_msg = f"Index {i} exceeds available responses ({len(mas_code_generation_output.batch['responses'])})"
            raise RuntimeError(error_msg)
        
        # Get execution stats for this question
        execution_stats = question_data.get('execution_stats', {})
        if not execution_stats:
            raise RuntimeError(f"No execution_stats found for {question_response_key}")
        
        # Create execution result for this question
        if execution_stats.get('total_executions', 1) == 1:
            # Single execution: populate real execution_result from execution_stats
            if execution_stats.get('execution_results'):
                # Get the single execution result from execution_stats
                single_execution_results = execution_stats.get('execution_results', [])
                if single_execution_results and len(single_execution_results) > 0:
                    # Extract the single execution result (result, success, error)
                    result, success, error = single_execution_results[0]
                    execution_result = {
                        'code': execution_stats.get('code', ''),
                        'result': result,
                        'success': success,
                        'error': error,
                        'question': question,
                        'response_idx': 0,  # Single response
                        'ground_truth': question_data.get('ground_truth', ''),
                        'original_index': question_data.get('original_index', i),
                        'is_validation': True,
                        # Add tree structure information for hierarchical rewards
                        'level': response_level,  # Use detected level instead of default 1
                        'parent_response_idx': question_data.get('parent_response_idx', None),
                        'parent_sub_task': question_data.get('parent_sub_task', None)
                    }
                else:
                    raise RuntimeError(f"No execution results found in execution_stats for {question_response_key}")
            else:
                raise RuntimeError(f"No execution results found in execution_stats for {question_response_key}")
        else:
            raise RuntimeError(f"Not implemented: Multiple executions: use placeholder values in execution_result")

        
        execution_results.append(execution_result)
        
        # Process execution_stats for reward function
        if execution_stats:
            # Convert execution_results to the format expected by reward function
            raw_results = execution_stats.get('execution_results', [])
            # The reward function expects (result, success, error) tuples
            execution_stats['execution_results'] = raw_results
        execution_stats_list.append(execution_stats)
        
        # Create reward_model entry for this question
        ground_truth = question_data.get('ground_truth', '')
        if not ground_truth:
            raise RuntimeError(f"Ground truth not found for {question_response_key}")
        
        reward_model_list.append({
            'ground_truth': ground_truth
        })
    
    return responses_tensor, execution_results, execution_stats_list, reward_model_list, batch_size
