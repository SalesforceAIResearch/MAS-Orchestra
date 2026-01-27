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
Extraction and generation utilities for MAS-R1 Trainer.
"""

from typing import Dict, Tuple, Any, List
from verl.protocol import DataProto
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code
from mas_r1_reasoner.rewards.utils.harmony_parser import extract_harmony_code_from_response
from mas_r1_reasoner.agents.shared_vars import get_global
from mas_r1_reasoner.trainer.utils.helper import get_safe_length
import torch
from mas_r1_reasoner.trainer.utils.rollout import rollout_generation

# Import necessary functions and classes for tree architecture validation
from mas_r1_reasoner.rewards.utils.extraction_tree import (
    SubAgentBatchManager,
    create_sub_agent_batch_from_sub_task_and_prompt
)

# Import the existing extract_codes function from extraction_stretegy.py
from mas_r1_reasoner.rewards.utils.extraction_strategy import (
    extract_codes,
)

def extract_questions_and_ground_truth(trainer_instance, batch: DataProto) -> Dict:
    """
    Extract questions and ground truth from batch using dictionary-based approach.
    This is shared between training and validation.
    """
    question_results = {}
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("EXTRACTING QUESTIONS AND GROUND TRUTH")
    main_rank_print(f"{'='*60}")
    
    for i in range(len(batch)):
        question = trainer_instance.processor.extract_math_question(batch, i)
        # Try to get ground truth from reward_model field if available
        try:
            ground_truth = batch.non_tensor_batch['reward_model'][i]['ground_truth']
        except Exception:
            ground_truth = None
        
        # Use question as key to maintain order and eliminate alignment issues
        # for duplicate questions, use the last one (we will have duplicated questions)
        # addressed in helper.py
        question_results[question] = {
            'original_index': i,
            'question': question,
            'ground_truth': ground_truth,
            'input_ids': batch.batch['input_ids'][i],
            'attention_mask': batch.batch['attention_mask'][i],
            'position_ids': batch.batch['position_ids'][i] if 'position_ids' in batch.batch else None,
        }
        # main_rank_print(f"Question {i}: {question[:100]}{'...' if len(question) > 100 else ''}")
        # main_rank_print(f"Ground truth {i}: {ground_truth}")

    main_rank_print(f"Question {i}: {question}")
    main_rank_print(f"Ground truth {i}: {ground_truth}")

    return question_results


def generate_and_extract_codes(trainer_instance, question_results: Dict, mas_code_generation_output: DataProto, is_validation: bool = False) -> Tuple[DataProto, Dict]:
    """
    Generate codes using Stage 1 and extract them into question_results dictionary.
    This is shared between training and validation.
    """
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("GENERATING AND EXTRACTING CODES")
    main_rank_print(f"{'='*60}")
    
    
    main_rank_print(f"MAS Code Generation output shape: {mas_code_generation_output.batch['responses'].shape}")
    
    # Extract codes and store in dictionary
    main_rank_print(f"\n{'='*60}")
    main_rank_print("EXTRACTING CODES AND STORING IN DICTIONARY")
    main_rank_print(f"{'='*60}")


    mas_code_generation_length = get_safe_length(mas_code_generation_output, "mas_code_generation_output")
    original_questions_list = list(question_results.keys())
    
    # Determine expected generation length based on mode
    if is_validation:
        # Validation: val_kwargs.n responses per original question
        val_n = trainer_instance.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1)
        expected_length = len(original_questions_list) * val_n
        main_rank_print(f"VALIDATION MODE: {val_n} responses per original question (total: {expected_length})")
    else:
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
    
    # Process each question and its responses
    if is_validation:
        # Validation: loop over batch indices directly (repeated questions)
        for generation_idx in range(mas_code_generation_length):
            response_text = trainer_instance.tokenizer.decode(mas_code_generation_output.batch['responses'][generation_idx], skip_special_tokens=True)
            
            # Extract question directly from mas_code_generation_output
            question = trainer_instance.processor.extract_math_question(mas_code_generation_output, generation_idx)
            
            # Try to extract code
            try:
                # Use harmony_parser for harmony problems, otherwise use regular extract_code_from_response
                if 'harmony' in get_global('global_problem_type'):
                    code, name, thought = extract_harmony_code_from_response(
                        response_text,
                        validate_python_code,
                        trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
                    )
                else:
                    code, name, thought = extract_code_from_response(
                        response_text,
                        validate_python_code,
                        trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
                    )
                
                # Store with correct keys for validation (single response)
                extracted_code_data = {
                    'mas_code_generation_response': response_text,
                    'extracted_code': code,
                    'extracted_name': name,
                    'extracted_thought': thought,
                    'code_extraction_success': code is not None,
                }
                
                # Store using question_response_key
                question_response_key = f"{question}<<MySep>>{generation_idx}"
                question_results[question_response_key] = {
                    'response_text': response_text,
                    'extracted_code_data': extracted_code_data,
                    'question': question,  # Use the extracted question
                    'response_idx': generation_idx,
                    'ground_truth': question_results[question].get('ground_truth') if question in question_results else None,
                    'original_index': question_results[question].get('original_index', generation_idx) if question in question_results else generation_idx,
                    'is_validation': True
                }
                
                if code is not None:
                    main_rank_print(f"✓ Response {generation_idx+1}: Valid code {code[:100]}{'...' if len(code) > 100 else ''} extracted for question: {question[:100]}{'...' if len(question) > 100 else ''}")
                else:
                    main_rank_print(f"✗ Response {generation_idx+1}: No valid code {code[:100]}{'...' if len(code) > 100 else ''} extracted for question: {question[:100]}{'...' if len(question) > 100 else ''}")
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
                
                # Store using question_response_key
                question_response_key = f"{question}<<MySep>>{generation_idx}"
                question_results[question_response_key] = {
                    'response_text': response_text,
                    'extracted_code_data': failed_extracted_code_data,
                    'question': question,  # Use the extracted question
                    'response_idx': generation_idx,
                    'ground_truth': question_results[question].get('ground_truth') if question in question_results else None,
                    'original_index': question_results[question].get('original_index', generation_idx) if question in question_results else generation_idx,
                    'is_validation': True
                }
    else:
        # Training: process rollout.n responses per question        
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
                # Use harmony_parser for harmony problems, otherwise use regular extract_code_from_response
                if 'harmony' in get_global('global_problem_type'):
                    code, name, thought = extract_harmony_code_from_response(
                        response_text,
                        validate_python_code,
                        trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
                    )
                else:
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
                    'is_validation': False
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
                    'is_validation': False
                }
    
    return mas_code_generation_output, question_results


def _process_level_2_validation(
    trainer_instance, 
    question_results: Dict, 
    mas_code_generation_output: DataProto,
    extracted_items: List[str],
    item_type: str,  # 'sub_tasks' or 'codes'
    mapping_key: str,  # 'sub_tasks_mapping' or 'codes_mapping'
    architecture_name: str  # 'Tree' or 'Sequential'
) -> Tuple[DataProto, Dict]:
    """
    Shared function for processing Level 2 validation results.
    
    Args:
        trainer_instance: The trainer instance
        question_results: Dictionary containing question information
        mas_code_generation_output: Output from Level 1 generation
        extracted_items: List of extracted items (sub-tasks or codes)
        item_type: Type of extracted items for logging
        mapping_key: Key used for mapping in question_results
        architecture_name: Name of the architecture for error messages
    
    Returns:
        Tuple of (mas_code_generation_output, question_results)
    """
    
    if not extracted_items:
        main_rank_print(f"📝 No {item_type} found for Level 2 validation generation")
        # do not replace, use the original (level 1)
        return mas_code_generation_output, question_results
    
    try:
        # Initialize batch manager for Level 2 generation
        from mas_r1_reasoner.rewards.utils.extraction_tree import SubAgentBatchManager
        manager = SubAgentBatchManager(trainer_instance)
        # For validation, generate only 1 response per item
        manager.new_sub_agents_per_sub_task = 1
        
        # Track used question_response_keys to prevent duplicates
        used_keys = set()
        
        main_rank_print(f"🔄 Generating 1 response per {item_type[:-1]} for validation")
        
        # Generate all responses in a single batch operation
        sub_agent_results = manager.generate_all_sub_agents_batch(extracted_items, question_results, is_validation=True)
        
        successful_count = len([r for r, _, _ in sub_agent_results if r is not None])
        main_rank_print(f"🚀 Level 2 validation generation completed: {successful_count}/{len(extracted_items)} responses successful")
        
        # Process Level 2 validation results
        for sub_agent_output, item, new_sub_agent_id in sub_agent_results:
            if sub_agent_output is not None:
                # Extract codes from sub-agent output
                sub_agent_length = get_safe_length(sub_agent_output, "sub_agent_output")
                
                # Process each response in the sub-agent output
                for response_idx in range(sub_agent_length):
                    try:
                        # Decode the response text
                        response_text = trainer_instance.tokenizer.decode(
                            sub_agent_output.batch['responses'][response_idx], 
                            skip_special_tokens=True
                        )
                        
                        # Extract code from response
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
                        
                        # Find the original question and response that generated this item
                        original_question = None
                        original_question_idx = None
                        generation_idx = None

                        # Search through question_results to find an UNUSED original question and generation_idx for this item
                        # This prevents duplicate question_response_key values
                        for key, value in question_results.items():
                            if isinstance(value, dict) and mapping_key in value:
                                # Check if this item exists in the mapping
                                if item in value[mapping_key]:
                                    # Get ALL mapping data for this item (there might be multiple entries)
                                    for mapping_data in value[mapping_key][item]:
                                        candidate_question = key
                                        candidate_generation_idx = mapping_data.get('response_idx')
                                        candidate_key = f"{candidate_question}<<MySep>>{candidate_generation_idx}"
                                        
                                        # Check if this combination is already used in this Level 2 validation
                                        if candidate_key not in used_keys:
                                            original_question = candidate_question
                                            original_question_idx = value.get('original_index')
                                            generation_idx = candidate_generation_idx
                                            used_keys.add(candidate_key)
                                            main_rank_print(f"📝 Item '{item[:50]}...' assigned to unused key '{candidate_key}'")
                                            break
                                    
                                    if original_question:
                                        break
                        
                        if original_question is None:
                            error_msg = f"Could not find original question for {item_type[:-1]} '{item}' in validation. This indicates a fundamental problem with the {architecture_name.lower()} structure mapping."
                            main_rank_print(f"❌ {error_msg}")
                            # import time
                            # time.sleep(1000000)
                            raise RuntimeError(error_msg)
                        
                        # Use the EXACT SAME key structure as generate_and_extract_codes
                        # SO that we are replacing the original level 1's entry
                        # only those with '_' will be executed
                        question_response_key = f"{original_question}<<MySep>>{generation_idx}"
                        
                        # Store Level 2 validation results in question_results
                        
                        # Update existing entry while preserving mapping keys
                        question_results[question_response_key].update({
                            'response_text': response_text,
                            'extracted_code_data': extracted_code_data,
                            'question': original_question,  # Use the actual original question
                            'response_idx': generation_idx,
                            'ground_truth': question_results[original_question].get('ground_truth'),
                            'original_index': original_question_idx,
                            'is_validation': True
                        })
                        
                        if code is not None:
                            main_rank_print(f"✓ Level 2 validation key '{question_response_key}' '{new_sub_agent_id[:50]}' Response {response_idx+1}: Valid code {code[:100]}{'...' if len(code) > 100 else ''} extracted")
                        else:
                            main_rank_print(f"✗ Level 2 validation '{question_response_key}' '{new_sub_agent_id[:50]}' Response {response_idx+1}: No valid code {code[:100]}{'...' if len(code) > 100 else ''} extracted")
                            
                    except Exception as e:
                        main_rank_print(f"✗ Level 2 validation '{new_sub_agent_id}' Response {response_idx+1}: Code extraction failed: {e}")
                        
                        # Store failed extraction
                        failed_extracted_code_data = {
                            'mas_code_generation_response': response_text if 'response_text' in locals() else "Unknown",
                            'extracted_code': "",
                            'extracted_name': "Unknown",
                            'extracted_thought': "",
                            'code_extraction_success': False,
                            'extraction_error': str(e)
                        }
                        
                        # Use the EXACT SAME key structure as generate_and_extract_codes
                        question_response_key = f"{original_question}<<MySep>>{generation_idx}"
                        
                        # Store failed Level 2 validation result                        
                        # Update existing entry while preserving mapping keys
                        question_results[question_response_key].update({
                            'response_text': response_text if 'response_text' in locals() else "Unknown",
                            'extracted_code_data': failed_extracted_code_data,
                            'question': original_question if 'original_question' in locals() else "Unknown",
                            'response_idx': generation_idx,
                            'ground_truth': question_results[original_question].get('ground_truth') if 'original_question' in locals() else "",
                            'original_index': original_question_idx if 'original_question_idx' in locals() else 0,
                            'is_validation': True
                        })
            else:
                main_rank_print(f"❌ Skipping failed Level 2 validation generation for '{new_sub_agent_id}'")
        
        main_rank_print(f"✅ Level 2 validation generation and code extraction completed")
        # main_rank_print(f"DEBUG: question_results: {question_results}")
        
        # Create mas_code_generation_output with Level 2 responses
        main_rank_print(f"\n{'='*60}")
        main_rank_print("CREATING DATAPROTO WITH LEVEL 2 VALIDATION RESPONSES ONLY")
        main_rank_print(f"{'='*60}")
        
        # Create DataProto with ONLY Level 2 responses
        level_2_batch = {}
        
        # Get the first sub_agent_result to understand the tensor structure
        first_sub_agent_result = sub_agent_results[0][0]  # (DataProto, item, new_sub_agent_id)
        
        if first_sub_agent_result is None:
            error_msg = f"First sub_agent_result is None. This indicates a fundamental failure in Level 2 validation generation."
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Extract tensor fields from Level 2 responses only
        for key in first_sub_agent_result.batch.keys():
            if key not in mas_code_generation_output.batch:
                continue  # Skip fields that don't exist in Level 1
                
            # Collect all Level 2 tensors for this field
            level_2_tensor_list = []
            
            for sub_agent_output, item, new_sub_agent_id in sub_agent_results:
                if sub_agent_output is None:
                    error_msg = f"Sub_agent_output is None for {item_type[:-1]} '{item}'. This indicates a fundamental failure in Level 2 validation generation."
                    main_rank_print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
                
                if key in sub_agent_output.batch and sub_agent_output.batch[key] is not None:
                    level_2_tensor_list.append(sub_agent_output.batch[key])
                else:
                    error_msg = f"Missing tensor field '{key}' in sub_agent_output for {item_type[:-1]} '{item}'. This indicates a fundamental failure in Level 2 validation tensor structure."
                    main_rank_print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
            
            # Concatenate all Level 2 tensors for this field
            if level_2_tensor_list:
                combined_level_2_tensor = torch.cat(level_2_tensor_list, dim=0)
                level_2_batch[key] = combined_level_2_tensor
                main_rank_print(f"✓ Created Level 2 tensor '{key}': {combined_level_2_tensor.shape}")
            else:
                error_msg = f"No valid Level 2 tensors found for field '{key}'. This indicates a fundamental failure in Level 2 validation tensor extraction."
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
        
        # Create the Level 2 DataProto
        level_2_dataproto = DataProto.from_single_dict(level_2_batch)
        
        # Copy meta_info from original (for consistency)
        if hasattr(mas_code_generation_output, 'meta_info'):
            level_2_dataproto.meta_info = mas_code_generation_output.meta_info.copy()
        
        # Validate the Level 2 structure
        total_level_2_responses = len(level_2_dataproto)
        
        main_rank_print(f"🎯 Level 2 DataProto created successfully:")
        main_rank_print(f"   Level 2 responses: {total_level_2_responses}")
        main_rank_print(f"   Tensor fields: {list(level_2_batch.keys())}")
        
        # Final validation: ensure all required fields are present and have correct sizes
        main_rank_print(f"\n🔍 Final validation of Level 2 DataProto...")
        
        # Required tensor fields (these are the core fields the trainer expects)
        required_tensor_fields = ['input_ids', 'attention_mask', 'responses']
        for field in required_tensor_fields:
            if field not in level_2_dataproto.batch:
                error_msg = f"Missing required tensor field '{field}' in Level 2 DataProto"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            field_size = level_2_dataproto.batch[field].size(0)
            if field_size != total_level_2_responses:
                error_msg = f"Tensor field '{field}' size mismatch: expected {total_level_2_responses}, got {field_size}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            main_rank_print(f"   ✓ Tensor field '{field}': {level_2_dataproto.batch[field].shape}")
        
        main_rank_print(f"✅ Successfully created Level 2 DataProto with {total_level_2_responses} responses")
        main_rank_print(f"{'='*60}")
        
        # Return the Level 2 DataProto instead of the original
        return level_2_dataproto, question_results
        
    except Exception as e:
        error_msg = f"Level 2 validation generation failed: {e}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


def generate_and_extract_codes_with_sequential_validation(trainer_instance, reward_manager, question_results: Dict, mas_code_generation_output: DataProto) -> Tuple[DataProto, Dict]:
    """
    Generate codes using Stage 1 and extract them into question_results dictionary.
    For validation with sequential architecture enabled, extract codes from Level 1 and generate Level 2 responses.
    """
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("GENERATING AND EXTRACTING CODES WITH SEQUENTIAL ARCHITECTURE VALIDATION")
    main_rank_print(f"{'='*60}")
    
    # First, call the original function to handle Level 1 generation and extraction
    mas_code_generation_output, question_results = generate_and_extract_codes(
        trainer_instance, question_results, mas_code_generation_output, is_validation=True
    )
    
    # Generate Level 2 responses using sequential architecture (extract codes from Level 1)
    main_rank_print(f"\n{'='*60}")
    main_rank_print("SEQUENTIAL ARCHITECTURE VALIDATION: GENERATING LEVEL 2 RESPONSES FROM EXTRACTED CODES")
    main_rank_print(f"{'='*60}")
    
    # Extract codes from Level 1 responses
    from mas_r1_reasoner.rewards.utils.extraction_strategy import extract_codes
    codes = extract_codes(trainer_instance, mas_code_generation_output, question_results, is_validation=True)
    
    # Use the shared function for Level 2 validation processing
    mas_code_generation_output, question_results = _process_level_2_validation(
        trainer_instance,
        question_results,
        mas_code_generation_output,
        codes,
        'codes',
        'sub_tasks_mapping',  # The existing function uses this key
        'Sequential'
    )
    
    return mas_code_generation_output, question_results





def generate_and_extract_codes_with_tree_validation(trainer_instance, question_results: Dict, mas_code_generation_output: DataProto) -> Tuple[DataProto, Dict]:
    """
    Generate codes using Stage 1 and extract them into question_results dictionary.
    For validation with tree architecture enabled, also generate Level 2 sub-agents.
    """
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("GENERATING AND EXTRACTING CODES WITH TREE VALIDATION")
    main_rank_print(f"{'='*60}")
    
    # First, call the original function to handle Level 1 generation and extraction
    mas_code_generation_output, question_results = generate_and_extract_codes(
        trainer_instance, question_results, mas_code_generation_output, is_validation=True
    )
    
    # Generate Level 2 sub-agents for validation
    main_rank_print(f"\n{'='*60}")
    main_rank_print("TREE ARCHITECTURE VALIDATION: GENERATING LEVEL 2 SUB-AGENTS")
    main_rank_print(f"{'='*60}")
    
    # Extract sub-tasks from Level 1 responses using the existing function
    # sub_tasks, _ = extract_sub_tasks_and_sub_agents(
    codes =  extract_codes(
        trainer_instance, 
        mas_code_generation_output, 
        question_results, 
        is_validation=True
    )
    # Use the shared function for Level 2 validation processing
    mas_code_generation_output, question_results = _process_level_2_validation(
        trainer_instance,
        question_results,
        mas_code_generation_output,
        codes,
        'codes',
        'sub_tasks_mapping',
        'Tree'
    )

    return mas_code_generation_output, question_results