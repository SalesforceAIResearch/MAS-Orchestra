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
from mas_r1_reasoner.agents.agent_system import AgentSystem as SequentialAgentSystem
from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
from mas_r1_reasoner.agents.logging_utils.stdout import PrettyPrinter as pp
import re
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code

def save_generated_code(save_dir, global_steps, current_step_generated_code, accumulated_generated_code, 
                       save_generated_code, save_code_summary, max_accumulated_steps):
    """Save generated code from Stage 1 to checkpoint folder for analysis"""
    # Skip if code saving is disabled
    if not save_generated_code:
        return
    
    # Create code directory
    code_dir = save_dir / 'generated_code'
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current step's generated code if available
    if current_step_generated_code:
        step_code_file = code_dir / f'step_{global_steps}_generated_code.json'
        
        with open(step_code_file, 'w') as f:
            # Convert any potential numpy types to native Python types for JSON serialization
            step_data = {
                'step': global_steps,
                'timestamp': str(datetime.now()),
                'generated_code': current_step_generated_code
            }
            step_data_converted = convert_numpy_types(step_data)
            json.dump(step_data_converted, f, indent=2)
        
        main_rank_print(f"✓ Saved generated code for step {global_steps} to {step_code_file}")
    
    # Save accumulated code samples (keep last N steps)
    if accumulated_generated_code:
        accumulated_file = code_dir / 'accumulated_generated_code.json'
        
        # Keep only last 10 steps to avoid file getting too large
        recent_code = accumulated_generated_code[-10:] if len(accumulated_generated_code) > 10 else accumulated_generated_code
        
        with open(accumulated_file, 'w') as f:
            # Convert any potential numpy types to native Python types for JSON serialization
            accumulated_data = {
                'total_steps': len(accumulated_generated_code),
                'recent_steps': len(recent_code),
                'generated_code_samples': recent_code
            }
            accumulated_data_converted = convert_numpy_types(accumulated_data)
            json.dump(accumulated_data_converted, f, indent=2)
        
        main_rank_print(f"✓ Saved accumulated generated code samples to {accumulated_file}")
    
    # Save summary statistics if enabled
    if save_code_summary and accumulated_generated_code:
        summary_file = code_dir / 'code_generation_summary.json'
        
        # Calculate statistics
        total_samples = 0
        successful_extractions = 0
        valid_code_samples = 0
        total_steps = len(accumulated_generated_code)
        
        # Track name and thought statistics
        name_statistics = {}
        thought_lengths = []
        
        for step_data in accumulated_generated_code:
            for code_sample in step_data['generated_code']:
                total_samples += 1
                if code_sample.get('extraction_success', False):
                    successful_extractions += 1
                if code_sample.get('code_is_valid', False):
                    valid_code_samples += 1
                
                # Track name statistics
                name = code_sample.get('extracted_name', 'Unknown')
                name_statistics[name] = name_statistics.get(name, 0) + 1
                
                # Track thought length statistics
                thought = code_sample.get('extracted_thought', '')
                if thought is None:
                    thought = ''  # Convert None to empty string
                thought_lengths.append(len(thought))
        
        summary_stats = {
            'total_steps': int(total_steps),  # Convert to native Python int
            'total_samples': int(total_samples),  # Convert to native Python int
            'successful_extractions': int(successful_extractions),  # Convert to native Python int
            'failed_extractions': int(total_samples - successful_extractions),  # Convert to native Python int
            'valid_code_samples': int(valid_code_samples),  # Convert to native Python int
            'invalid_code_samples': int(total_samples - valid_code_samples),  # Convert to native Python int
            'extraction_success_rate': float(successful_extractions / total_samples if total_samples > 0 else 0.0),  # Convert to native Python float
            'code_validity_rate': float(valid_code_samples / total_samples if total_samples > 0 else 0.0),  # Convert to native Python float
            'name_statistics': name_statistics,
            'thought_length_stats': {
                'avg_length': float(np.mean(thought_lengths) if thought_lengths else 0.0),  # Convert to native Python float
                'min_length': int(np.min(thought_lengths) if thought_lengths else 0.0),  # Convert to native Python int
                'max_length': int(np.max(thought_lengths) if thought_lengths else 0.0),  # Convert to native Python int
                'total_thoughts': int(len(thought_lengths))  # Convert to native Python int
            },
            'last_updated': str(datetime.now())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        main_rank_print(f"✓ Saved code generation summary to {summary_file}")


def save_responses_and_ground_truth(save_dir, global_steps, current_step_responses, accumulated_responses, 
                                   save_responses_and_ground_truth):
    """Save extracted responses and ground truth responses to checkpoint folder for analysis"""
    # Skip if response saving is disabled
    if not save_responses_and_ground_truth:
        return
    
    # Create responses directory
    responses_dir = save_dir / 'responses_and_ground_truth'
    responses_dir.mkdir(parents=True, exist_ok=True)
    
    # Save current step's responses and ground truth if available
    if current_step_responses:
        step_responses_file = responses_dir / f'step_{global_steps}_responses.json'
        
        with open(step_responses_file, 'w') as f:
            # Convert any potential numpy types to native Python types for JSON serialization
            step_data = {
                'step': global_steps,
                'timestamp': str(datetime.now()),
                'responses_and_ground_truth': current_step_responses
            }
            step_data_converted = convert_numpy_types(step_data)
            json.dump(step_data_converted, f, indent=2)
        
        main_rank_print(f"✓ Saved responses and ground truth for step {global_steps} to {step_responses_file}")
    
    # Save accumulated responses and ground truth (keep last N steps)
    if accumulated_responses:
        accumulated_file = responses_dir / 'accumulated_responses.json'
        
        # Keep only last 10 steps to avoid file getting too large
        recent_responses = accumulated_responses[-10:] if len(accumulated_responses) > 10 else accumulated_responses
        
        with open(accumulated_file, 'w') as f:
            # Convert any potential numpy types to native Python types for JSON serialization
            accumulated_data = {
                'total_steps': len(accumulated_responses),
                'recent_steps': len(recent_responses),
                'responses_and_ground_truth_samples': recent_responses
            }
            accumulated_data_converted = convert_numpy_types(accumulated_data)
            json.dump(accumulated_data_converted, f, indent=2)
        
        main_rank_print(f"✓ Saved accumulated responses and ground truth to {accumulated_file}")
    
    # Save summary statistics for responses and ground truth
    if accumulated_responses:
        summary_file = responses_dir / 'responses_summary.json'
        
        # Calculate statistics
        total_samples = 0
        correct_predictions = 0
        successful_executions = 0
        total_steps = len(accumulated_responses)
        
        # Track answer statistics
        answer_statistics = {}
        
        for step_data in accumulated_responses:
            for response_sample in step_data['responses_and_ground_truth']:
                total_samples += 1
                if response_sample.get('answer_correct', False):
                    correct_predictions += 1
                if response_sample.get('execution_success', False):
                    successful_executions += 1
                
                # Track ground truth statistics
                ground_truth = response_sample.get('ground_truth', 'Unknown')
                answer_statistics[ground_truth] = answer_statistics.get(ground_truth, 0) + 1
        
        summary_stats = {
            'total_steps': int(total_steps),  # Convert to native Python int
            'total_samples': int(total_samples),  # Convert to native Python int
            'correct_predictions': int(correct_predictions),  # Convert to native Python int
            'incorrect_predictions': int(total_samples - correct_predictions),  # Convert to native Python int
            'successful_executions': int(successful_executions),  # Convert to native Python int
            'failed_executions': int(total_samples - successful_executions),  # Convert to native Python int
            'accuracy_rate': float(correct_predictions / total_samples if total_samples > 0 else 0.0),  # Convert to native Python float
            'execution_success_rate': float(successful_executions / total_samples if total_samples > 0 else 0.0),  # Convert to native Python float
            'answer_statistics': answer_statistics,
            'last_updated': str(datetime.now())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        main_rank_print(f"✓ Saved responses summary to {summary_file}")


def collect_generated_code_from_dict(question_results, mas_code_generation_output, global_steps, save_generated_code, 
                                   max_accumulated_steps, logger=None):
    """Collect generated code from Stage 1 for saving, using dictionary-based approach."""
    if not save_generated_code:
        return None, None

    generated_code = []
    
    # Collect all {question}_{response} keys
    question_response_keys = []
    for key in question_results.keys():
        if '_' in key and key.split('_')[-1].isdigit():
            question_response_keys.append(key)
    
    # Sort by question and response index for consistent ordering
    question_response_keys.sort(key=lambda x: (x.rsplit('_', 1)[0], int(x.rsplit('_', 1)[1])))
    
    # Group by question to combine multiple responses
    question_groups = {}
    for question_response_key in question_response_keys:
        question = question_response_key.rsplit('_', 1)[0]
        response_idx = int(question_response_key.rsplit('_', 1)[1])
        
        if question not in question_groups:
            question_groups[question] = []
        question_groups[question].append((response_idx, question_response_key))
    
    # Sort responses within each question group
    for question in question_groups:
        question_groups[question].sort(key=lambda x: x[0])
    
    for i, (question, responses) in enumerate(question_groups.items()):
        if i >= len(mas_code_generation_output.batch['responses']):
            break
            
        # Combine all responses and codes in 'rollout_1:...rollout_2:...' format
        combined_response_text = ""
        combined_extracted_code = ""
        combined_extracted_name = ""
        combined_extracted_thought = ""
        
        response_parts = []
        code_parts = []
        name_parts = []
        thought_parts = []
        
        for response_idx, question_response_key in responses:
            response_data = question_results[question_response_key]
            response_text = response_data.get('response_text', '')
            extracted_code_data = response_data.get('extracted_code_data', {})
            
            response_parts.append(f"rollout_{response_idx+1}: {response_text}")
            
            code = extracted_code_data.get('extracted_code', '')
            name = extracted_code_data.get('extracted_name', 'Unknown')
            thought = extracted_code_data.get('extracted_thought', '')
            
            code_parts.append(f"rollout_{response_idx+1}: {code}")
            name_parts.append(f"rollout_{response_idx+1}: {name}")
            thought_parts.append(f"rollout_{response_idx+1}: {thought}")
        
        combined_response_text = "\n\n".join(response_parts)
        combined_extracted_code = "\n\n".join(code_parts)
        combined_extracted_name = "\n\n".join(name_parts)
        combined_extracted_thought = "\n\n".join(thought_parts)
        
        # Note: Ensure none of the extracted values are None
        if combined_extracted_code is None:
            combined_extracted_code = ''
        if combined_extracted_name is None:
            combined_extracted_name = 'Unknown'
        if combined_extracted_thought is None:
            combined_extracted_thought = ''
            
        # For extraction success, consider it successful if any extraction was successful
        extraction_success = any(
            question_results[key].get('extracted_code_data', {}).get('code_extraction_success', False) 
            for _, key in responses
        )
        extraction_error = None  # Combine errors if needed
        
        code_is_valid = validate_python_code(combined_extracted_code, logger) if combined_extracted_code and extraction_success else False
        
        generated_code.append({
            'sample_index': i,
            'question': question,
            'full_response': combined_response_text,
            'extracted_code': combined_extracted_code,
            'extracted_name': combined_extracted_name,
            'extracted_thought': combined_extracted_thought,
            'extraction_success': extraction_success,
            'extraction_error': extraction_error,
            'code_is_valid': code_is_valid,
            'timestamp': str(datetime.now())
        })
    
    # Store current step's generated code
    current_step_generated_code = generated_code
    
    # Add to accumulated code (keep last N steps to avoid memory issues)
    accumulated_generated_code = [{
        'step': global_steps,
        'timestamp': str(datetime.now()),
        'generated_code': generated_code
    }]
    
    # Keep only last N steps based on configuration
    if len(accumulated_generated_code) > max_accumulated_steps:
        accumulated_generated_code = accumulated_generated_code[-max_accumulated_steps:]
    
    # Calculate statistics
    total_samples = len(generated_code)
    successful_extractions = sum(1 for code in generated_code if code['extraction_success'])
    valid_code_samples = sum(1 for code in generated_code if code['code_is_valid'])
    
    main_rank_print(f"\n{'='*50}")
    main_rank_print("GENERATED CODE COLLECTED (DICTIONARY APPROACH)")
    main_rank_print(f"{'='*50}")
    main_rank_print(f"Step: {global_steps}")
    main_rank_print(f"Total samples: {total_samples}")
    main_rank_print(f"Successful extractions: {successful_extractions}/{total_samples} ({successful_extractions/total_samples*100:.1f}%)")
    main_rank_print(f"Valid code samples: {valid_code_samples}/{total_samples} ({valid_code_samples/total_samples*100:.1f}%)")
    main_rank_print(f"Accumulated steps: {len(accumulated_generated_code)}")
    
    # Log sample names and thoughts for the first few samples
    main_rank_print(f"\nSample Names and Thoughts:")
    for i, code_sample in enumerate(generated_code[:3]):  # Show first 3 samples
        main_rank_print(f"  Sample {i+1}:")
        main_rank_print(f"    Name: {code_sample.get('extracted_name', 'Unknown')}")
        thought = code_sample.get('extracted_thought', '')
        main_rank_print(f"    Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")
    
    main_rank_print(f"{'='*50}\n")
    
    return current_step_generated_code, accumulated_generated_code


def collect_responses_and_ground_truth_from_validation_reward(batch, reward_extra_info, global_steps, save_responses_and_ground_truth):
    """Collect responses and ground truth from validation reward function output for saving."""
    if not save_responses_and_ground_truth:
        return None, None

    responses_and_ground_truth = []
    
    # Extract execution results from batch non_tensor_batch (not meta_info)
    execution_results = batch.non_tensor_batch.get('execution_results', [])
    
    # Get final answer correctness from reward extra info
    final_answer_correctness = reward_extra_info.get('final_answer_correctness', [])
    
    for i, execution_result in enumerate(execution_results):
        # Extract data from execution result
        result = execution_result.get('result', '')
        ground_truth = execution_result.get('ground_truth', '')
        success = execution_result.get('success', False)
        error = execution_result.get('error', '')
        question = execution_result.get('question', '')
        
        # Get correctness from reward function output
        answer_correct = final_answer_correctness[i] if i < len(final_answer_correctness) else False
        
        responses_and_ground_truth.append({
            'sample_index': i,
            'question': question,
            'execution_result': result,
            'ground_truth': ground_truth,
            'execution_success': success,
            'execution_error': error,
            'answer_correct': answer_correct,
            'timestamp': str(datetime.now())
        })
    
    # Store current step's responses and ground truth
    current_step_responses = responses_and_ground_truth
    
    # Add to accumulated responses (keep last N steps to avoid memory issues)
    accumulated_responses = [{
        'step': global_steps,
        'timestamp': str(datetime.now()),
        'responses_and_ground_truth': responses_and_ground_truth
    }]
    
    # Calculate statistics
    total_samples = len(responses_and_ground_truth)
    successful_executions = sum(1 for resp in responses_and_ground_truth if resp['execution_success'])
    correct_predictions = sum(1 for resp in responses_and_ground_truth if resp['answer_correct'])
    
    main_rank_print(f"\n{'='*50}")
    main_rank_print("VALIDATION RESPONSES AND GROUND TRUTH COLLECTED")
    main_rank_print(f"{'='*50}")
    main_rank_print(f"Step: {global_steps}")
    main_rank_print(f"Total samples: {total_samples}")
    main_rank_print(f"Successful executions: {successful_executions}/{total_samples} ({successful_executions/total_samples*100:.1f}%)")
    main_rank_print(f"Correct predictions: {correct_predictions}/{total_samples} ({correct_predictions/total_samples*100:.1f}%)")
    
    # Log sample results for the first few samples
    main_rank_print(f"\nSample Results:")
    for i, response_sample in enumerate(responses_and_ground_truth[:3]):  # Show first 3 samples
        main_rank_print(f"  Sample {i+1}:")
        main_rank_print(f"    Execution Result: {response_sample['execution_result']}")
        main_rank_print(f"    Ground Truth: {response_sample['ground_truth']}")
        main_rank_print(f"    Correct: {response_sample['answer_correct']}")
        main_rank_print(f"    Execution Success: {response_sample['execution_success']}")
    
    main_rank_print(f"{'='*50}\n")
    
    return current_step_responses, accumulated_responses


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_mas_r1_checkpoint(trainer_instance, save_dir: Path):
    """
    Enhanced checkpoint saving for MAS-R1 trainer.
    Saves both standard checkpoint and MAS-R1 specific data.
    
    Args:
        trainer_instance: The MAS-R1 trainer instance
        save_dir: Directory to save checkpoint to
    """
    # Call parent save_checkpoint first
    trainer_instance._save_checkpoint_parent()
    
    # Save generated code
    save_generated_code(
        save_dir=save_dir,
        global_steps=trainer_instance.global_steps,
        current_step_generated_code=getattr(trainer_instance, '_current_step_generated_code', None),
        accumulated_generated_code=getattr(trainer_instance, '_accumulated_generated_code', []),
        save_generated_code=getattr(trainer_instance, 'save_generated_code', False),
        save_code_summary=getattr(trainer_instance, 'save_code_summary', False),
        max_accumulated_steps=getattr(trainer_instance, 'max_accumulated_steps', 10)
    )
    
    # Save responses and ground truth
    save_responses_and_ground_truth(
        save_dir=save_dir,
        global_steps=trainer_instance.global_steps,
        current_step_responses=getattr(trainer_instance, '_current_step_responses', None),
        accumulated_responses=getattr(trainer_instance, '_accumulated_responses', []),
        save_responses_and_ground_truth=getattr(trainer_instance, 'save_responses_and_ground_truth', False)
    )
    
    pp.status("SAVE", f"Saved checkpoint, generated code, and responses to {save_dir}", "success")




def save_mas_r1_generated_code(trainer_instance, save_dir: Path):
    """
    Wrapper for saving generated code.
    
    Args:
        trainer_instance: The MAS-R1 trainer instance
        save_dir: Directory to save to
    """
    save_generated_code(
        save_dir=save_dir,
        global_steps=trainer_instance.global_steps,
        current_step_generated_code=getattr(trainer_instance, '_current_step_generated_code', None),
        accumulated_generated_code=getattr(trainer_instance, '_accumulated_generated_code', []),
        save_generated_code=getattr(trainer_instance, 'save_generated_code', False),
        save_code_summary=getattr(trainer_instance, 'save_code_summary', False),
        max_accumulated_steps=getattr(trainer_instance, 'max_accumulated_steps', 10)
    )


def save_mas_r1_responses(trainer_instance, save_dir: Path):
    """
    Wrapper for saving responses and ground truth.
    
    Args:
        trainer_instance: The MAS-R1 trainer instance
        save_dir: Directory to save to
    """
    save_responses_and_ground_truth(
        save_dir=save_dir,
        global_steps=trainer_instance.global_steps,
        current_step_responses=getattr(trainer_instance, '_current_step_responses', None),
        accumulated_responses=getattr(trainer_instance, '_accumulated_responses', []),
        save_responses_and_ground_truth=getattr(trainer_instance, 'save_responses_and_ground_truth', False)
    ) 
