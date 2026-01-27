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
from mas_r1_reasoner.agents.logging_utils.stdout import PrettyPrinter as pp
import re
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code


def log_mas_r1_validation_scores(val_metrics: dict, step: int):
    """
    Log MAS-R1 validation scores to console and prepare for main logger.
    The main logger (ReasonRLTracking) will handle the actual wandb logging.
    """
    if not val_metrics:
        return
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("MAS-R1 VALIDATION SCORES SUMMARY")
    main_rank_print(f"{'='*60}")
    
    # Group metrics by category
    performance_metrics = {}
    metadata_metrics = {}
    
    for k, v in val_metrics.items():
        if k.startswith('val/mas_r1/'):
            if k in ['val/mas_r1/overall_score', 'val/mas_r1/code_execution_success', 'val/mas_r1/final_answer_correctness']:
                performance_metrics[k] = v
            else:
                metadata_metrics[k] = v
    
    # Log performance metrics
    if performance_metrics:
        main_rank_print("PERFORMANCE METRICS (0-1 range):")
        for k, v in performance_metrics.items():
            main_rank_print(f"  {k}: {v}")
    
    # Log metadata metrics
    if metadata_metrics:
        main_rank_print("METADATA:")
        for k, v in metadata_metrics.items():
            main_rank_print(f"  {k}: {v}")
    

    
    # The main logger will handle wandb logging automatically
    # No need to call wandb.log() directly here
    main_rank_print(f"✓ Validation metrics will be logged to wandb by main logger at step {step}")
    main_rank_print(f"  - {len(performance_metrics)} performance metrics")
    main_rank_print(f"  - {len(metadata_metrics)} metadata metrics")
    main_rank_print(f"  - Total: {len(val_metrics)} metrics")


def maybe_log_generations_to_wandb(inputs, outputs, scores, config, global_steps, table=None,
                                 batch=None, reward_extra_info=None, mode='val'):
    """Log a table of samples to wandb (training or validation)"""

    # Determine configuration key and logging key based on mode
    if mode == 'train':
        generations_to_log = config.trainer.get('train_generations_to_log_to_wandb', 0)
        log_key = "train/generations"
        config_key_name = 'train_generations_to_log_to_wandb'
    else:  # mode == 'val'
        generations_to_log = config.trainer.val_generations_to_log_to_wandb
        log_key = "val/generations"
        config_key_name = 'val_generations_to_log_to_wandb'

    if generations_to_log == 0:
        return table

    if generations_to_log > 0 and 'wandb' not in config.trainer.logger:
        print(
            f'WARNING: `{config_key_name}` is set to a positive value, but no wandb logger is found. ')
        return table

    import wandb
    import numpy as np

    # Create tuples of (input, output, score) in original order
    samples = list(zip(inputs, outputs, scores))

    # Take first N samples without any reordering
    samples = samples[:generations_to_log]

    # Collect MAS-R1 specific sample data if available
    ground_truths = []
    code_execution_scores = []
    final_answer_scores = []
    combined_scores = []
    questions = []
    mas_codes = []
    extracted_answers = []  # Raw execution output
    predicted_answers = []  # Processed predicted answer from reward computation

    if batch is not None and reward_extra_info is not None:
        # Validate that required MAS-R1 detailed scores are present
        required_scores = ['code_execution_success', 'final_answer_correctness', 'combined_reward', 'predicted_answer']
        missing_scores = [score for score in required_scores if score not in reward_extra_info]
        
        if missing_scores:
            error_msg = f"ERROR: Missing required detailed scores in reward_extra_info for {mode} logging: {missing_scores}"
            error_msg += f"\nAvailable keys in reward_extra_info: {list(reward_extra_info.keys())}"
            error_msg += f"\nThis indicates a problem with the reward computation pipeline."
            error_msg += f"\nThe reward manager should be storing these scores in reward_extra_info."
            raise RuntimeError(error_msg)
        
        # Note: Extract MAS-R1 specific metrics from reward_extra_info (all samples)
        # Be careful, if you have a very large batch size, this will cause memory issues
        code_execution_scores.extend(reward_extra_info['code_execution_success'])
        final_answer_scores.extend(reward_extra_info['final_answer_correctness'])
        combined_scores.extend(reward_extra_info['combined_reward'])
        predicted_answers.extend(reward_extra_info['predicted_answer'])
        
        # Extract other sample data (all samples)
        for i in range(len(batch)):
            # Extract ground truth
            ground_truth = batch[i].non_tensor_batch['reward_model'].get('ground_truth', 'N/A')
            ground_truths.append(ground_truth)
            
            # Extract question and MAS code from execution results and stats
            execution_results = batch[i].non_tensor_batch.get('execution_results', [])
            execution_stats = batch[i].non_tensor_batch.get('execution_stats', {})
            
            # Determine if this is multiple execution case
            total_executions = execution_stats.get('total_executions', 1) if execution_stats else 1
            
            if total_executions > 1:
                raise RuntimeError(f"Multiple executions are not supported for wandb logging")
            else:
                # Single execution: extract data from execution_results
                if execution_results:
                    # Handle execution_results as a dictionary (single execution result)
                    if isinstance(execution_results, dict):
                        question = execution_results.get('question', 'N/A')
                        mas_code = execution_results.get('code', 'N/A')
                        # Use error if it exists and is not empty, otherwise use result
                        error = execution_results.get('error')
                        if error:
                            extracted_answers.append(error)
                        else:
                            extracted_answers.append(execution_results.get('result'))
                    else:
                        # Fallback for list/array case
                        if i < len(execution_results):
                            question = execution_results[i].get('question', 'N/A')
                            mas_code = execution_results[i].get('code', 'N/A')
                            # Use error if it exists and is not empty, otherwise use result
                            error = execution_results[i].get('error')
                            if error:
                                extracted_answers.append(error)
                            else:
                                extracted_answers.append(execution_results[i].get('result'))
                        else:
                            raise RuntimeError(f"No execution_results found or index {i} out of bounds for sample {i}")
                else:
                    raise RuntimeError(f"No execution_results found or index {i} out of bounds for sample {i}")
            
            questions.append(question)
            mas_codes.append(mas_code)

    # Create column names for one sample per row format
    columns = ["step", "input", "output", "score"] 
    
    # Add MAS-R1 specific columns if data is available
    # Raise errors if required MAS-R1 data is missing
    if not ground_truths or len(ground_truths) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient ground_truths data for {mode} logging. Expected {len(samples)}, got {len(ground_truths) if ground_truths else 0}")
    columns.append("ground_truth")
    
    if not code_execution_scores or len(code_execution_scores) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient code_execution_scores data for {mode} logging. Expected {len(samples)}, got {len(code_execution_scores) if code_execution_scores else 0}")
    columns.append("code_execution")
    
    if not final_answer_scores or len(final_answer_scores) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient final_answer_scores data for {mode} logging. Expected {len(samples)}, got {len(final_answer_scores) if final_answer_scores else 0}")
    columns.append("final_answer")
    
    if not combined_scores or len(combined_scores) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient combined_scores data for {mode} logging. Expected {len(samples)}, got {len(combined_scores) if combined_scores else 0}")
    columns.append("combined_score")
    
    if not questions or len(questions) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient questions data for {mode} logging. Expected {len(samples)}, got {len(questions) if questions else 0}")
    columns.append("question")
    
    if not mas_codes or len(mas_codes) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient mas_codes data for {mode} logging. Expected {len(samples)}, got {len(mas_codes) if mas_codes else 0}")
    columns.append("mas_code")
    
    if not predicted_answers or len(predicted_answers) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient predicted_answers data for {mode} logging. Expected {len(samples)}, got {len(predicted_answers) if predicted_answers else 0}")
    columns.append("predicted_answer")  # Processed predicted answer from reward computation
    
    if not extracted_answers or len(extracted_answers) < len(samples):
        raise RuntimeError(f"ERROR: Missing or insufficient extracted_answers data for {mode} logging. Expected {len(samples)}, got {len(extracted_answers) if extracted_answers else 0}")
    columns.append("execution_output")  # Raw execution output

    if table is None:
        # Initialize the table on first call
        table = wandb.Table(columns=columns)

    # Create a new table with same columns and existing data
    # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
    new_table = wandb.Table(columns=columns, data=table.data)

    # Add one row per sample
    for i, (input_text, output_text, score) in enumerate(samples):
        row_data = [global_steps, input_text, output_text, score]
        
        # Add MAS-R1 specific data (already validated above)
        row_data.append(ground_truths[i])
        row_data.append(code_execution_scores[i])
        row_data.append(final_answer_scores[i])
        row_data.append(combined_scores[i])
        row_data.append(questions[i])
        row_data.append(mas_codes[i])
        row_data.append(predicted_answers[i])  # predicted_answer column (processed from reward computation)
        row_data.append(extracted_answers[i])  # execution_output column (raw execution result)

        new_table.add_data(*row_data)

    # Update reference and log
    wandb.log({log_key: new_table}, step=global_steps)
    
    # Log confirmation message similar to log_mas_r1_validation_scores
    main_rank_print(f"✓ {mode.capitalize()} generations will be logged to wandb by main logger at step {global_steps}")
    main_rank_print(f"  - {len(samples)} samples logged to {log_key}")
    
    # Print first row data for debugging
    try:
        first_sample = samples[0]
        first_input, first_output, first_score = first_sample
        main_rank_print(f"  - First row data:")
        main_rank_print(f"    Input: {first_input[:100]}{'...' if len(first_input) > 100 else ''}")
        main_rank_print(f"    Output: {first_output[:100]}{'...' if len(first_output) > 100 else ''}")
        main_rank_print(f"    Score: {first_score}")
        main_rank_print(f"    Ground Truth: {ground_truths[0][:100] if isinstance(ground_truths[0], str) else ground_truths[0]}{'...' if isinstance(ground_truths[0], str) and len(ground_truths[0]) > 100 else ''}")
        main_rank_print(f"    Code Execution Score: {code_execution_scores[0]}")
        main_rank_print(f"    Final Answer Score: {final_answer_scores[0]}")
        main_rank_print(f"    Combined Score: {combined_scores[0]}")
        main_rank_print(f"    Question: {questions[0][:100]}{'...' if len(questions[0]) > 100 else ''}")
        main_rank_print(f"    MAS Code: {mas_codes[0][:100]}{'...' if len(mas_codes[0]) > 100 else ''}")
        main_rank_print(f"    Predicted Answer: {predicted_answers[0][:100]}{'...' if len(str(predicted_answers[0])) > 100 else ''}")
        main_rank_print(f"    Execution Output: {extracted_answers[0][:100]}{'...' if len(extracted_answers[0]) > 100 else ''}")
    except Exception as e:
        main_rank_print(f"  - Error printing first row data: {e}")

    return new_table

def maybe_log_val_generations_to_wandb(inputs, outputs, scores, config, global_steps, validation_table=None,
                                     test_batch=None, reward_extra_info=None):
    """Log a table of validation samples to wandb (alias for backward compatibility)"""
    return maybe_log_generations_to_wandb(
        inputs=inputs, outputs=outputs, scores=scores, config=config, 
        global_steps=global_steps, table=validation_table, batch=test_batch, 
        reward_extra_info=reward_extra_info, mode='val'
    )
    

def maybe_log_train_generations_to_wandb(inputs, outputs, scores, config, global_steps, training_table=None,
                                       train_batch=None, reward_extra_info=None):
    """Log a table of training samples to wandb (alias for backward compatibility)"""
    return maybe_log_generations_to_wandb(
        inputs=inputs, outputs=outputs, scores=scores, config=config, 
        global_steps=global_steps, table=training_table, batch=train_batch, 
        reward_extra_info=reward_extra_info, mode='train'
    )


def collect_samples_by_file(test_batch, reward_tensor, reward_extra_info, input_texts, sample_outputs, 
                           samples_by_file):
    """
    Collect and organize validation samples by their source file.
    
    Args:
        test_batch: DataProto containing validation batch
        reward_tensor: Tensor of rewards
        reward_extra_info: Dictionary of extra reward information
        input_texts: List of input text strings
        sample_outputs: List of output strings (accumulated across batches)
        samples_by_file: Existing dictionary to merge into (can be empty)
        
    Returns:
        dict: Updated dictionary mapping source_file -> sample data
    """
    # Calculate scores for this batch
    batch_scores = reward_tensor.sum(-1).cpu().tolist()
    
    # Get source files
    source_files = test_batch.non_tensor_batch.get('source_file', [None] * len(test_batch))
    current_output_offset = len(sample_outputs)
    
    for i in range(len(test_batch)):
        source_file = source_files[i] if i < len(source_files) else None
        if source_file is not None:
            # Initialize dict for this source file if not exists
            if source_file not in samples_by_file:
                samples_by_file[source_file] = {
                    'inputs': [],
                    'outputs': [],
                    'scores': [],
                    'batches': [],  # Store individual sample batches
                    'reward_infos': []  # Store individual sample reward_extra_info
                }
            
            # Add sample data to dictionary
            samples_by_file[source_file]['inputs'].append(input_texts[i])
            
            # Get the corresponding output for this sample
            output_idx = current_output_offset + i
            samples_by_file[source_file]['outputs'].append(
                sample_outputs[output_idx] if output_idx < len(sample_outputs) else ""
            )
            samples_by_file[source_file]['scores'].append(batch_scores[i])
            
            # Create a single-sample batch for per-file logging
            sample_batch = test_batch[i:i+1]
            samples_by_file[source_file]['batches'].append(sample_batch)
            
            # Create single-sample reward_extra_info
            sample_reward_info = {}
            for key, values in reward_extra_info.items():
                if isinstance(values, list) and i < len(values):
                    sample_reward_info[key] = [values[i]]
                else:
                    sample_reward_info[key] = []
            samples_by_file[source_file]['reward_infos'].append(sample_reward_info)
    
    return samples_by_file


def accumulate_per_file_metrics(test_batch, reward_extra_info, scores, per_file_metrics):
    """
    Accumulate per-file validation metrics for line chart logging.
    
    Args:
        test_batch: DataProto containing validation batch
        reward_extra_info: Dictionary of extra reward information
        scores: List of scores for this batch
        per_file_metrics: Existing dictionary to accumulate into (can be empty)
        
    Returns:
        dict: Updated dictionary mapping source_file -> metrics
    """
    # Extract source_file from test_batch
    source_files = test_batch.non_tensor_batch.get('source_file', [None] * len(test_batch))
    
    for i in range(len(test_batch)):
        source_file = source_files[i] if i < len(source_files) else None
        if source_file is not None:
            # Initialize metrics dict for this source file if not exists
            if source_file not in per_file_metrics:
                per_file_metrics[source_file] = {
                    'overall_scores': [],
                    'code_execution_success': [],
                    'final_answer_correctness': [],
                    'scores': []
                }
            
            # Accumulate metrics for this sample
            if 'combined_reward' in reward_extra_info and i < len(reward_extra_info['combined_reward']):
                per_file_metrics[source_file]['overall_scores'].append(reward_extra_info['combined_reward'][i])
            
            if 'code_execution_success' in reward_extra_info and i < len(reward_extra_info['code_execution_success']):
                per_file_metrics[source_file]['code_execution_success'].append(reward_extra_info['code_execution_success'][i])
            
            if 'final_answer_correctness' in reward_extra_info and i < len(reward_extra_info['final_answer_correctness']):
                per_file_metrics[source_file]['final_answer_correctness'].append(reward_extra_info['final_answer_correctness'][i])
            
            # Add score for this sample
            if i < len(scores):
                per_file_metrics[source_file]['scores'].append(scores[i])
    
    return per_file_metrics


def log_per_file_generations_to_wandb(inputs, outputs, scores, config, global_steps, table=None,
                                      test_batch=None, reward_extra_info=None, file_name='unknown'):
    """Log a table of validation samples for a specific file to wandb"""
    
    # Check if wandb logging is enabled
    generations_to_log = config.trainer.val_generations_to_log_to_wandb
    if generations_to_log == 0:
        return table
    
    if generations_to_log > 0 and 'wandb' not in config.trainer.logger:
        return table
    
    import wandb
    import numpy as np
    
    # Create tuples of (input, output, score) in original order
    samples = list(zip(inputs, outputs, scores))
    
    # Take first N samples without any reordering
    samples = samples[:generations_to_log]
    
    # Collect MAS-R1 specific sample data if available
    ground_truths = []
    code_execution_scores = []
    final_answer_scores = []
    combined_scores = []
    questions = []
    mas_codes = []
    extracted_answers = []
    predicted_answers = []
    
    if test_batch is not None and reward_extra_info is not None:
        # Validate that required MAS-R1 detailed scores are present
        required_scores = ['code_execution_success', 'final_answer_correctness', 'combined_reward', 'predicted_answer']
        missing_scores = [score for score in required_scores if score not in reward_extra_info]
        
        if not missing_scores:
            # Extract MAS-R1 specific metrics from reward_extra_info
            code_execution_scores.extend(reward_extra_info['code_execution_success'][:len(samples)])
            final_answer_scores.extend(reward_extra_info['final_answer_correctness'][:len(samples)])
            combined_scores.extend(reward_extra_info['combined_reward'][:len(samples)])
            predicted_answers.extend(reward_extra_info['predicted_answer'][:len(samples)])
            
            # Extract other sample data
            for i in range(min(len(samples), len(test_batch))):
                # Extract ground truth
                ground_truth = test_batch[i].non_tensor_batch['reward_model'].get('ground_truth', 'N/A')
                ground_truths.append(ground_truth)
                
                # Extract question and MAS code from execution results
                execution_results = test_batch[i].non_tensor_batch.get('execution_results', [])
                
                if execution_results:
                    if isinstance(execution_results, dict):
                        question = execution_results.get('question', 'N/A')
                        mas_code = execution_results.get('code', 'N/A')
                        error = execution_results.get('error')
                        extracted_answers.append(error if error else execution_results.get('result', 'N/A'))
                    else:
                        if i < len(execution_results):
                            question = execution_results[i].get('question', 'N/A')
                            mas_code = execution_results[i].get('code', 'N/A')
                            error = execution_results[i].get('error')
                            extracted_answers.append(error if error else execution_results[i].get('result', 'N/A'))
                        else:
                            question = 'N/A'
                            mas_code = 'N/A'
                            extracted_answers.append('N/A')
                else:
                    question = 'N/A'
                    mas_code = 'N/A'
                    extracted_answers.append('N/A')
                
                questions.append(question)
                mas_codes.append(mas_code)
    
    # Create column names
    columns = ["step", "input", "output", "score"]
    
    # Add MAS-R1 specific columns if data is available
    if ground_truths and len(ground_truths) >= len(samples):
        columns.append("ground_truth")
    if code_execution_scores and len(code_execution_scores) >= len(samples):
        columns.append("code_execution")
    if final_answer_scores and len(final_answer_scores) >= len(samples):
        columns.append("final_answer")
    if combined_scores and len(combined_scores) >= len(samples):
        columns.append("combined_score")
    if questions and len(questions) >= len(samples):
        columns.append("question")
    if mas_codes and len(mas_codes) >= len(samples):
        columns.append("mas_code")
    if predicted_answers and len(predicted_answers) >= len(samples):
        columns.append("predicted_answer")
    if extracted_answers and len(extracted_answers) >= len(samples):
        columns.append("execution_output")
    
    if table is None:
        # Initialize the table on first call
        table = wandb.Table(columns=columns)
    
    # Create a new table with same columns and existing data
    new_table = wandb.Table(columns=columns, data=table.data)
    
    # Add one row per sample
    for i, (input_text, output_text, score) in enumerate(samples):
        row_data = [global_steps, input_text, output_text, score]
        
        # Add MAS-R1 specific data if available
        if ground_truths and i < len(ground_truths):
            row_data.append(ground_truths[i])
        if code_execution_scores and i < len(code_execution_scores):
            row_data.append(code_execution_scores[i])
        if final_answer_scores and i < len(final_answer_scores):
            row_data.append(final_answer_scores[i])
        if combined_scores and i < len(combined_scores):
            row_data.append(combined_scores[i])
        if questions and i < len(questions):
            row_data.append(questions[i])
        if mas_codes and i < len(mas_codes):
            row_data.append(mas_codes[i])
        if predicted_answers and i < len(predicted_answers):
            row_data.append(predicted_answers[i])
        if extracted_answers and i < len(extracted_answers):
            row_data.append(extracted_answers[i])
        
        new_table.add_data(*row_data)
    
    # Log with file-specific key
    log_key = f"val/per_file_table/{file_name}"
    wandb.log({log_key: new_table}, step=global_steps)
    
    main_rank_print(f"✓ Logged {len(samples)} samples to {log_key}")
    
    return new_table


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


def collect_generated_code_from_dict(question_results, global_steps, save_generated_code, 
                                   max_accumulated_steps, logger=None):
    """Collect generated code from Stage 1 for saving, using dictionary-based approach."""
    if not save_generated_code:
        return None, None

    generated_code = []
    
    # Collect all {question}<<MySep>>{response} keys
    question_response_keys = []
    for key in question_results.keys():
        if '<<MySep>>' in key and key.split('<<MySep>>')[-1].isdigit():
            question_response_keys.append(key)
    
    # Sort by question and response index for consistent ordering
    question_response_keys.sort(key=lambda x: (x.rsplit('<<MySep>>', 1)[0], int(x.rsplit('<<MySep>>', 1)[1])))
    
    # Group by question to combine multiple responses
    question_groups = {}
    for question_response_key in question_response_keys:
        question = question_response_key.rsplit('<<MySep>>', 1)[0]
        response_idx = int(question_response_key.rsplit('<<MySep>>', 1)[1])
        
        if question not in question_groups:
            question_groups[question] = []
        question_groups[question].append((response_idx, question_response_key))
    
    # Sort responses within each question group
    for question in question_groups:
        question_groups[question].sort(key=lambda x: x[0])
    
    for i, (question, responses) in enumerate(question_groups.items()):
            
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


def create_mas_r1_dataloaders(config, tokenizer):
    """
    Create MAS-R1 specific dataloaders using PreprocessedRLDataset and StatefulDataLoader.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        
    Returns:
        tuple: (train_dataloader, val_dataloader, total_training_steps)
    """
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    from omegaconf import open_dict
    
    # Import collate_fn for dataloader
    try:
        from verl.utils.dataset.rl_dataset import collate_fn
    except ImportError:
        # Fallback import path
        from mas_r1_reasoner.agents.dataset.rl_dataset import collate_fn
    
    main_rank_print(f"\n{'='*60}")
    main_rank_print("CREATING PREPROCESSED DATALOADERS WITH STATEFULDATALOADER")
    main_rank_print(f"{'='*60}")
    main_rank_print(f"Train files: {config.data.train_files}")
    main_rank_print(f"Val files: {config.data.val_files}")
    main_rank_print(f"Using PreprocessedRLDataset with StatefulDataLoader for VERL compatibility")
    main_rank_print(f"{'='*60}\n")
    
    # Import PreprocessedRLDataset here to avoid circular imports
    try:
        from verl.utils.dataset.preprocessed_rl_dataset import PreprocessedRLDataset
    except ImportError:
        # Fallback import path
        from mas_r1_reasoner.agents.dataset.preprocessed_rl_dataset import PreprocessedRLDataset
    
    # Create training dataset using PreprocessedRLDataset
    train_dataset = PreprocessedRLDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,  # Not used for preprocessed data but kept for compatibility
        prompt_key='prompt_text',  # Key for the raw prompt text in preprocessed data
        input_ids_key='input_ids',  # Key for preprocessed input_ids
        attention_mask_key='attention_mask',  # Key for preprocessed attention_mask
        max_prompt_length=config.data.max_prompt_length,  # Not used for preprocessed data
        filter_prompts=False,  # Not used for preprocessed data
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,  # Not used for preprocessed data
        return_raw_chat=True,  # Always return raw chat for preprocessed data
        truncation='error',  # Not used for preprocessed data
        extra_source_key=f"mas_r1_preprocessed_train"
    )
    
    # Create validation dataset using PreprocessedRLDataset
    val_dataset = PreprocessedRLDataset(
        parquet_files=config.data.val_files,
        tokenizer=tokenizer,  # Not used for preprocessed data but kept for compatibility
        prompt_key='prompt_text',  # Key for the raw prompt text in preprocessed data
        input_ids_key='input_ids',  # Key for preprocessed input_ids
        attention_mask_key='attention_mask',  # Key for preprocessed attention_mask
        max_prompt_length=config.data.max_validation_prompt_length,  # Not used for preprocessed data
        filter_prompts=False,  # Not used for preprocessed data
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,  # Not used for preprocessed data
        return_raw_chat=True,  # Always return raw chat for preprocessed data
        truncation='error',  # Not used for preprocessed data
        extra_source_key=f"mas_r1_preprocessed_val"
    )
    
    # Use sampler for better checkpoint resume
    if config.data.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.data.get('seed', 1))
        sampler = RandomSampler(train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(train_dataset)
    
    # Create training dataloader using StatefulDataLoader (VERL compatible)
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.data.train_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    # Create validation dataloader using StatefulDataLoader (VERL compatible)
    val_batch_size = config.data.val_batch_size  # Prefer config value if set
    if val_batch_size is None:
        val_batch_size = len(val_dataset)
    
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,  # No shuffling for validation
        drop_last=False,  # Don't drop last for validation
        collate_fn=collate_fn,
    )
    
    assert len(train_dataloader) >= 1, "Train dataloader is empty!"
    assert len(val_dataloader) >= 1, "Validation dataloader is empty!"
    
    main_rank_print(f"Size of train dataloader: {len(train_dataloader)}")
    main_rank_print(f"Size of val dataloader: {len(val_dataloader)}")
    
    # Calculate total training steps
    total_training_steps = len(train_dataloader) * config.trainer.total_epochs
    
    if config.trainer.total_training_steps is not None:
        total_training_steps = config.trainer.total_training_steps
    
    main_rank_print(f"Total training steps: {total_training_steps}")
    
    # Inject total_training_steps to actor/critic optim_config
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        if OmegaConf.select(config, "actor_rollout_ref.actor.optim"):
            config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        if OmegaConf.select(config, "critic.optim"):
            config.critic.optim.total_training_steps = total_training_steps
    
    main_rank_print(f"✅ Created StatefulDataLoader instances with VERL compatibility")
    main_rank_print(f"✅ Dataloaders support state_dict() and load_state_dict() for checkpointing")
    main_rank_print(f"✅ Will work with VERL's standard _load_checkpoint method")
    main_rank_print(f"{'='*60}\n")
    
    return train_dataloader, val_dataloader, total_training_steps


def initialize_mas_r1_agent_system(trainer_instance, config):
    """
    Initialize MAS-R1 agent system.
    
    Args:
        trainer_instance: The MAS-R1 trainer instance
        config: Training configuration
    """
    if trainer_instance.agent_system is None:
        # Configure MAS agent from config
        agent_config = trainer_instance.mas_r1_config.get('agent', {})
        main_rank_print(f"MAS agent config: {agent_config}")
        
        # Global variables are already set up by the caller
        # No need to call _setup_global_variables_from_dataset_processor again
        
        # Choose between async and sequential AgentSystem based on configuration
        if trainer_instance.enable_async_execution:
            main_rank_print(f"Initializing ASYNC AgentSystem...")
            trainer_instance.agent_system = AsyncAgentSystem(agent_config)
            main_rank_print(f"✓ Async AgentSystem initialized with agent configuration")
        else:
            main_rank_print(f"Initializing SEQUENTIAL AgentSystem...")
            trainer_instance.agent_system = SequentialAgentSystem(agent_config)
            main_rank_print(f"✓ Sequential AgentSystem initialized with agent configuration")
    else:
        main_rank_print(f"AgentSystem already initialized")


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

def compute_pass_at_k_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], val_n: int, seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute pass@k metrics for powers of 2 up to the total number of responses per question.
    
    This function groups responses by original questions and computes pass@k metrics
    by taking the first k responses and checking if at least one is correct.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        val_n: Number of responses per question (validation n parameter).
        seed: Random seed (not used in this implementation). Defaults to 42.
        
    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "pass@1/mean": Pass@1 (first response correctness)
        - "pass@2/mean": Pass@2 (at least one correct in first 2 responses)
        - "pass@4/mean": Pass@4 (at least one correct in first 4 responses)
        - etc. (powers of 2 up to val_n)
    """
    import numpy as np
    from collections import defaultdict
    
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate pass@k metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                
                # Compute pass@k for powers of 2 up to n_resps
                ns = []
                n = 1  # Start from 1 (unlike VERL which starts from 2)
                while n <= n_resps:
                    ns.append(n)
                    n *= 2
                
                # Remove duplicates and ensure we don't exceed n_resps
                ns = sorted(list(set(ns)))
                ns = [n for n in ns if n <= n_resps]

                for n in ns:
                    # Take the first n responses and check if at least one is correct (> 0)
                    first_n_responses = var_vals[:n]
                    pass_at_k = 1.0 if any(val > 0 for val in first_n_responses) else 0.0
                    metric[f"pass@{n}/mean"] = pass_at_k

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val 