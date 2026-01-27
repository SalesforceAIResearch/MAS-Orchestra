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
# from mas_r1_reasoner.agents.agent_system_process import AsyncAgentSystem as ProcessAgentSystem
from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem as AsyncAgentSystem
import re
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code
from mas_r1_reasoner.agents.shared_vars import get_global
# Building blocks execution will be imported when needed
import torch
import copy
from verl.utils.model import compute_position_id_with_mask
from mas_r1_reasoner.agents.common import get_prompt, get_known_prompt


def tokenize_and_left_pad(text, max_length, tokenizer, pad_token_id):
    """
    Tokenize and left-pad text to a specified max_length.
    
    Args:
        text: Text to tokenize
        max_length: Maximum sequence length
        tokenizer: Tokenizer instance
        pad_token_id: Padding token ID
        
    Returns:
        tuple: (input_ids, attention_mask)
    """
    new_ids = tokenizer.encode(text, return_tensors='pt')[0]
    
    # Truncate if necessary
    if new_ids.size(0) > max_length:
        new_ids = new_ids[:max_length]
        padding_length = 0
    else:
        # Left pad if necessary
        padding_length = max_length - new_ids.size(0)
        if padding_length > 0:
            new_ids = torch.cat([torch.full((padding_length,), pad_token_id), new_ids])
    
    # Create attention mask
    attention_mask = torch.ones_like(new_ids)
    if padding_length > 0:
        attention_mask[:padding_length] = 0
    
    return new_ids, attention_mask

def _truncate_building_block_output(block_output: str, max_tokens_before_answer: int = 50) -> str:
    """
    Truncate building block output to keep only max_tokens_before_answer tokens before final "Answer:" + the answer.
    
    Args:
        block_output: The full building block output text
        max_tokens_before_answer: Maximum tokens to keep before the final answer
        
    Returns:
        Truncated output string
    """
    if not block_output or not block_output.strip():
        return block_output
    
    # Find the last occurrence of "\n\nAnswer:" (case-insensitive)
    import re
    answer_pattern = r'(?i)\n\nAnswer\s*:'
    matches = list(re.finditer(answer_pattern, block_output))
    
    if not matches:
        # No "Answer:" found, truncate to reasonable length (e.g., 200 tokens)
        # usuall means it is not a valid answer
        words = block_output.split()
        if len(words) <= max_tokens_before_answer:
            return block_output
        else:
            return " ".join(words[:max_tokens_before_answer]) + "..."
    
    # Get the last "Answer:" position
    last_answer_match = matches[-1]
    answer_start = last_answer_match.start()
    
    # Get text before the answer
    text_before_answer = block_output[:answer_start]
    answer_and_after = block_output[answer_start:]
    
    # Approximate tokenization by splitting on whitespace
    # This is not perfect but good enough for truncation purposes
    words_before = text_before_answer.split()
    
    if len(words_before) <= max_tokens_before_answer:
        # Text before answer is already short enough
        return block_output
    else:
        # Keep only the last max_tokens_before_answer words before the answer
        truncated_before = " ".join(words_before[-max_tokens_before_answer:])
        # Add ellipsis to indicate truncation
        truncated_output = "..." + truncated_before + " " + answer_and_after
        
        # Log truncation for debugging
        original_length = len(words_before)
        main_rank_print(f"  📏 Truncated building block output: {original_length} words → {max_tokens_before_answer} words before Answer:")
        
        return truncated_output

def _execute_basic_building_blocks_batch(self, questions: List[str]) -> List[Dict[str, str]]:
    """
    Execute the given basic building blocks (CoT, CoT_SC, LLM_debate, Reflexion, WebSearch...) on a batch of questions in parallel
    Returns a list of dictionaries with their outputs to be used as placeholders
    """
    try:
        # Use sync execution with process pool
        from mas_r1_reasoner.trainer.utils.execute_blocks import execute_basic_blocks_batch

        main_rank_print(f"  -> Executing building blocks for batch of {len(questions)} questions using execute_mas_batch_sync")
        
        # Execute the building blocks in parallel using the high-performance execute_mas_batch_sync infrastructure
        # Use the trainer's configured timeout value instead of hardcoded 300
        timeout = self.code_execution_timeout
        main_rank_print(f"  -> Using timeout: {timeout} seconds per building block")
        
        results = execute_basic_blocks_batch(
            questions=questions,
            timeout=timeout,
            config=self.config
        )

        # Convert results to the expected format (list of block_outputs dicts)
        batch_outputs = []
        for result in results:
            batch_outputs.append(result['block_outputs'])
        
        main_rank_print(f"  -> Building blocks completed for all {len(questions)} questions")
        return batch_outputs
        
    except Exception as e:
        main_rank_print(f"  -> ERROR: executing building blocks batch: {e}")
        raise Exception(f"Failed to execute building blocks batch: {e}")

def _prepare_raw_data_batch_for_generation(self, batch: DataProto) -> DataProto:
    """
    Prepare raw data batch for generation by constructing MAS prompts and tokenizing on-the-fly.
    
    Args:
        batch: Raw data batch containing questions
        
    Returns:
        DataProto: Batch with tokenized tensors added and then popped
    """
    main_rank_print(f"RAW DATA MODE: Constructing MAS prompts and tokenizing on-the-fly")
    
    # Load MAS templates
    try:
        # Check if known_prompt is configured
        known_prompt = get_global("global_known_prompt")
        problem_type = get_global("global_problem_type")

        if known_prompt is not None:
            # Use known prompt instead of dynamic prompt
            system_prompt, mas_prompt = get_known_prompt("PLACEHOLDER_QUESTION", known_prompt)
            developer_prompt = None
        else:
            if problem_type in ["harmony_minimal", "harmony_medium"]:
                system_prompt, mas_prompt, developer_prompt = get_prompt("PLACEHOLDER_QUESTION")
            else:
                # Use dynamic prompt as before
                system_prompt, mas_prompt = get_prompt("PLACEHOLDER_QUESTION")
                developer_prompt = None
                
        mas_template = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mas_prompt}
        ]
        
        #TODO: we cannot use "developer" as the apply_chat_templte do not support it
        # Add developer prompt if available (should be after system but before user)
        #TODO: developer prompt is in the last, as we replace the question with the first user prompt

        
        if developer_prompt is not None:
            mas_template.append({"role": "user", "content": developer_prompt})
                    
    except Exception as e:
        raise RuntimeError(f"Failed to load MAS prompt templates: {e}")

    
    # Process each sample in the batch
    batch_size = len(batch)
    all_input_ids = []
    all_attention_masks = []
    all_position_ids = []
    
    # Check if add_judge is enabled and pre-execute building blocks for the entire batch
    add_judge = get_global("global_add_judge")
    
    # Collect all questions first
    questions = []
    for i in range(batch_size):
        question = batch.non_tensor_batch.get('question', [None])[i]
        if question is None:
            raise ValueError(f"No question found in batch.non_tensor_batch at index {i}")
        questions.append(question)
    
    # Execute building blocks for all questions in parallel if add_judge is enabled
    eval_building_blocks = get_global("global_eval_building_blocks")

    if add_judge or eval_building_blocks:
        main_rank_print(f"ADD_JUDGE=True: Executing building blocks for entire batch of {batch_size} questions in parallel")
        try:
            # Run the synchronous batch execution (now using execute_mas_batch_sync)
            batch_building_block_outputs = _execute_basic_building_blocks_batch(self, questions)
        except Exception as e:
            main_rank_print(f"  -> ERROR: Failed to execute building blocks batch: {e}")
            raise Exception(f"Failed to execute building blocks batch: {e}")
    else:
        batch_building_block_outputs = [{}] * batch_size
    
    for i in range(batch_size):
        # Get raw prompt data
        raw_prompt = batch.non_tensor_batch.get('raw_prompt', [None])[i]
        question = batch.non_tensor_batch.get('question', [None])[i]

        if raw_prompt is None:
            # Fallback to question if raw_prompt not available
            raise ValueError("No raw_prompt found in batch.non_tensor_batch")
        if question is None:
            raise ValueError("No question found in batch.non_tensor_batch")
        
        
        # Use pre-computed building block outputs for this question
        block_outputs = batch_building_block_outputs[i]
        
        # Create MAS prompt using template
        chat_template = copy.deepcopy(mas_template)
        for msg in chat_template:
            if msg.get("role") == "user":
                # Replace main question placeholder
                msg["content"] = msg["content"].replace("PLACEHOLDER_QUESTION", question)
                
                # Replace building block output placeholders if add_judge is enabled
                if add_judge and block_outputs:
                    for block_key, block_output in block_outputs.items():
                        # Truncate block output to keep only 50 tokens before final "Answer:"
                        truncated_output = _truncate_building_block_output(str(block_output), max_tokens_before_answer=100)
                        
                        # Replace placeholders like [CoT], [CoT_SC], etc.
                        placeholder = "[" + block_key + "]"
                        msg["content"] = msg["content"].replace(placeholder, truncated_output)
                break
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and getattr(self.tokenizer, 'chat_template', None):
            code_prompt = self.tokenizer.apply_chat_template(
                chat_template, tokenize=False, add_generation_prompt=True
            )
        else:
            raise RuntimeError("Tokenizer does not support chat template")
        
        # Tokenize and left-pad
        input_ids, attention_mask = tokenize_and_left_pad(code_prompt, self.config.data.max_prompt_length, self.tokenizer, self.tokenizer.pad_token_id)
        
        # Compute position_ids
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]
        
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_position_ids.append(position_ids)
    
    # Stack tensors
    input_ids = torch.stack(all_input_ids)
    attention_mask = torch.stack(all_attention_masks)
    position_ids = torch.stack(all_position_ids)
    
    # Remove placeholder tensors from batch.batch if they exist
    # These were added by RawDataDataset to satisfy VERL's DataProto requirements
    placeholder_count = 0
    if 'input_ids' in batch.batch:
        # The tensors already exist with correct names, just replace their values
        placeholder_count += 1
    if 'attention_mask' in batch.batch:
        placeholder_count += 1
    if 'position_ids' in batch.batch:
        placeholder_count += 1
    
    if placeholder_count > 0:
        main_rank_print(f"✅ Found {placeholder_count} placeholder tensor fields, replacing with actual values")
    
    # Replace placeholder tensor values with actual tokenized data
    batch.batch["input_ids"] = input_ids
    batch.batch["attention_mask"] = attention_mask
    batch.batch["position_ids"] = position_ids
    
    
    main_rank_print(f"✅ Tokenized {batch_size} samples on-the-fly")
    main_rank_print(f"✅ Input shape: {input_ids.shape}")
    
    # Decode and print first input_ids for debugging
    first_input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    main_rank_print(f"🔍 First input text: {first_input_text} [CUT OFF]")
    
    # Check if building blocks evaluation is enabled and save results (only one batch in evaluation)
    if eval_building_blocks:
        main_rank_print(f"📊 BUILDING BLOCKS EVALUATION: Saving results for {len(questions)} questions")
        
        # Save results using helper function (it will handle ground truth extraction and format conversion)
        save_building_blocks_evaluation_results(
            questions=questions,
            batch=batch,
            batch_building_block_outputs=batch_building_block_outputs,
            config=self.config,
        )
        
        main_rank_print(f"✅ Building blocks evaluation results saved!")
    
    # Now pop the tensors for generation (standard behavior)
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    
    return gen_batch


def _prepare_preprocessed_batch_for_generation(self, batch: DataProto) -> DataProto:
    """
    Prepare preprocessed batch for generation by popping existing tokenized tensors.
    
    Args:
        batch: Preprocessed batch containing input_ids, attention_mask, position_ids
        
    Returns:
        DataProto: Batch with tokenized tensors popped
    """
    main_rank_print(f"PREPROCESSED MODE: Using pre-tokenized data")
    
    # pop those keys for generation
    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
    
    return gen_batch


def prepare_batch_for_generation(self, batch: DataProto) -> DataProto:
    """
    Prepare batch for Stage 1 generation by popping input tensors.
    This is shared between training and validation.
    
    For raw data mode, this function also constructs MAS prompts and tokenizes them on-the-fly.
    """
    main_rank_print(f"\n{'='*60}")
    main_rank_print("PREPARING BATCH FOR MAS CODE GENERATION")
    main_rank_print(f"{'='*60}")
    
    # Check if we're in raw data mode
    raw_data_raw = self.config.data.get('raw_data', False)
    if isinstance(raw_data_raw, str):
        raw_data_mode = raw_data_raw.lower() in ['true', '1', 'yes', 'on']
    else:
        raw_data_mode = bool(raw_data_raw)
    
    if raw_data_mode:
        # RAW DATA MODE: Use dedicated function for on-the-fly processing
        gen_batch = _prepare_raw_data_batch_for_generation(self, batch)
    else:
        # PREPROCESSED MODE: Use dedicated function for pre-tokenized data
        gen_batch = _prepare_preprocessed_batch_for_generation(self, batch)
    
    return gen_batch

def get_safe_length(obj, obj_name="object"):
    """
    Safely get the length of an object that could be either DataProto or DataProtoItem.
    
    Args:
        obj: The object to get length for
        obj_name: Name of the object for logging purposes
        
    Returns:
        int: The length of the object
    """
    if hasattr(obj, '__len__'):
        return len(obj)
    else:
        # It's a DataProtoItem, we need to handle it differently
        if hasattr(obj, 'batch') and obj.batch is not None:
            # For DataProtoItem, we can get the batch size from the batch
            if hasattr(obj.batch, 'batch_size'):
                return obj.batch.batch_size[0]
        else:
                # Fallback: try to get length from the first tensor in batch
                for key, tensor in obj.batch.items():
                    if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
                        return tensor.shape[0]
        # If all else fails, assume it's a single item
        main_rank_print(f"WARNING: Could not determine length of {obj_name} (type: {type(obj)}), assuming length 1")
        return 1


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


def create_raw_data_datasets(config, tokenizer):
    """
    Create raw data datasets that simply load raw data files.
    Tokenization will happen on-the-fly in the trainer during batch processing.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    import pandas as pd
    
    # Import MAS prompt construction utilities
    from mas_r1_reasoner.agents.common import get_prompt, main_rank_print
    
    # Helper function to ensure file inputs are lists
    def ensure_file_list(file_input):
        """Ensure file_input is a list, handling both string and list cases"""
        from omegaconf import ListConfig
        if isinstance(file_input, str):
            return [file_input]
        elif isinstance(file_input, (list, ListConfig)):
            return list(file_input)  # Convert ListConfig to regular list
        else:
            raise ValueError(f"Expected string or list for file input, got {type(file_input)}")
    
    def _extract_question(prompt_data):
        """Extract question from various data formats"""
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            # DeepScaleR format: prompt is a list with [question, answer]
            question = prompt_data[0]
        elif isinstance(prompt_data, str):
            # String format: prompt is the question directly
            question = prompt_data
        elif hasattr(prompt_data, '__array__') and len(prompt_data) > 0:
            # Numpy array format: extract from first element
            first_item = prompt_data[0]
            if isinstance(first_item, dict) and 'content' in first_item:
                question = first_item['content']
            else:
                question = str(first_item)
        else:
            question = str(prompt_data)
        return question
    
    def process_raw_data_files(data_files, dataset_type):
        """Process raw data files and extract questions"""
        processed_samples = []
        
        # Track question occurrences to handle duplicates
        question_counter = {}
        duplicate_count = 0
        
        for file_path in data_files:
            main_rank_print(f"Processing raw data file: {file_path}")
            
            # Load raw data
            df = pd.read_parquet(file_path)
            main_rank_print(f"  - Loaded {len(df)} samples from {file_path}")
            
            for i, row in df.iterrows():
                if i % 1000 == 0:  # Progress every 1000 samples
                    main_rank_print(f"  - Processing sample {i+1}/{len(df)}...")
                
                try:
                    # Extract prompt data
                    prompt_data = row.get('prompt', None)
                    if prompt_data is None:
                        continue
                    
                    # TODO: The running one is still skip, you may want to resume if you continue the running 
                    # Replace empty ground_truth with "no answer" instead of skipping
                    # This ensures same dataset size for consistent RandomSampler behavior
                    # ------------------------------------------------------------
                    # Check if ground_truth exists and is not empty in reward_model field
                    reward_model = row.get('reward_model', None)
                    if reward_model is None or not isinstance(reward_model, dict):
                        main_rank_print(f"  - Sample {i+1}: reward_model is missing or invalid, using default ('no answer')")
                        reward_model = {'ground_truth': 'no answer'}
                    
                    ground_truth = reward_model.get('ground_truth', None)
                    if ground_truth is None or ground_truth == "" or (isinstance(ground_truth, str) and ground_truth.strip() == ""):
                        main_rank_print(f"  - Sample {i+1}: ground_truth is empty, replacing with 'no answer'")
                        question = _extract_question(prompt_data)
                        main_rank_print(f"  - Sample {i+1}: question: {question[:100]}...")
                        reward_model['ground_truth'] = 'no answer'
                    # ------------------------------------------------------------

                    # Extract question
                    question = _extract_question(prompt_data)
                    
                    # ------------------------------------------------------------
                    # HANDLE DUPLICATE QUESTIONS: Append unique identifier
                    # This prevents tensor reordering errors when duplicate questions exist
                    # The identifier is appended as a comment, preserving the original question
                    # ------------------------------------------------------------
                    original_question = question
                    if question in question_counter:
                        # This is a duplicate question
                        question_counter[question] += 1
                        occurrence_num = question_counter[question]
                        # Append a unique identifier as a comment at the end
                        # This makes the question unique while keeping it meaningful
                        question = f"{original_question}\n<!-- Duplicate #{occurrence_num}, Sample ID: {i} -->"
                        duplicate_count += 1
                        if occurrence_num == 1:
                            # Log only the first duplicate
                            if duplicate_count % 100 == 1:  # Log occasionally to avoid spam
                                main_rank_print(f"  - Found duplicate question (will make unique): {original_question[:80]}...")
                    else:
                        # First occurrence of this question
                        question_counter[question] = 0
                    # ------------------------------------------------------------
                    
                    # Create processed sample with raw data
                    processed_sample = {
                        'question': question,  # Modified question (unique if duplicate)
                        'raw_prompt': prompt_data,  # Keep original prompt data
                        'source_file': file_path,  # Track which file this sample came from
                    }
                    
                    # Preserve all original information
                    for key, value in row.items():
                        if key not in processed_sample:
                            processed_sample[key] = value
                    
                    processed_samples.append(processed_sample)
                    
                except Exception as e:
                    main_rank_print(f"  - Error processing sample {i}: {e}")
                    continue
        
        main_rank_print(f"✅ Processed {len(processed_samples)} samples for {dataset_type}")
        
        # Log duplicate statistics
        total_questions_with_duplicates = sum(1 for count in question_counter.values() if count > 0)
        if total_questions_with_duplicates > 0:
            main_rank_print(f"⚠️  Found {total_questions_with_duplicates} questions with duplicates (total {duplicate_count} duplicates)")
            main_rank_print(f"✅ All duplicate questions made unique by appending identifiers")
        
        return processed_samples
    
    # Ensure train_files and val_files are lists
    train_files = ensure_file_list(config.data.train_files)
    val_files = ensure_file_list(config.data.val_files)
    
    main_rank_print(f"✅ Training files: {train_files}")
    main_rank_print(f"✅ Validation files: {val_files}")
    
    # Process training data
    main_rank_print(f"Loading training data...")
    train_samples = process_raw_data_files(
        train_files, 
        "training"
    )
    
    # Process validation data
    main_rank_print(f"Loading validation data...")
    val_samples = process_raw_data_files(
        val_files, 
        "validation"
    )
    
    # Create DataFrames
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    
    # Add metadata
    train_df['dataset_type'] = 'mas_r1_raw_train'
    val_df['dataset_type'] = 'mas_r1_raw_val'
    train_df['stage'] = 'stage1'
    val_df['stage'] = 'stage1'
    train_df['processed_at'] = pd.Timestamp.now().isoformat()
    val_df['processed_at'] = pd.Timestamp.now().isoformat()
    
    # Create simple datasets that just hold the raw data
    # Note: These datasets provide placeholder tensors to satisfy VERL's DataProto requirements.
    # The actual tokenization and prompt construction happens on-the-fly in the trainer
    # during batch processing via prepare_batch_for_generation.
    class RawDataDataset:
        """
        Simple dataset that holds raw data for on-the-fly processing.
        
        Note: This dataset provides placeholder tensor fields with the correct names
        to satisfy VERL's DataProto requirements. These placeholder tensors will be 
        replaced with actual processed tensors during on-the-fly processing in 
        prepare_batch_for_generation.
        """
        
        def __init__(self, df):
            self.df = df
            self._data = df
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            # Get the raw data
            raw_data = self.df.iloc[idx].to_dict()
            
            # Add placeholder tensor fields with the correct names to satisfy VERL's DataProto requirements
            # These will be replaced during on-the-fly processing in prepare_batch_for_generation
            import torch
            
            # Create placeholder tensors with minimal size
            # VERL requires at least one tensor field to satisfy the assertion "tensors must not be empty"
            # Use the actual tensor names that will be used later for consistency
            placeholder_tensor = torch.tensor([0], dtype=torch.long)
            
            # Return a dict with both raw data and placeholder tensors
            return {
                **raw_data,  # All the original raw data
                'input_ids': placeholder_tensor,  # Placeholder tensor field with correct name
                'attention_mask': placeholder_tensor,  # Placeholder tensor field with correct name
                'position_ids': placeholder_tensor,  # Placeholder tensor field with correct name
            }
    
    # Create training and validation datasets
    train_dataset = RawDataDataset(train_df)
    val_dataset = RawDataDataset(val_df)
    
    main_rank_print(f"✅ Created raw data datasets")
    main_rank_print(f"✅ Training samples: {len(train_samples)}")
    main_rank_print(f"✅ Validation samples: {len(val_samples)}")
    main_rank_print(f"✅ Data will be tokenized on-the-fly during training")
    
    return train_dataset, val_dataset


def create_preprocessed_datasets(config, tokenizer):
    """
    Create preprocessed datasets using PreprocessedRLDataset for pre-tokenized data.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer instance
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
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
    
    main_rank_print(f"✅ Created PreprocessedRLDataset instances for preprocessed data")
    main_rank_print(f"✅ Using pre-tokenized data for faster training")
    
    return train_dataset, val_dataset


def create_mas_r1_dataloaders(config, tokenizer):
    """
    Create MAS-R1 specific dataloaders with support for two modes:
    - raw_data=False: Use PreprocessedRLDataset for preprocessed data (current behavior)
    - raw_data=True: Use RLHFDataset for raw data preprocessing on-the-fly
    
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
    
    # Check if raw_data mode is enabled
    raw_data_raw = config.data.get('raw_data', False)
    
    # Convert string values to boolean if needed (common issue with YAML configs)
    if isinstance(raw_data_raw, str):
        raw_data_mode = raw_data_raw.lower() in ['true', '1', 'yes', 'on']
    else:
        raw_data_mode = bool(raw_data_raw)
    
    main_rank_print(f"\n{'='*60}")
    if raw_data_mode:
        main_rank_print("CREATING RAW DATA DATALOADERS WITH RLHFDataset")
        main_rank_print("Using VERL-style on-the-fly preprocessing")
    else:
        main_rank_print("CREATING PREPROCESSED DATALOADERS WITH STATEFULDATALOADER")
        main_rank_print("Using PreprocessedRLDataset with StatefulDataLoader for VERL compatibility")
    main_rank_print(f"{'='*60}\n")
    
    if raw_data_mode:
        # RAW DATA MODE: Use RLHFDataset for on-the-fly preprocessing
        train_dataset, val_dataset = create_raw_data_datasets(config, tokenizer)
    else:
        # PREPROCESSED MODE: Use PreprocessedRLDataset (current behavior)
        train_dataset, val_dataset = create_preprocessed_datasets(config, tokenizer)
    
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
    
    if raw_data_mode:
        main_rank_print(f"✅ Created RLHFDataset instances with VERL-style preprocessing")
        main_rank_print(f"✅ Data will be tokenized on-the-fly during training")
        main_rank_print(f"✅ Supports dynamic prompt processing and chat templates")
    else:
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
        
        # Choose between async and process AgentSystem based on multiply_processes setting
        multiply_processes = get_global("global_multiply_processes")
        
        if multiply_processes == 0:
            # Use async execution
            # raise ValueError("Async execution is not supported")
            main_rank_print(f"Initializing ASYNC AgentSystem...")
            trainer_instance.agent_system = AsyncAgentSystem(agent_config)
            main_rank_print(f"✓ Async AgentSystem initialized with agent configuration")
        else:
            raise ValueError("Process execution is not supported")
            # Use process execution
            # main_rank_print(f"Initializing PROCESS AgentSystem (multiply_processes={multiply_processes})...")
            # trainer_instance.agent_system = ProcessAgentSystem(agent_config)
            # main_rank_print(f"✓ Process AgentSystem initialized with agent configuration")

    else:
        main_rank_print(f"AgentSystem already initialized")


def get_bool_config(config, config_path, default_value=False, required=False):
    """
    Helper function to safely extract boolean configuration values from nested config paths.
    
    This function handles both string and boolean values, converting strings like 'true', '1', 'yes', 'on'
    to boolean True, and other values to boolean False.
    
    Args:
        config: Configuration object (e.g., OmegaConf config)
        config_path: String path to the config value (e.g., 'azr.mas_r1.diff_based_reward')
        default_value: Default value to return if config path doesn't exist and not required
        required: If True, raises RuntimeError when config path doesn't exist
        
    Returns:
        bool: The boolean configuration value
        
    Raises:
        RuntimeError: If required=True and config path doesn't exist
    """
    try:
        # Split the config path into parts
        path_parts = config_path.split('.')
        current_config = config
        
        # Navigate through the config path
        for part in path_parts:
            if hasattr(current_config, part):
                current_config = getattr(current_config, part)
            else:
                if required:
                    raise RuntimeError(f"Could not find config.{config_path}. This configuration is required.")
                else:
                    return default_value
        
        # Convert the value to boolean
        if isinstance(current_config, str):
            return current_config.lower() in ['true', '1', 'yes', 'on']
        else:
            return bool(current_config)
            
    except Exception as e:
        if required:
            raise RuntimeError(f"Failed to read config.{config_path}: {e}")
        else:
            return default_value


def save_building_blocks_evaluation_results(questions, batch, batch_building_block_outputs, config=None):
    """
    Save building block results to CSV files and upload to wandb.
    
    Args:
        questions: List of questions
        batch: DataProto batch containing ground truth data
        batch_building_block_outputs: Raw building block execution outputs
        config: Configuration object containing val_files information
    """
    import pandas as pd
    import os
    from mas_r1_reasoner.agents.logging_utils.stdout import PrettyPrinter as pp
    
    pp.section_header("Building Blocks Evaluation Results")
    pp.status("Processing", f"Processing {len(questions)} questions with building blocks results", "info")
    
    # Collect ground truth from batch
    ground_truths = []
    for i in range(len(questions)):
        # Try to get ground truth from reward_model field
        try:
            if hasattr(batch, 'non_tensor_batch') and 'reward_model' in batch.non_tensor_batch:
                ground_truth = batch.non_tensor_batch['reward_model'][i].get('ground_truth') if i < len(batch.non_tensor_batch['reward_model']) else None
            else:
                ground_truth = None
        except:
            ground_truth = None
        ground_truths.append(ground_truth)
    
    # Prepare CSV data - create 4 separate CSV files (using direct format)
    pp.status("Saving Results", "Preparing 4 separate CSV files", "info")
    
    # Get building blocks from global configuration instead of hardcoding
    building_blocks = get_global("global_init_archive")
    
    # Map building block names for CSV filenames and display
    name_mapping = {
        'COT': 'CoT',
        'COT_THINK': 'CoT_Think',
        'COT_SC': 'CoT_SC', 
        'LLM_debate': 'Debate',
        'Reflexion': 'Refine',
        'WebSearch': 'WebSearch'
    }

    # Extract path components from val_files for directory structure
    # Example: data/igsm/depth/test_depth2.parquet -> dataset=igsm, dimension=depth, value=depth2
    base_save_path = "/export/xgen-finance/meta_agent/mas_r1/post_process/rl_analysis/eval/building_blocks"
    dataset_name = ""
    dimension_value = ""
    
    if config is not None and hasattr(config, 'data') and hasattr(config.data, 'val_files'):
        val_files = config.data.val_files
        # Handle list or single file
        val_file = val_files[0] if isinstance(val_files, list) else val_files
        
        # Parse val_file path to extract dataset, dimension, and value
        # Example: data/igsm/depth/test_depth2.parquet
        import re
        from pathlib import Path
        
        path_parts = Path(val_file).parts
        
        # Extract dataset name (e.g., 'igsm')
        if len(path_parts) >= 2:
            dataset_name = path_parts[1]  # Assuming format: data/{dataset_name}/...
        
        # Extract dimension value from subdirectory + filename (preserve full structure)
        # Example: data/igsm/parallel/test_d2_n20_domains.parquet -> parallel/test_d2_n20_domains
        filename = Path(val_file).stem  # Remove extension
        
        # Get subdirectory (e.g., 'parallel', 'horizon')
        subdirectory = path_parts[2] if len(path_parts) >= 3 else ""
        
        # Combine subdirectory and filename to preserve full path structure
        dimension_value = f"{subdirectory}/{filename}" if subdirectory else filename
        
        pp.status("Dataset", f"Extracted dataset: {dataset_name}, dimension_value: {dimension_value}", "info")
    
    # Get agent model name from config if available
    model_name = ""
    if config is not None and hasattr(config, 'azr') and hasattr(config.azr, 'mas_r1') and hasattr(config.azr.mas_r1, 'agent') and hasattr(config.azr.mas_r1.agent, 'model_name'):
        model_name = config.azr.mas_r1.agent.model_name
        # Extract simple model name (e.g., gpt_oss_120b from path)
        model_name_simple = model_name.split('/')[-1] if '/' in model_name else model_name
        pp.status("Agent", f"Using model name: {model_name_simple}", "info")
    else:
        model_name_simple = "unknown_model"
    
    # Get reasoning_effort from config if available
    reasoning_effort = ""
    if config is not None and hasattr(config, 'azr') and hasattr(config.azr, 'mas_r1') and hasattr(config.azr.mas_r1, 'agent') and hasattr(config.azr.mas_r1.agent, 'reasoning_effort'):
        reasoning_effort = config.azr.mas_r1.agent.reasoning_effort
        pp.status("Reasoning Effort", f"Using reasoning_effort: {reasoning_effort}", "info")
    else:
        reasoning_effort = "medium"
    
    # Construct save directory path
    # Format: /export/xgen-finance/meta_agent/mas_r1/post_process/rl_analysis/eval/building_blocks/{model_name}_{reasoning_effort}/{dataset_name}/{dimension_value}/
    save_dir = os.path.join(
        base_save_path,
        f"{model_name_simple}_{reasoning_effort}",
        dataset_name,
        dimension_value
    )
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    pp.status("Save Directory", f"Created/verified directory: {save_dir}", "success")
    
    csv_files_created = []
    all_csv_rows = []  # For wandb upload
    
    # Create separate CSV file for each building block
    for block_name in building_blocks:
        block_rows = []
        mapped_name = name_mapping.get(block_name)
        
        for i, block_outputs in enumerate(batch_building_block_outputs):
            question = questions[i] if i < len(questions) else "Unknown"
            ground_truth = ground_truths[i] if i < len(ground_truths) else None
            block_output = block_outputs.get(block_name)
            
            csv_row = {
                'step': 0,
                'question': question,
                'execution_output': block_output,
                'ground_truth': ground_truth
            }
            block_rows.append(csv_row)
            
            # Also keep for wandb (add mas_code for wandb compatibility)
            all_csv_rows.append({
                'mas_code': mapped_name,
                'step': 0,
                'question': question,
                'execution_output': block_output,
                'ground_truth': ground_truth
            })
        
        # Save separate CSV file for this building block
        # Format: mas_r1_cot_evaluation.csv (with appropriate building block name)
        filename = f"mas_r1_{mapped_name.lower().replace('-', '_')}_evaluation.csv"
        output_file = os.path.join(save_dir, filename)
        
        df = pd.DataFrame(block_rows)
        df.to_csv(output_file, index=False)
        csv_files_created.append(output_file)
        
        pp.status("CSV", f"Saved {mapped_name}: {output_file} ({len(block_rows)} rows)", "success")
    
    # Print summary
    pp.status("Completed", f"Evaluation completed successfully", "success")
    pp.status("Output", f"Created {len(csv_files_created)} CSV files:", "success")
    for csv_file in csv_files_created:
        pp.status("  ", f"- {csv_file}", "info")
    pp.status("Summary", f"{len(questions)} questions × {len(building_blocks)} building blocks", "info")
    
    # Print sample of results
    sample_results = []
    for block in name_mapping.values():
        block_rows = [r for r in all_csv_rows if r['mas_code'] == block]
        if block_rows:
            sample_results.append([block, len(block_rows), "✅" if any(r['execution_output'] != "Unavailable" for r in block_rows) else "❌"])
    
    pp.table(["Building Block", "Total Evaluations", "Has Valid Outputs"], sample_results, "Evaluation Summary")
    
    # Upload 4 separate tables to wandb (one for each building block)
    try:
        import wandb
        if wandb.run is not None:
            pp.status("Wandb Upload", "Uploading building blocks tables to wandb", "info")
            
            # Create separate tables for each building block
            for block_name in name_mapping.values():
                # Filter rows for this building block
                block_rows = [r for r in all_csv_rows if r['mas_code'] == block_name]
                
                if block_rows:
                    # Create wandb table for this building block
                    columns = ["step", "question", "execution_output", "ground_truth"]
                    data = []
                    
                    for row in block_rows:
                        # Use full text without truncation
                        execution_output = row['execution_output']
                        question = row['question']
                            
                        data.append([
                            row['step'],
                            question,
                            execution_output,
                            str(row['ground_truth'])
                        ])
                    
                    # Create wandb table
                    table = wandb.Table(columns=columns, data=data)
                    
                    # Log table to wandb with descriptive name
                    table_name = f"building_blocks_evaluation/{block_name.lower().replace('-', '_')}_results"
                    wandb.log({table_name: table})
                    
                    pp.status("Wandb", f"Uploaded {block_name} table ({len(block_rows)} rows)", "success")
                
        else:
            pp.status("Wandb", "No active wandb run - skipping table upload", "warning")
            
    except ImportError:
        pp.status("Wandb", "wandb not installed - skipping table upload", "warning")
    except Exception as e:
        pp.status("Wandb", f"Failed to upload tables: {e}", "error")

