"""
GSM8K Dataset Processor for MAS-R1 Code Generation Training

This processor handles GSM8K data for MAS-R1 training, specifically for code generation and execution.
It removes the incorrect "Let's think step by step and output the final answer after \"####\"" instruction
that was added by the wrong preprocessing script, ensuring clean questions are used for code generation.
"""
import copy
from typing import Dict
from verl.protocol import DataProto
from mas_r1_reasoner.data_precessor.BaseDatasetProcessor import BaseDatasetProcessor
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code
from mas_r1_reasoner.agents.common import get_prompt, main_rank_print
from mas_r1_reasoner.agents.agent_system import AgentSystem, LLMAgentBase, Info
from mas_r1_reasoner.agents.shared_vars import set_global, get_global
import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from omegaconf import OmegaConf


class MathDatasetProcessor(BaseDatasetProcessor):
    """Processor for GSM8K dataset"""
    
    
    def extract_math_question(self, batch: DataProto, index: int) -> str:
        """Extract question from GSM8K data format"""
        try:
            # Comprehensive debugging of the DataProto object
            # main_rank_print(f"\n{'='*80}")
            # main_rank_print(f"COMPREHENSIVE DATAPROTO DEBUG FOR INDEX {index}")
            # main_rank_print(f"{'='*80}")
            
            # Debug the entire DataProto structure
            # main_rank_print(f"DataProto type: {type(batch)}")
            # main_rank_print(f"DataProto batch keys: {list(batch.batch.keys())}")
            # main_rank_print(f"DataProto non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
            
            # Debug non_tensor_batch content
            # main_rank_print(f"\nNON_TENSOR_BATCH DETAILS:")
            # main_rank_print(f"non_tensor_batch type: {type(batch.non_tensor_batch)}")
            # main_rank_print(f"non_tensor_batch length: {len(batch.non_tensor_batch)}")
            # main_rank_print(f"non_tensor_batch content: {batch.non_tensor_batch}")
            
            # Debug batch content
            # main_rank_print(f"\nBATCH DETAILS:")
            # for key, value in batch.batch.items():
            #     main_rank_print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
            

            # Extract question directly from the batch's non_tensor_batch
            # main_rank_print(f"\nEXTRACTING QUESTION FOR INDEX {index}:")
            
            # Get the question array directly from the batch
            question_array = batch.non_tensor_batch.get('question', [])
            # main_rank_print(f"question_array type: {type(question_array)}")
            # main_rank_print(f"question_array length: {len(question_array) if hasattr(question_array, '__len__') else 'N/A'}")
            # main_rank_print(f"question_array content: {repr(question_array)}")
            
            # Check if we have a valid array and index is within bounds
            if not hasattr(question_array, '__len__'):
                main_rank_print(f"✗ Question data is not an array: type={type(question_array)}")
                raise ValueError(f"Question data is not an array: {type(question_array)}")
            
            if index >= len(question_array):
                main_rank_print(f"✗ Index {index} out of bounds for question array of length {len(question_array)}")
                raise ValueError(f"Index {index} out of bounds for question array")
            
            # Extract the specific question for this index
            question = question_array[index]
            
            # Remove the "Let's think step by step and output the final answer after \"####\"" instruction
            # This instruction was incorrectly added by the wrong preprocessing script
            instruction_to_remove = "Let's think step by step and output the final answer after \"####\"."
            if instruction_to_remove in question:
                question = question.replace(instruction_to_remove, "").strip()
                main_rank_print(f"✓ Removed instruction from question: {question[:100]}{'...' if len(question) > 100 else ''}")
            
            # Validate the extracted question
            if isinstance(question, str) and question.strip():
                # main_rank_print(f"✓ Extracted question: {question[:100]}{'...' if len(question) > 100 else ''} from Index {index}")
                return question
            else:
                main_rank_print(f"✗ Extracted question is not a valid string: type={type(question)}, value={repr(question)}")
                raise ValueError(f"Extracted question is not a valid string: {type(question)}")
            
        except Exception as e:
            main_rank_print(f"ERROR extracting GSM8K question for sample {index}: {e}")
            raise RuntimeError(f"Failed to extract question from GSM8K data for sample {index}: {e}")
    
