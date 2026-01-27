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
from mas_r1_reasoner.agents.common import extract_xml
from mas_r1_reasoner.agents.blocks.cot import COT
from mas_r1_reasoner.agents.blocks.cot_sc import COT_SC
from mas_r1_reasoner.agents.blocks.llm_debate import LLM_debate
from mas_r1_reasoner.agents.blocks.reflexion import Reflexion
import re

def _replace_building_block_reference_with_code(extracted_code: str) -> str:
    """
    When add_judge=True, replace building block references like 'CoT:10' with actual building block code.
    
    Args:
        extracted_code: The extracted code string
        
    Returns:
        Either the original code or the building block implementation code
    """
    from mas_r1_reasoner.agents.shared_vars import get_global
    
    # Check if add_judge is enabled
    add_judge = get_global("global_add_judge")
    if not add_judge:
        return extracted_code
    
    # Building blocks mapping
    building_blocks_map = {
        'CoT': COT,
        'CoT_SC': COT_SC, 
        'Debate': LLM_debate,
        'Refine': Reflexion
    }
    
    # Check if extracted_code matches building block reference pattern
    if not extracted_code or not extracted_code.strip():
        return extracted_code
    
    for block_name, block_def in building_blocks_map.items():
        # Pattern matches: "CoT:10", "cot_sc:42", "DEBATE:3.14", etc. (case insensitive)
        pattern = rf'^{re.escape(block_name)}\s*:\s*(.+)$'
        match = re.search(pattern, extracted_code.strip(), re.DOTALL | re.IGNORECASE)
        
        if match:
            # Replace with actual building block code
            building_block_code = block_def.get('code', '')
            main_rank_print(f"  🔄 Replacing building block reference '{extracted_code.strip()}' with {block_name} implementation code")
            return building_block_code
    
    # No building block pattern found, return original code
    return extracted_code

def extract_codes(trainer_instance, mas_code_generation_output: DataProto, question_results: Dict, is_validation: bool = False) -> List[str]:
    """
    Extract codes from the first-level generation output.
    
    This function processes the responses from the first-level generation to identify
    extracted codes that will be used for the second-level generation.
    
    Instead of complex index mapping, this function directly extracts the question
    from each response using the same method as extract_questions_and_ground_truth.
    This ensures accurate question-code mapping regardless of response ordering.
    
    Args:
        trainer_instance: The trainer instance with tokenizer
        mas_code_generation_output: Output from first-level generation
        question_results: Dictionary containing question information
    
    Returns:
        List of extracted codes
        
    Note:
        Codes will be used to generate new responses in the second layer of the hierarchical generation.
    """
    
    codes = []
    
    try:
        # Get the response text from batch['responses'] using tokenizer
        response_texts = []
        for generation_idx in range(len(mas_code_generation_output.batch['responses'])):
            response_text = trainer_instance.tokenizer.decode(
                mas_code_generation_output.batch['responses'][generation_idx], 
                skip_special_tokens=True
            )
            response_texts.append(response_text)
        
        if not response_texts:
            error_msg = "No response texts found in mas_code_generation_output.batch['responses']"
            main_rank_print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        main_rank_print(f"📝 Extracting codes from {len(response_texts)} response texts...")
        
        # Check if mock_sub_task_sub_agent is enabled
        mock_sub_task_sub_agent = getattr(trainer_instance.config.azr.mas_r1, 'mock_sub_task_sub_agent', False)
        
        if mock_sub_task_sub_agent:
            main_rank_print(f"🔧 MOCK MODE ENABLED: Will use 'DEBUG' values for all codes")
        
        # Process all responses using the same logic regardless of validation/training mode
        for generation_idx in range(len(response_texts)):
            response_text = response_texts[generation_idx]
            
            # Check if mock mode is enabled
            if mock_sub_task_sub_agent:
                extracted_code = 'DEBUG'
            else:
                # Extract code using extract_code_from_response
                try:
                    code, name, thought = extract_code_from_response(
                        response_text,
                        validate_python_code,
                        trainer_instance.logger if hasattr(trainer_instance, 'logger') and trainer_instance.logger is not None else None
                    )
                    extracted_code = code if code is not None else f'Invalid code {generation_idx}'
                except Exception as e:
                    main_rank_print(f"Code extraction failed for response {generation_idx}: {e}")
                    extracted_code = f'Invalid code {generation_idx}'
                
                # #TODO: debug
                # extracted_code = 'CoT_SC:10'

                # Replace building block references with actual building block code when add_judge=True
                extracted_code = _replace_building_block_reference_with_code(extracted_code)
            
            codes.append(extracted_code)
            
            # Extract question directly from the original input data at this position
            try:
                question = trainer_instance.processor.extract_math_question(mas_code_generation_output, generation_idx)
                
                # Store codes mapping for this question using question only as key
                question_response_key = question
                
                if question_response_key not in question_results:
                    question_results[question_response_key] = {}
                if 'sub_tasks_mapping' not in question_results[question_response_key]:
                    question_results[question_response_key]['sub_tasks_mapping'] = {}
                
                # Use extracted code as the key to handle non-unique codes
                if extracted_code not in question_results[question_response_key]['sub_tasks_mapping']:
                    question_results[question_response_key]['sub_tasks_mapping'][extracted_code] = []
                
                # Add this (generation_idx, response_text) pair to the list for this code
                question_results[question_response_key]['sub_tasks_mapping'][extracted_code].append({
                    'response_idx': generation_idx,
                    'response_text': response_text
                })
                
                main_rank_print(f"✅ Response {generation_idx}: Extracted code for question: {question[:100]}{'...' if len(question) > 100 else ''}")
                
            except Exception as e:
                raise RuntimeError(f"⚠️ Warning: Could not extract question for response {generation_idx}: {e}")

        # Each Level 1 response should generate exactly one Level 2 response
        main_rank_print(f"✅ Extracted {len(codes)} codes")
        
        return codes
        
    except Exception as e:
        error_msg = f"Failed to extract codes: {e}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
