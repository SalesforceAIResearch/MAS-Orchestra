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
Harmony parser module - routes to appropriate parser based on problem type.
"""

from typing import Tuple
from mas_r1_reasoner.agents.shared_vars import get_global


def extract_harmony_code_from_response(response_text: str, validate_python_code, logger) -> Tuple[str, str, str]:
    """
    Extract code from harmony response, routing to the appropriate parser based on problem type.
    
    Args:
        response_text: The harmony response text
        validate_python_code: Function to validate Python code
        logger: Logger instance
        
    Returns:
        Tuple of (code, name, thought)
    """
    problem_type = get_global('global_problem_type')
    
    # Check if we should use IGSM prompt
    use_igsm_prompt = get_global("global_use_igsm_prompt")
    if use_igsm_prompt is None:
        use_igsm_prompt = False
    
    if problem_type == 'harmony_minimal':
        # Use minimal parser for harmony_minimal (0-1 agents)
        from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import extract_harmony_code_from_response as minimal_extract
        return minimal_extract(response_text, validate_python_code, logger)
    elif problem_type == 'harmony_medium':
        # Use medium parser for harmony_medium (1+ agents with graph)
        # If use_igsm_prompt is enabled, use the IGSM-specific parser
        if use_igsm_prompt:
            from mas_r1_reasoner.rewards.utils.harmony_parser.medium_igsm import extract_harmony_code_from_response as medium_igsm_extract
            print("Using IGSM parser")
            return medium_igsm_extract(response_text, validate_python_code, logger)
        else:
            from mas_r1_reasoner.rewards.utils.harmony_parser.medium import extract_harmony_code_from_response as medium_extract
            return medium_extract(response_text, validate_python_code, logger)
    else:
        # Default to minimal parser for backward compatibility
        from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import extract_harmony_code_from_response as minimal_extract
        return minimal_extract(response_text, validate_python_code, logger)


__all__ = ['extract_harmony_code_from_response']

