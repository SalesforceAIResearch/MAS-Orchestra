import json
import numpy as np
from datetime import datetime
import re

# Import main_rank_print and extract_xml from common.py to avoid circular import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from mas_r1_reasoner.agents.common import main_rank_print, extract_xml

# Note: The main logger (ReasonRLTracking) handles wandb logging automatically
# No need to call wandb.log() directly here to avoid step synchronization issues
# If wandb logging is needed, it should be done through the main logger in the training loop
# Note: these will only be logged to console, not wandb (since it is not in the training loop)

def extract_code_from_response(response: str, validate_python_code, logger, max_retries: int = 3):
    """
    Extract code from the model response using XML format with sanity checks.
    The response is expected to be XML with <code>, <name>, and <thought> tags.
    Args:
        response: The raw response from the model
        validate_python_code: function to validate python code
        logger: REQUIRED logger for wandb logging (must have wandb backend)
        max_retries: Maximum number of regeneration attempts (not used - kept for compatibility)
    Returns:
        Tuple of (code, name, thought) - returns (None, None, None) if extraction fails
    """
    # main_rank_print(f"\n{'='*50}")
    # main_rank_print("EXTRACTING CODE FROM RESPONSE (XML MODE WITH SANITY CHECKS)")
    # main_rank_print(f"{'='*50}")
    # main_rank_print(f"Response length: {len(response)} characters")
    # main_rank_print(f"Response type: {type(response)}")
    # main_rank_print(f"Response (str): {response}")
    
    try:
        # Step 1: Extract XML tags
        # main_rank_print(f"Step 1: Extracting XML tags...")
        
        # Extract code, name, and thought using extract_xml function
        code = extract_xml(response, "code")
        name = extract_xml(response, "name")
        thought = extract_xml(response, "thought")

        # main_rank_print(f"Extracted code: {code}")
        # main_rank_print(f"Extracted thought: {thought}")
        # main_rank_print(f"Extracted name: {name}")
        # main_rank_print(f"Extracted thought length: {len(thought)} characters")
        # main_rank_print(f"Extracted code length: {len(code)} characters")

        # Step 2: Check if code was found
        # main_rank_print(f"Step 2: Checking if code was found...")
        if not code or not code.strip():
            main_rank_print(f"✗ No code found in XML response.  Response: {response}, Code: {code}")
            # main_rank_print(f"Returning (None, None, None) for trainer to handle")
            return f"✗ No code found in XML response, Code: {code}", name, thought
        
        # main_rank_print(f"✓ Code found in XML response")
        
        # Step 3: Check if code is not empty
        # main_rank_print(f"Step 3: Checking code is not empty...")
        if not code.strip():
            main_rank_print(f"✗ Code is empty or whitespace only. Response: {response}, Code: {code}")
            # main_rank_print(f"Returning (None, None, None) for trainer to handle")
            return f"✗ Code is empty or whitespace only, Code: {code}", name, thought
        
        # main_rank_print(f"✓ Code is not empty")
        
        # Step 4: Validate Python syntax
        # main_rank_print(f"Step 4: Validating Python syntax...")
        if not validate_python_code(code.strip(), logger):
            main_rank_print(f"✗ Python syntax is invalid. Response: {response}, Code: {code}")
            # main_rank_print(f"Returning (None, None, None) for trainer to handle")
            return f"✗ Python syntax is invalid, Code: {code}", name, thought
        
        # main_rank_print(f"✓ Python syntax is valid")
        
        # All checks passed!
        # main_rank_print(f"\n{'='*50}")
        # main_rank_print("ALL SANITY CHECKS PASSED!")
        # main_rank_print(f"{'='*50}")
        # main_rank_print(f"Successfully extracted valid Python code from XML response.")
        # main_rank_print(f"Code preview: {code[:100]}...")
        # main_rank_print(f"Code length: {len(code)} characters")
        # main_rank_print(f"Name: {name}")
        # main_rank_print(f"Thought preview: {thought[:100]}...")
        # main_rank_print(f"{'='*50}\n")
        return code, name, thought
        
    except Exception as e:
        main_rank_print(f"Error during code extraction: {e}")
        # main_rank_print(f"Returning (None, None, None) for trainer to handle")
        # return None, None, None
        return f'Error {e}. Response: {response}', f'Error {e}. Response: {response}', f'Error {e}. Response: {response}'


def validate_python_code(code: str, logger) -> bool:
    """
    Validate Python code syntax using ast.parse
    Args:
        code: Python code string to validate
    Returns:
        True if code has valid Python syntax, False otherwise
    """
    try:
        import ast
        import re
        from mas_r1_reasoner.agents.shared_vars import get_global
        
        # main_rank_print(f"Validating Python code: {code}")
        if not isinstance(code, str):
            return False
        
        # Check if add_judge is enabled and this might be a building block reference
        add_judge = get_global("global_add_judge")
        
        if add_judge:
            # Check if code matches building block reference pattern (e.g., "CoT:10", "cot_sc:42", "debate:123")
            building_blocks = ['CoT', 'CoT_SC', 'Debate', 'Refine']
            for block in building_blocks:
                pattern = rf'^{re.escape(block)}\s*:\s*(.+)$'
                if re.match(pattern, code.strip(), re.DOTALL | re.IGNORECASE):
                    # This is a building block reference, consider it valid
                    # main_rank_print(f"ADD_JUDGE: Allowing building block reference: {code}")
                    return True
        
        # Regular Python syntax validation
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False