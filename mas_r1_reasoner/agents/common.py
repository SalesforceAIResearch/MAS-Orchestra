import json
import string
import sys
import os
from pathlib import Path

# Add the project root to the path to find blocks and prompts modules
# Use a more robust method to find the project root
current_file = Path(__file__).resolve()
# Go up from utils/common.py to mas_r1
project_root = current_file.parent.parent.parent
# Also try alternative path resolution
if not project_root.exists() or not (project_root / 'prompts').exists():
    # Try going up from current working directory
    cwd = Path.cwd()
    if (cwd / 'prompts').exists():
        project_root = cwd
    elif (cwd.parent / 'prompts').exists():
        project_root = cwd.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mas_r1_reasoner.agents.blocks.cot import COT
from mas_r1_reasoner.agents.blocks.cot_sc import COT_SC
from mas_r1_reasoner.agents.blocks.llm_debate import LLM_debate
from mas_r1_reasoner.agents.blocks.reflexion import Reflexion
from mas_r1_reasoner.agents.blocks_harmony.cot import COT as COT_HARMONY
from mas_r1_reasoner.agents.blocks_harmony.cot_sc import COT_SC as COT_SC_HARMONY
from mas_r1_reasoner.agents.blocks_harmony.llm_debate import LLM_debate as LLM_debate_HARMONY
from mas_r1_reasoner.agents.blocks_harmony.reflexion import Reflexion as Reflexion_HARMONY
from mas_r1_reasoner.agents.blocks_harmony.web_search import WebSearch as WebSearch_HARMONY
from mas_r1_reasoner.agents.blocks_harmony.igsm.cot import COT as COT_HARMONY_IGSM
from mas_r1_reasoner.agents.blocks_harmony.igsm.cot_think import COT as COT_THINK_HARMONY_IGSM
from mas_r1_reasoner.agents.blocks_harmony.igsm.cot_sc import COT_SC as COT_SC_HARMONY_IGSM
from mas_r1_reasoner.agents.blocks_harmony.igsm.llm_debate import LLM_debate as LLM_debate_HARMONY_IGSM
from mas_r1_reasoner.agents.blocks_harmony.igsm.reflexion import Reflexion as Reflexion_HARMONY_IGSM
import copy
import random
import numpy as np
import re
from typing import Any, Optional, Dict, List
import torch
from verl import DataProto
from mas_r1_reasoner.agents.shared_vars import get_global



EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

# Helper function to check if this is the main rank
def is_main_rank():
    """Check if this is the main rank (rank 0) for distributed training"""
    try:
        import torch.distributed
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        else:
            # If distributed is not initialized, assume we're the main process
            return True
    except:
        # If any error occurs, assume we're the main process
        return True

def main_rank_print(*args, **kwargs):
    """Print only on the main rank to avoid duplicate output"""
    if is_main_rank():
        print(*args, **kwargs)


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses.
    Made flexible to handle truncated/partial responses.

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    # Try multiple patterns in order of preference
    patterns = [
        # Full XML
        rf'<{tag}>\s*(.*?)\s*</{tag}>',
        # Partial tag until next tag
        rf'<{tag}>\s*(.*?)(?=\s*<[^>]+>)',
        # Partial tag until end of text
        rf'<{tag}>\s*(.*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            return content
    
    return ""
    
def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id

def get_init_archive(blocks):
    # Check if we should use IGSM blocks
    use_igsm_prompt = get_global("global_use_igsm_prompt")
    if use_igsm_prompt is None:
        use_igsm_prompt = False
    
    if use_igsm_prompt:
        # Use IGSM harmony blocks
        block_map = {
            'COT': COT_HARMONY_IGSM,
            'COT_THINK': COT_THINK_HARMONY_IGSM,
            'COT_SC': COT_SC_HARMONY_IGSM,
            'Reflexion': Reflexion_HARMONY_IGSM,
            'LLM_debate': LLM_debate_HARMONY_IGSM,
            'WebSearch': WebSearch_HARMONY  # Please holder. No WebSearch in IGSM
        }
    else:
        # Use regular harmony blocks
        block_map = {
            'COT': COT_HARMONY,
            'COT_SC': COT_SC_HARMONY,
            'Reflexion': Reflexion_HARMONY,
            'LLM_debate': LLM_debate_HARMONY,
            'WebSearch': WebSearch_HARMONY
        }

    return [copy.deepcopy(block_map[block]) for block in blocks] # it may be the same architecture, copy to avpod cross modification


def get_prompt(question, sub_task=None, level=1): 

    known_prompt = get_global("global_known_prompt")

    assert known_prompt is None, "known_prompt is set. Please use get_known_prompt instead."

    
    # Get archive from global variable, with fallback to default
    init_archive = get_global("global_init_archive")
    if init_archive is None:
        raise ValueError("global_init_archive is not set")
    
    archive = get_init_archive(init_archive) 


    archive_str = ",\n".join([json.dumps(sol) for sol in archive])
    archive_str = f"[{archive_str}]"

    # Check if decompose_only or architecture_only is set and import appropriate prompt
    decompose_only = get_global("global_decompose_only")
    architecture_only = get_global("global_architecture_only")
    architecture_only_sequential = get_global("global_architecture_only_sequential")
    enable_tree_architecture = get_global("global_enable_tree_architecture")
    include_blocks = get_global("global_include_blocks")
    add_judge = get_global("global_add_judge")
    problem_type = get_global("global_problem_type")
    node_model = get_global("global_node_model") 
    eval_building_blocks = get_global("global_eval_building_blocks")
    no_decompose = get_global("global_no_decompose")
    dataset_name = get_global("global_dataset_name")
    use_igsm_prompt = get_global("global_use_igsm_prompt")

    if problem_type == 'direct':
        if use_igsm_prompt:
            from mas_r1_reasoner.agents.prompts.propose_direct_igsm import base
        else:
            from mas_r1_reasoner.agents.prompts.propose_direct import base
         #placeholder. There is no example for direct prompt
        EXAMPLE = ''
    elif problem_type == 'mcp':
        from mas_r1_reasoner.agents.prompts.propose_mcp import base
        EXAMPLE = ''
    elif problem_type == 'harmony_minimal':
        if no_decompose:
            from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_minmal_no_decompose import base
        else:
            if dataset_name == 'multiple_choice':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_minmal_mcq import base
            else:
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_minmal import base
        EXAMPLE = ''
    elif problem_type == 'harmony_medium':
        if use_igsm_prompt:
            # Get the IGSM variant (breadth, depth, horizon, parallel, or combine)
            igsm_variant = get_global("global_igsm_variant")
            if igsm_variant == 'breadth':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium_igsm_breadth import base
            elif igsm_variant == 'depth':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium_igsm_depth import base
            elif igsm_variant == 'horizon':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium_igsm_horizon import base
            elif igsm_variant == 'parallel':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium_igsm_parallel import base
            elif igsm_variant == 'combine':
                from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium_igsm_combine import base
            else:
                raise ValueError(f"Invalid IGSM variant: {igsm_variant}")
        else:
            from mas_r1_reasoner.agents.prompts.user_prompt.propose_harmony_medium import base
        EXAMPLE = ''
    elif eval_building_blocks:
        # Not really run, just a placeholder for evaluation
        from mas_r1_reasoner.agents.prompts.propose import base, EXAMPLE
        EXAMPLE = ''
    elif decompose_only:
        if include_blocks and add_judge:
            from mas_r1_reasoner.agents.prompts.propose_decompose_only_inlucde_block_judge import base, EXAMPLE
        elif include_blocks:
            from mas_r1_reasoner.agents.prompts.propose_decompose_only_inlucde_block import base, EXAMPLE
        else:
            from mas_r1_reasoner.agents.prompts.propose_decompose_only import base, EXAMPLE
    elif architecture_only:
        raise NotImplementedError #TODO: not run this yet
    elif architecture_only_sequential:
        raise NotImplementedError #TODO: not run this yet

    elif enable_tree_architecture:
        if level == 1:
            if include_blocks and add_judge:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_1_inlucde_block_judge import base, EXAMPLE
            elif include_blocks:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_1_inlucde_block import base, EXAMPLE
            else:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_1 import base, EXAMPLE
        elif level == 2:
            if include_blocks and add_judge:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_2_include_block_judge import base, EXAMPLE
            elif include_blocks:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_2_include_block import base, EXAMPLE
            else:
                from mas_r1_reasoner.agents.prompts.propose_hirerchical_level_2 import base, EXAMPLE
        else:
            raise ValueError(f"Invalid level: {level}")
    else:
        raise NotImplementedError #TODO: not run this yet
        # full design
        from mas_r1_reasoner.agents.prompts.propose import base, EXAMPLE
    

    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    
    if enable_tree_architecture and level == 2:
        if sub_task is None:
            raise ValueError("sub_task is required for level 2 prompts")
        elif '[SUB-TASK]' in prompt:
            prompt = prompt.replace("[SUB-TASK]", sub_task)
        elif '[CODE]' in prompt:
            prompt = prompt.replace("[CODE]", sub_task)
        else:
            raise ValueError(f"Invalid prompt: {prompt}")

    if architecture_only_sequential and level == 2:
        prompt = prompt.replace("[CODE]", sub_task)        

    if '[QUESTION]' in prompt:
        prompt = prompt.replace("[QUESTION]", question)


    if problem_type == 'harmony_minimal':
        from mas_r1_reasoner.agents.prompts.system_prompt.harmony_minimal import system_prompt
        from mas_r1_reasoner.agents.prompts.developer_prompt.harmony_minimal import developer_prompt

        cot = json.dumps(archive[0])
        cot_sc = json.dumps(archive[1])
        reflexion = json.dumps(archive[2])
        llm_debate = json.dumps(archive[3])

        developer_prompt = developer_prompt.replace("[COT]", cot)        
        developer_prompt = developer_prompt.replace("[COT_SC]", cot_sc)        
        developer_prompt = developer_prompt.replace("[Debate]", llm_debate)        
        developer_prompt = developer_prompt.replace("[Reflexion]", reflexion)        
        system_prompt = system_prompt.replace("[MODEL]", node_model)        

        return system_prompt, prompt, developer_prompt

    elif problem_type == 'harmony_medium':
        from mas_r1_reasoner.agents.prompts.system_prompt.harmony_medium import system_prompt
        if use_igsm_prompt:
            from mas_r1_reasoner.agents.prompts.developer_prompt.harmony_medium_igsm import developer_prompt
        else:
            from mas_r1_reasoner.agents.prompts.developer_prompt.harmony_medium import developer_prompt

        cot = json.dumps(archive[0])
        cot_sc = json.dumps(archive[1])
        reflexion = json.dumps(archive[2])
        llm_debate = json.dumps(archive[3])
        web_search = json.dumps(archive[4])


        developer_prompt = developer_prompt.replace("[COT]", cot)        
        developer_prompt = developer_prompt.replace("[COT_SC]", cot_sc)        
        developer_prompt = developer_prompt.replace("[Debate]", llm_debate)        
        developer_prompt = developer_prompt.replace("[Reflexion]", reflexion)      
        developer_prompt = developer_prompt.replace("[Websearch]", web_search)          
        system_prompt = system_prompt.replace("[MODEL]", node_model)        

        return system_prompt, prompt, developer_prompt

    if 'direct' in problem_type:
        from mas_r1_reasoner.agents.prompts.system_prompt.direct import system_prompt
    elif problem_type == 'mcp':
        from mas_r1_reasoner.agents.prompts.system_prompt.mcp import system_prompt
    else:
        # default is with code
        from mas_r1_reasoner.agents.prompts.system_prompt.default import system_prompt

    return system_prompt, prompt



def get_known_prompt(question, indicator, level=1):
    """
    Get known prompt based on the indicator.
    
    Args:
        question: The question to solve
        indicator: The prompt type indicator (e.g., 'hierarchical_grpo', 'sub_task_grpo')
        
    Returns:
        tuple: (system_prompt, prompt)
    """
    if indicator == 'hierarchical_grpo':
        if level == 1:
            from mas_r1_reasoner.agents.known_prompt.hierarchical_grpo_level_1_prompt import prompt
        elif level == 2:
            from mas_r1_reasoner.agents.known_prompt.hierarchical_grpo_level_2_prompt import prompt
        else:
            raise ValueError(f"Unknown level: {level}")
        from mas_r1_reasoner.agents.known_prompt.hierarchical_grpo_system_prompt import system_prompt
    elif indicator == 'sub_task_grpo':
        from mas_r1_reasoner.agents.known_prompt.sub_task_grpo_prompt import prompt
        from mas_r1_reasoner.agents.known_prompt.sub_task_grpo_system_prompt import system_prompt
    else:
        raise ValueError(f"Unknown prompt indicator: {indicator}")

    if '[QUESTION]' in prompt:
        prompt = prompt.replace("[QUESTION]", question)

    return system_prompt, prompt