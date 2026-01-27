#!/usr/bin/env python3
"""
Utility module for executing building blocks.
Uses the high-performance execute_mas_batch_sync infrastructure from agent_system_async.py.
"""

import os
import sys
import asyncio
import concurrent.futures
from typing import Dict, List

# Add the mas_r1_reasoner path for proper imports
mas_r1_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mas_r1_path not in sys.path:
    sys.path.append(mas_r1_path)

    # Import MAS-R1 components
from mas_r1_reasoner.agents.agent_system import LLMAgentBase, AgentSystem, Info
from mas_r1_reasoner.agents.shared_vars import set_global, get_global
from mas_r1_reasoner.agents.common import main_rank_print
# from mas_r1_reasoner.agents.agent_system_process import AsyncAgentSystem as ProcessAgentSystem
from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem as AsyncAgentSystem
from mas_r1_reasoner.agents.blocks.cot import COT
from mas_r1_reasoner.agents.blocks.cot_sc import COT_SC
from mas_r1_reasoner.agents.blocks.llm_debate import LLM_debate
from mas_r1_reasoner.agents.blocks.reflexion import Reflexion
from mas_r1_reasoner.agents.blocks.web_search import WebSearch

#TODO: add web search block, this should be flwxible

def execute_basic_blocks_batch(questions: List[str], timeout: int = 30, config=None) -> List[Dict[str, str]]:
    """
    Synchronous function to execute 4 basic building blocks on multiple questions in parallel.
    Uses the high-performance execute_mas_batch_sync infrastructure from agent_system_async.py.
    
    Args:
        questions: List of questions to process
        timeout: Timeout for execution
        config: Configuration object to determine which blocks to use
    """
    if not questions:
        return []
    
    # Determine which blocks to use based on global variable
    use_igsm_blocks = get_global("global_use_igsm_prompt")
    if use_igsm_blocks is None:
        use_igsm_blocks = False
    
    # Import blocks conditionally
    if use_igsm_blocks:
        print("🔧 Using IGSM blocks for execution")
        # Use IGSM blocks
        from mas_r1_reasoner.agents.blocks.igsm.cot import COT as IGSM_COT
        from mas_r1_reasoner.agents.blocks.igsm.cot_think import COT as IGSM_COT_THINK
        from mas_r1_reasoner.agents.blocks.igsm.llm_debate import LLM_debate as IGSM_LLM_debate
        from mas_r1_reasoner.agents.blocks.igsm.cot_sc import COT_SC as IGSM_COT_SC
        from mas_r1_reasoner.agents.blocks.igsm.reflexion import Reflexion as IGSM_Reflexion
        # TODO: Only COT is available in igsm directory currently
        # For other blocks, fall back to regular blocks
        COT_SC_IMPORT = IGSM_COT_SC
        LLM_debate_IMPORT = IGSM_LLM_debate
        Reflexion_IMPORT = IGSM_Reflexion
        WebSearch_IMPORT = WebSearch
        COT_IMPORT = IGSM_COT
        COT_THINK_IMPORT = IGSM_COT_THINK
    else:
        print("🔧 Using regular blocks for execution")
        # Use regular blocks
        COT_IMPORT = COT
        COT_SC_IMPORT = COT_SC
        LLM_debate_IMPORT = LLM_debate
        Reflexion_IMPORT = Reflexion
        WebSearch_IMPORT = WebSearch
        #TODO: COT_THINK_IMPORT is not defined in regular blocks
    
    print(f"🚀 Executing building blocks for {len(questions)} questions using execute_mas_batch_sync")

    # Get building blocks from global archive (names only)
    building_blocks = get_global("global_init_archive")


    # Get building block codes using the conditionally imported blocks
    building_blocks_code_map = {
        'COT': COT_IMPORT['code'],
        'COT_SC': COT_SC_IMPORT['code'], 
        'LLM_debate': LLM_debate_IMPORT['code'],
        'Reflexion': Reflexion_IMPORT['code'],
        'WebSearch': WebSearch_IMPORT['code']
    }
    
    # Add COT_THINK only when using IGSM blocks
    if use_igsm_blocks:
        building_blocks_code_map['COT_THINK'] = COT_THINK_IMPORT['code']
    
    # Create all tasks (question × building_block combinations)
    codes = []
    task_infos = []
    task_metadata = []  # To track which question/block each task belongs to
    
    for question_idx, question in enumerate(questions):
        for block_name in building_blocks:
            # Get code from building_blocks_code_map using block_name
            block_code = building_blocks_code_map.get(block_name)
            if block_code is None:
                raise ValueError(f"Building block '{block_name}' not found in building_blocks_code_map")
            codes.append(block_code)
            # Create proper Info namedtuple object
            task_info = Info(
                name='task',
                author='user',
                content=question,
                msg=None,
                sub_tasks=[],
                agents=[],
                iteration_idx=0,
                final_answer=None
            )
            task_infos.append(task_info)
            task_metadata.append((question_idx, block_name))
    
    print(f"📊 Total tasks: {len(codes)} ({len(questions)} questions × {len(building_blocks)} blocks)")
    
    # Use AsyncAgentSystem to execute all tasks
    try:
        multiply_processes = get_global("global_multiply_processes")

        if multiply_processes == 0:
            # Use async execution
            # raise ValueError("Async execution is not supported")
            main_rank_print(f"Initializing ASYNC AgentSystem...")
            agent_system = AsyncAgentSystem()
            main_rank_print(f"✓ Async AgentSystem initialized with agent configuration")
        else:
            raise ValueError("Process execution is not supported")
            # Use process execution
            # main_rank_print(f"Initializing PROCESS AgentSystem (multiply_processes={multiply_processes})...")
            # agent_system = ProcessAgentSystem()
            # main_rank_print(f"✓ Process AgentSystem initialized with agent configuration")

        results = agent_system.execute_mas_batch_sync(codes, task_infos, timeout)
        
        # Organize results by question
        question_results = {}
        for i, (result, success, error) in enumerate(results):
            question_idx, block_name = task_metadata[i]
            
            # Initialize question result if not exists
            if question_idx not in question_results:
                question_results[question_idx] = {
                'question_idx': question_idx,
                'question': questions[question_idx],
                'block_outputs': {},
                'errors': []
            }
            
            # Store result
            if success:
                question_results[question_idx]['block_outputs'][block_name] = result
            else:
                question_results[question_idx]['block_outputs'][block_name] = error #TODO: use unavailable later
                question_results[question_idx]['errors'].append(f"{block_name}: {error}")
        
        # Convert to list and sort by question index
        final_results = []
        for question_idx in sorted(question_results.keys()):
            final_results.append(question_results[question_idx])
        
        print(f"✅ Completed building blocks execution for {len(questions)} questions")
        return final_results
            
    except Exception as e:
        print(f"❌ Error in building blocks execution: {e}")
        raise Exception(f"Error in building blocks execution: {e}")
