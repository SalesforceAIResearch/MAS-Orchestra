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
Execution utilities for MAS-R1 Trainer.
"""

from typing import Dict, List, Tuple, Any
from mas_r1_reasoner.agents.common import main_rank_print


def execute_codes_and_store_results(trainer_instance, question_results: Dict, is_validation: bool = False) -> Dict:
    """
    Execute codes and store results in question_results dictionary.
    This is shared between training and validation.
    Supports multiple executions per code for continuous reward computation.
    
    Args:
        trainer_instance: The trainer instance
        question_results: Dictionary containing question data
        is_validation: Whether this is validation (evaluation) mode
        
    Returns:
        Updated question_results dictionary with execution results
    """
    main_rank_print(f"\n{'='*60}")
    main_rank_print("EXECUTING CODES FOR EACH QUESTION")
    main_rank_print(f"{'='*60}")
    
    codes = []
    task_infos = []
    questions_for_execution = []
    ground_truths_for_execution = [] # Added for alignment

    # Collect all {question}<<MySep>>{response} keys for execution
    question_response_keys = []
    for key in question_results.keys():
        if '<<MySep>>' in key and key.split('<<MySep>>')[-1].isdigit():
            question_response_keys.append(key)
    
    # Sort by question and response index for consistent ordering
    # only those with '<<MySep>>' will be executed
    question_response_keys.sort(key=lambda x: (x.rsplit('<<MySep>>', 1)[0], int(x.rsplit('<<MySep>>', 1)[1])))
    
    # Set up global variables once before processing all questions (optimization)
    agent_config = trainer_instance.mas_r1_config.get('agent', {})
    mas_r1_config = trainer_instance.mas_r1_config # Assuming mas_r1_config is available
    trainer_instance.processor.setup_global_variables(agent_config, mas_r1_config, trainer_instance.config)
    
    for question_response_key in question_response_keys:
        response_data = question_results[question_response_key]
        extracted_code_data = response_data.get('extracted_code_data', {})
        
        code = extracted_code_data.get('extracted_code', "")
        if code:
            main_rank_print(f"✓ {question_response_key}: Using extracted code")
        else:
            main_rank_print(f"✗ {question_response_key}: No valid code available")
            code = ""
        
        question = response_data.get('question', '')
        
        # Extract task_info for this question
        task_info = trainer_instance.processor.build_task_info(question)
        
        # #TODO: debug
        # code = 'CoT_SC:10'

        codes.append(code)
        task_infos.append(task_info)
        questions_for_execution.append(question)
        ground_truths_for_execution.append(response_data.get('ground_truth', "Unknown"))
    
    # Get multiple execution configuration
    if is_validation:
        # For evaluation: always use single execution
        num_executions = 1
        main_rank_print(f"EVALUATION MODE: Executing each code once for hard reward computation")
    else:
        # For training: use configured number of executions
        num_executions = trainer_instance.mas_r1_config.get('num_executions', 1)
        main_rank_print(f"TRAINING MODE: Executing each code {num_executions} times for continuous reward computation")
    
    # Execute all codes multiple times using batch execution
    main_rank_print(f"Executing {len(codes)} codes {num_executions} times each...")
    all_execution_results = execute_code_multiple_times(trainer_instance, codes, task_infos, num_executions, questions_for_execution, ground_truths_for_execution)
    
    # Store execution results in dictionary
    main_rank_print(f"\n{'='*60}")
    main_rank_print("STORING EXECUTION RESULTS IN DICTIONARY")
    main_rank_print(f"{'='*60}")
    
    # Process each execution result
    for i, execution_stats in enumerate(all_execution_results):
        if i >= len(question_response_keys):
            raise RuntimeError(f"More execution results ({len(all_execution_results)}) than question-response pairs ({len(question_response_keys)})")
        
        question_response_key = question_response_keys[i]
        response_data = question_results[question_response_key]
        
        question = response_data.get('question', '')
        response_idx = response_data.get('response_idx', 0)
        extracted_code_data = response_data.get('extracted_code_data', {})
        
        # Get ground truth from both sources for validation
        question_ground_truth = response_data.get('ground_truth', 'Unknown')
        execution_ground_truth = execution_stats.get('ground_truth', 'Unknown')
        
        # Validate ground truth alignment
        if question_ground_truth != execution_ground_truth:
            error_msg = f"Ground truth mismatch for {question_response_key}: question='{question_ground_truth}', execution='{execution_ground_truth}'"
            raise RuntimeError(error_msg)
        else:
            main_rank_print(f"✓ Ground truth alignment verified for {question_response_key}")
        
        # Create individual execution stats for this response
        individual_execution_stats = {
            'total_executions': execution_stats.get('total_executions', 1),
            'successful_executions': execution_stats.get('successful_executions', 0),
            'failed_executions': execution_stats.get('failed_executions', 0),
            'execution_results': execution_stats.get('execution_results', []),
            'success_rate': execution_stats.get('success_rate', 0.0),
            'question': question,
            'response_idx': response_idx,
            'ground_truth': question_ground_truth,
            'code': extracted_code_data.get('extracted_code', ''),
            'is_validation': is_validation
        }
        
        # Merge execution stats with existing data in {question}<<MySep>>{response} structure
        question_results[question_response_key]['execution_stats'] = individual_execution_stats
        
        total_executions = execution_stats.get('total_executions', 0)
        successful_executions = execution_stats.get('successful_executions', 0)
        success_rate = execution_stats.get('success_rate', 0.0)
        
        mode_str = "EVALUATION" if is_validation else "TRAINING"
        main_rank_print(f"{question_response_key} ({mode_str}): {successful_executions}/{total_executions} successful "
                        f"(success rate: {success_rate:.2f})")
    
    return question_results


def execute_code_multiple_times(trainer_instance, codes: List[str], task_infos: List[Dict], num_executions: int, questions: List[str] = None, ground_truths: List[str] = None) -> List[Dict]:
    """
    Execute each code multiple times and aggregate results.
    
    Args:
        trainer_instance: The trainer instance
        codes: List of code strings to execute
        task_infos: List of task information dictionaries
        num_executions: Number of times to execute each code
        questions: List of questions (optional) - for alignment verification
        ground_truths: List of ground truths (optional) - for alignment verification
        
    Returns:
        List of execution statistics dictionaries
    """
    if num_executions <= 1:
        # Single execution - execute and compute statistics inline
        execution_results_raw = execute_code(trainer_instance, codes, task_infos)
        
        # Convert single execution results to statistics format
        execution_stats_list = []
        for i, (result, success, error) in enumerate(execution_results_raw):
            execution_stats = {
                'total_executions': 1,
                'successful_executions': 1 if success else 0,
                'failed_executions': 0 if success else 1,
                'execution_results': [(result, success, error)],
                'success_rate': 1.0 if success else 0.0,
                'question': questions[i] if questions and i < len(questions) else f"Question_{i}",
                'ground_truth': ground_truths[i] if ground_truths and i < len(ground_truths) else "Unknown",
                'code_index': i  # Add code index for alignment verification
            }
            execution_stats_list.append(execution_stats)
        
        return execution_stats_list

    else:
        raise ValueError("only support single execution")


def execute_code(trainer_instance, codes: List[str], task_infos: List[Dict]) -> List[Tuple[str, bool, str]]:
    """
    Execute codes using the agent system.
    When add_judge=True, handles special case where code like "CoT:10" should return "10" directly.
    
    Args:
        trainer_instance: The trainer instance
        codes: List of code strings to execute
        task_infos: List of task information dictionaries
        
    Returns:
        List of tuples (result, success, error)
    """
    if trainer_instance.agent_system is None:
        # Return mock results if code execution is disabled
        return [("", False, "Code execution disabled") for _ in codes]
    
    # Check if add_judge is enabled
    from mas_r1_reasoner.agents.shared_vars import get_global
    add_judge = get_global("global_add_judge")
    
    results = []
    codes_to_execute = []
    code_indices_to_execute = []
    
    if add_judge:
        import re
        main_rank_print(f"ADD_JUDGE=True: Checking for building block reference patterns in {len(codes)} codes")
        
        # Building blocks that can be referenced directly
        building_blocks = ['CoT', 'CoT_SC', 'Debate', 'Refine']
        
        for i, code in enumerate(codes):
            main_rank_print(f"Checking code {i}: {code[:100]}{'...' if len(code) > 100 else ''}")
            if not code or not code.strip():
                results.append(("", False, "Empty code"))
                continue
            
            # Check if code matches pattern: BuildingBlock:Answer
            # Pattern matches: "CoT:10", "CoT_SC:42", "Debate:3.14", "Refine:hello world", etc.
            building_block_match = None
            for block in building_blocks:
                pattern = rf'^{re.escape(block)}\s*:\s*(.+)$'
                match = re.search(pattern, code.strip(), re.DOTALL | re.IGNORECASE)
                if match:
                    building_block_match = match
                    break
            
            if building_block_match:
                # Extract the answer directly
                answer = building_block_match.group(1).strip()
                main_rank_print(f"  Code {i}: Building block reference detected -> Direct answer: '{answer}'")
                results.append((answer, True, ""))
            else:
                # Regular code that needs execution
                codes_to_execute.append(code)
                code_indices_to_execute.append(i)
                results.append(None)  # Placeholder
    else:
        # add_judge=False: execute all codes normally
        codes_to_execute = codes
        code_indices_to_execute = list(range(len(codes)))
        results = [None] * len(codes)
    
    # Execute remaining codes that don't match building block patterns
    if codes_to_execute:
        main_rank_print(f"Executing {len(codes_to_execute)} codes using async batch execution")
        task_infos_to_execute = [task_infos[i] for i in code_indices_to_execute]
        execution_results = trainer_instance.agent_system.execute_mas_batch_sync(
            codes_to_execute, task_infos_to_execute, trainer_instance.code_execution_timeout
        )
        
        # Fill in the execution results
        for j, execution_result in enumerate(execution_results):
            original_index = code_indices_to_execute[j]
            results[original_index] = execution_result
    
    return results