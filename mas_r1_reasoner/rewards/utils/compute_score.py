"""
MAS-R1 Compute Score Function for VERL NaiveRewardManager.
This implements the MAS-R1 reward logic as a compute_score function that can be used
with the existing VERL NaiveRewardManager instead of creating a custom reward manager.

MULTIPLE EXECUTION SUPPORT:
This reward function supports multiple executions per generated code for continuous reward computation.

Configuration Example:
```yaml
azr:
  mas_r1:
    num_executions: 5  # Execute each code 5 times
    enable_code_execution: true
    enable_async_execution: true
    code_execution_timeout: 30
```

Reward Logic:
1. Code execution success: Binary (1.0 if any execution succeeds, 0.0 otherwise)
2. Final answer correctness: Continuous (fraction of executions that produced correct answer)

For single execution (num_executions=1):
- Both execution success and answer correctness are binary

"""
import re
import torch
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from verl import DataProto
from mas_r1_reasoner.agents.common import main_rank_print, EQUALITY_TEMPLATE
from mas_r1_reasoner.agents.shared_vars import get_global, set_global
from typing import Tuple
import os
from mas_r1_reasoner.rewards.utils.string_match_score import MathScorer
from math_verify import parse, verify
from math_verify.errors import TimeoutException
# HF-verify is not reliable, in particular in equation problem. We use math_scorer instead (fixed the known issue, heading 0 below)


async def mas_r1_compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute MAS-R1 reward score for code generation + execution approach.
    
    This function implements the MAS-R1 reward logic:
    1. Code execution success (0.5 weight)
    2. Final answer correctness (0.5 weight) 
    
    Supports both single and multiple execution scenarios.
    
    **EVALUATION vs TRAINING MODE:**
    - **Training**: Uses multiple executions for continuous rewards (gradient information)
    - **Evaluation**: Uses single execution for hard rewards (accurate assessment)
    
    Args:
        data_source: Source of the data (e.g., 'gsm8k', 'math')
        solution_str: Generated solution string (code) (TODO: Not used in this function)
        ground_truth: Ground truth answer
        extra_info: Optional dictionary containing execution results and statistics
        
    Returns:
        Reward score (0.0 to 1.0)
    """
    if not extra_info:
        main_rank_print("WARNING: No extra_info provided, returning 0.0")
        return 0.0
    
    execution_result = extra_info.get('execution_result', {})
    execution_stats = extra_info.get('execution_stats', {})
    is_validation = extra_info.get('is_validation', False)
    
    # Get configuration
    config = extra_info['mas_r1_config']
    
    # Determine if this is multiple execution scenario
    total_executions = execution_stats.get('total_executions', 1)
    
    if total_executions > 1 and not is_validation:
        # Multiple executions for training - use continuous rewards
        raise ValueError("Multiple executions for training are not supported for now")
    else:
        # Single execution (evaluation or single execution training) - use hard rewards
        return await _compute_single_execution_reward(execution_result, ground_truth, config, extra_info, is_validation)


async def _compute_single_execution_reward(execution_result: Dict[str, Any], ground_truth: str, config: Dict[str, Any], extra_info: Dict[str, Any], is_validation: bool) -> float:
    """
    Compute reward for single execution scenario (original behavior).
    
    Args:
        execution_result: Dictionary containing execution results
        ground_truth: The ground truth answer
        config: Configuration dictionary
        extra_info: Extra info dictionary to store scores
        is_validation: Boolean indicating if this is evaluation mode
        
    Returns:
        Combined reward score (0.0 to 1.0)
    """
    # Detect if this is a horizon multi-answer problem
    is_horizon_problem = '<<horizon>>' in ground_truth
    
    # Step 1: Evaluate code execution success
    code_execution_success = 1.0 if execution_result.get('success', False) else 0.0
    
    # Step 2: Evaluate final answer correctness
    predicted_answer = _extract_predicted_answer_from_execution(execution_result, config, extra_info, is_horizon_problem)
    question = execution_result.get('question')
    answer_correct = await _compare_answers(question, predicted_answer, ground_truth, extra_info, is_validation, is_horizon_problem)
    final_answer_correctness = 1.0 if answer_correct else 0.0
    
    # Log the final answer correctness
    main_rank_print(f"final_answer_correctness: {final_answer_correctness}")
    
    # Step 3: Compute combined reward using weights
    combined_reward = (
        config['code_execution_weight'] * code_execution_success +
        config['final_answer_weight'] * final_answer_correctness
    )
    
    # Store detailed scores in extra_info for logging/debugging
    if extra_info is not None:
        extra_info['mas_r1_scores'] = {
            'code_execution_success': code_execution_success,
            'final_answer_correctness': final_answer_correctness,
            'combined_reward': combined_reward,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'execution_success': execution_result.get('success', False)
        }
    
    return combined_reward




def _extract_predicted_answer_from_execution(execution_result: Dict[str, Any], config: Dict[str, Any], extra_info: Dict[str, Any], is_horizon_problem: bool) -> str:
    """
    Extract predicted answer from execution result.
    
    Args:
        execution_result: Dictionary containing execution results
        config: Configuration dictionary with answer_pattern and default_answer_on_failure
        extra_info: Dictionary containing extra information
        is_horizon_problem: Boolean indicating if this is a horizon multi-answer problem
        
    Returns:
        Extracted answer string
        
    Raises:
        RuntimeError: If execution result is invalid or answer cannot be extracted
    """
    # make_finak_answer directly return the final_answer, no need to extract anymore

    if not execution_result:
        raise RuntimeError("Execution result is empty")

    result = execution_result.get('result')
    error = execution_result.get('error')
    
    # For horizon multi-answer problems, directly return the whole result
    if is_horizon_problem:
        main_rank_print(f"Return the answer, using whole result as answer: '{result}'")
        return str(result).strip()

    # We still use the two scores for multi-choice, as they are general enough
    # we cannot use direct comparison, as it does not fit the training data (WebInstruct)
    
    if error: #error
        #TODO: this only correct if there is one single answer
        main_rank_print(f"WARNING: Execution was not successful: {error}")
        # for error, we may need to extract answer
        math_scorer = extra_info['math_scorer']
        ans = math_scorer.extract_solution(error)
        if ans:
            extracted = ans.strip()
        else:
            print(f'No answer extracted from error {error}. Return the error directly.')
            extracted = error
        print(f"✓ Extracted with math_scorer: '{extracted}'")
        return extracted

    main_rank_print(f"Return the answer, using whole result as answer: '{result}'")
    return str(result).strip()



def _fix_leading_zero_numbers(expr: str) -> str:
    # Number w/o leading 0
    pattern = r'\b0+([1-9]\d*)\b'
    
    # Replace with second group
    return re.sub(pattern, r'\1', expr)

def _extract_all_boxed_answers(solution_str: str) -> List[str]:
    """
    Extract ALL boxed answers from a solution string for R-HORIZON style multi-answer problems.
    
    Args:
        solution_str: Solution string containing multiple \\boxed{} notations
        
    Returns:
        List of extracted answers in order of appearance
    """
    answers = []
    idx = 0
    
    while True:
        # Find next occurrence of \boxed
        idx = solution_str.find("\\boxed", idx)
        if idx < 0:
            break
            
        # Check if it's \boxed{} notation
        if idx + 7 <= len(solution_str) and solution_str[idx:idx+7] == "\\boxed{":
            # Handle standard \boxed{} notation
            i = idx + 7  # Start after \boxed{
            num_braces_open = 1
            right_brace_idx = None
            
            while i < len(solution_str):
                if solution_str[i] == "{":
                    num_braces_open += 1
                elif solution_str[i] == "}":
                    num_braces_open -= 1
                    if num_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1
            
            if right_brace_idx is not None:
                answer = solution_str[idx+7:right_brace_idx].strip()
                answers.append(answer)
                idx = right_brace_idx + 1
            else:
                idx += 7
        else:
            idx += 6
    
    return answers


async def _compare_multi_answers(question: str, predicted: str, ground_truth: str, extra_info: Dict[str, Any], is_validation: bool) -> bool:
    """
    Compare multiple predicted answers with ground truth for R-HORIZON style problems.
    Returns True only if ALL sub-problem answers are correct.
    
    Args:
        question: The question text
        predicted: The full model response containing multiple \\boxed{} answers
        ground_truth: <<horizon>> delimited ground truth answers (e.g., "9<<horizon>>5")
        extra_info: Dictionary containing scorer
        is_validation: Whether this is validation mode
        
    Returns:
        True if ALL answers match their corresponding ground truths, False otherwise
    """
    # Parse ground truth (<<horizon>> delimited)
    ground_truth_answers = [ans.strip() for ans in ground_truth.split('<<horizon>>')]
    num_expected_answers = len(ground_truth_answers)
    
    main_rank_print(f"  - Multi-answer problem: expecting {num_expected_answers} answers")
    main_rank_print(f"  - Ground truth answers: {ground_truth_answers}")
    
    # Extract all boxed answers from predicted response
    predicted_answers = _extract_all_boxed_answers(predicted)
    
    main_rank_print(f"  - Found {len(predicted_answers)} boxed answers in response")
    main_rank_print(f"  - Predicted answers: {predicted_answers}")
    
    # Check if we have the right number of answers
    if len(predicted_answers) != num_expected_answers:
        main_rank_print(f"  - Answer count mismatch: expected {num_expected_answers}, got {len(predicted_answers)}")
        return False
    
    # Compare each answer
    use_llm_judge = get_global("global_use_llm_judge")
    problem_type = get_global("global_problem_type")
    
    # Determine when to use LLM judge
    should_use_llm_judge = False
    if use_llm_judge:
        if 'medium' in problem_type:
            should_use_llm_judge = True
        else:
            should_use_llm_judge = not is_validation
    
    all_correct = True
    for i, (pred_ans, gt_ans) in enumerate(zip(predicted_answers, ground_truth_answers)):
        main_rank_print(f"  - Comparing answer {i+1}: '{pred_ans}' vs '{gt_ans}'")
        
        if should_use_llm_judge:
            llm_judge = extra_info['llm_judge']
            from mas_r1_reasoner.rewards.utils.llm_as_judge import check_equality_llm
            is_correct = await check_equality_llm(question, gt_ans, pred_ans, llm_judge)
        else:
            # Use MathScorer
            try:
                fixed_gt = _fix_leading_zero_numbers(gt_ans)
                scorer = extra_info['math_scorer']
                score_my_scorer = scorer.grade_answer(pred_ans, fixed_gt)
                
                # Try math_verify
                score_hf_verify = False
                try:
                    gold = parse(gt_ans)
                    answer = parse(pred_ans)
                    score_hf_verify = verify(gold, answer)
                except TimeoutException:
                    score_hf_verify = False
                except Exception:
                    score_hf_verify = False
                
                is_correct = score_my_scorer or score_hf_verify
            except Exception as e:
                main_rank_print(f"  - Error comparing answer {i+1}: {e}")
                is_correct = False
        
        main_rank_print(f"  - Answer {i+1} correct: {is_correct}")
        
        if not is_correct:
            all_correct = False
            break  # Early exit if any answer is wrong
    
    main_rank_print(f"  - All answers correct: {all_correct}")
    return all_correct


async def _compare_answers(question: str, predicted: str, ground_truth: str, extra_info: Dict[str, Any] = None, is_validation: bool = False, is_horizon_problem: bool = False) -> bool:
    """
    Compare predicted answer with ground truth using either LLM judge or MathScorer.
    Supports both single-answer and multi-answer (R-HORIZON style) problems.
    
    Args:
        question: The question text (used by LLM judge)
        predicted: Predicted answer string (already extracted)
        ground_truth: Ground truth answer string (format: "answer1<<horizon>>answer2<<horizon>>..." for multi-answer)
        extra_info: Dictionary that must contain either 'llm_judge' (sampler) or 'math_scorer'
        is_validation: Boolean indicating if this is validation mode (LLM judge not used during validation)
        is_horizon_problem: Boolean indicating if this is a horizon multi-answer problem
        
    Returns:
        True if answers match, False otherwise. For multi-answer problems, returns True only if ALL answers match.
        
    Raises:
        RuntimeError: If inputs are invalid or required scorer is not provided
    """
    main_rank_print(f"Starting _compare_answers...")
    main_rank_print(f"  - Predicted: '{predicted}'")
    main_rank_print(f"  - Ground truth: '{ground_truth}'")
    main_rank_print(f"  - Is validation: {is_validation}")
    
    # Check if this is a multi-answer problem (R-HORIZON style)
    if is_horizon_problem:
        main_rank_print(f"  - Detected multi-answer problem (R-HORIZON style)")
        return await _compare_multi_answers(question, predicted, ground_truth, extra_info, is_validation)
    
    if not predicted:
        return False

    if not ground_truth:
        raise RuntimeError("Ground truth cannot be empty")

    #TODO: for some datasets, one might need to use LLM to grade the answer
    use_llm_judge = get_global("global_use_llm_judge")
    problem_type = get_global("global_problem_type")
    
    # Determine when to use LLM judge based on problem type
    # - 'minimal' or 'harmony_minimal': Use LLM judge only for training (not validation)
    # - 'medium' or 'harmony_medium': Use LLM judge for both training and validation
    should_use_llm_judge = False
    if use_llm_judge:
        if 'medium' in problem_type:
            # For medium: use LLM judge for both training and validation
            should_use_llm_judge = True
            main_rank_print(f"  - Problem type '{problem_type}': Using LLM judge for both training and validation")
        else:
            # For minimal: use LLM judge only for training
            should_use_llm_judge = not is_validation
            if should_use_llm_judge:
                main_rank_print(f"  - Problem type '{problem_type}': Using LLM judge for training only")
            else:
                main_rank_print(f"  - Problem type '{problem_type}': Skipping LLM judge for validation")
    
    if should_use_llm_judge:
        main_rank_print(f"  - Using LLM judge for grading")
        llm_judge = extra_info['llm_judge']
        from mas_r1_reasoner.rewards.utils.llm_as_judge import check_equality_llm
        # check_equality_llm(question, correct_answer, candidate_response, grader_model)
        score = await check_equality_llm(question, ground_truth, predicted, llm_judge)
        return score

    else:
        main_rank_print(f"  - Using MathScorer for grading")
        try:
            # Fix AIME24 ground truth leading '0' issue before grading
            fixed_ground_truth = _fix_leading_zero_numbers(ground_truth)
            main_rank_print(f"  - Fixed ground truth: '{fixed_ground_truth}'")

            scorer = extra_info['math_scorer']
            score_my_scorer = scorer.grade_answer(predicted, fixed_ground_truth)

            # Try math_verify but catch TimeoutException (inherits from BaseException, not Exception)
            score_hf_verify = False
            try:
                gold = parse(ground_truth)
                answer = parse(predicted)
                score_hf_verify = verify(gold, answer)
            except TimeoutException:
                main_rank_print(f"  - math_verify timeout, using MathScorer only")
                score_hf_verify = False
            except Exception as e:
                main_rank_print(f"  - math_verify error: {e}, using MathScorer only")
                score_hf_verify = False
                    
            main_rank_print(f"  - score_hf_verify: '{score_hf_verify}', score_my_scorer: '{score_my_scorer}'")

            # Return True if either scorer returns True
            return score_my_scorer or score_hf_verify
            
        except Exception as e:
            main_rank_print(f"  - Error in grading: {e}")
            return False


# Factory function to create MAS-R1 compute_score with custom config
def create_mas_r1_compute_score(config: Dict[str, Any]):
    """
    Create a MAS-R1 compute_score function with custom configuration.
    
    Args:
        config: Configuration dictionary with MAS-R1 specific settings
        
    Returns:
        A compute_score function that can be used with NaiveRewardManager
    """
    async def mas_r1_compute_score_with_config(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
        # Merge config with extra_info
        extra_info['mas_r1_config'] = config
        extra_info['math_scorer'] = MathScorer()


        use_llm_judge = get_global("global_use_llm_judge")
        if use_llm_judge:
            model_sampler_map = get_global("global_model_sampler_map")
            llm_judge = model_sampler_map['qwen-2.5-72b-instr'] #fix it as 72b qwen
            extra_info['llm_judge'] = llm_judge

        return await mas_r1_compute_score(data_source, solution_str, ground_truth, extra_info)
    
    return mas_r1_compute_score_with_config 