import pandas as pd
from mas_r1_reasoner.agents.shared_vars import set_global, get_global
from mas_r1_reasoner.agents.common import extract_xml
import re
import json


# LLM-based grader template from browse_ref.py
GRADER_TEMPLATE = """
Judge whether the following [response] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:


thinking: Explain why the [response] is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the [response]. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if [response] matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the [response] is incorrect.

""".strip()


def check_equality_llm(question, correct_answer, candidate_response, grader_model):
    """
    Check equality using LLM-based judging.
    
    Args:
        correct_answer: The correct answer to compare against
        candidate_response: The candidate's response to evaluate
        grader_model: The sampler object to use for grading
        
    Returns:
        bool: True if the response is correct, False otherwise
    """

    output_fields = ['thinking', 'correct']

    # Use the same system prompt structure as agent_system.py
    FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following XML format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n"""

    output_description = "Return ONLY 'yes' or 'no' and DO NOT return anything other than these two."
    thinking_description = "Give your detailed reasoning for judging the response correctness."

    output_fields_and_description = '\n'.join([f"<{key}> [Your {key}. {thinking_description}] </{key}>" if 'thinking' in key else f"<{key}> [Your {key}. {output_description}] </{key}>\n" for key in output_fields])
    
    system_prompt = 'You are a helpful assistant. ' + FORMAT_INST(output_fields_and_description)
    
    grader_prompt = GRADER_TEMPLATE.format(
        correct_answer=correct_answer,
        response=candidate_response,
        question=question,
    )

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": grader_prompt}
    ]

    # Define output fields
    output_fields = ['thinking', 'correct']
    max_retries = 5
    debug_count = 0
    all_errors = []
    response_text = ""
    
    while debug_count < max_retries:
        debug_count += 1
        try:
            # Use the grader_model sampler directly (don't reinitialize)
            response_text = grader_model(prompt_messages)
            
            # Initialize response dict
            response_dict = {}
            
            # Validate XML response
            is_valid_response = True
            for field in output_fields:
                extracted_content = extract_xml(response_text, field)
                if not extracted_content:
                    is_valid_response = False
                    print(f"✗ Missing field: {field}")
                    break
                else:
                    response_dict[field] = extracted_content.strip()

            # Check if answer field is not empty
            if 'correct' in output_fields:
                answer_content = extract_xml(response_text, 'correct')
                if not answer_content or len(answer_content.strip()) == 0:
                    is_valid_response = False
                    print("✗ Correct field is empty")

            if is_valid_response:
                # Convert to boolean
                thinking = response_dict['thinking']
                correct = response_dict['correct']
                result = correct.lower().strip() == "yes"
                
                print(f"LLM Judge - Question: {question}; Correct Answer: {correct_answer}; Response: {candidate_response}; Thinking: {thinking}; Correct: {correct} -> {result}")
                return result
            else:
                print(f'Invalid XML response. Required fields: {output_fields}, response: {response_text[:200]}... (Attempt {debug_count})')
                
                extra_instr = '\n'.join([f"<{key}> [Your {key}.] </{key}>" for key in output_fields])
                prompt_messages[-1]['content'] += f'Reply EXACTLY with the following XML format.\n{extra_instr}\n\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!'

                all_errors.append(f"Attempt {debug_count}: Invalid XML response - Required fields: {output_fields}, response: {response_text}")

                if debug_count >= max_retries:
                    break

        except Exception as e:
            error_msg = f"Attempt {debug_count}: {type(e).__name__}: {str(e)}"
            print(f'Execute Error: {error_msg}')
            all_errors.append(error_msg)
            
            if response_text:
                print(f'Response text: {response_text[:200]}...')
            
            if debug_count >= max_retries:
                break
    
    # If all retries failed, treat as incorrect (False) instead of raising error
    error_summary = f"Failed to get valid LLM judge response after {max_retries} attempts. Errors: {all_errors}"
    print(error_summary)
    print("⚠️ Treating LLM judge failure as incorrect answer (returning False)")
    return False
