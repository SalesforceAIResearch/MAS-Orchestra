"""
[ToolOrchestra step 2] Answer extraction for multi-turn (dynamic plan-execute) episodes.

The decoded response of a multi-turn episode is the whole conversation after the prompt:
assistant turns, <tool_call> blocks and <tool_response> blocks (agent results). Agents were already
executed during the rollout, so every episode is scored like a single-shot "direct answer": the final
<answer> written by the orchestrator is compared with the ground truth.
"""

import re
from typing import Tuple

from mas_r1_reasoner.agents.common import main_rank_print

_TOOL_RESPONSE = re.compile(r"<tool_response>.*?(</tool_response>|$)", re.DOTALL)
_TOOL_CALL = re.compile(r"<tool_call>.*?(</tool_call>|$)", re.DOTALL)
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_final_turn(response_text: str) -> str:
    """Text of the orchestrator's last turn (after the last agent result)."""
    last_turn = response_text.rsplit("</tool_response>", 1)[-1]
    return _TOOL_CALL.sub("", last_turn).strip()


def extract_final_answer(response_text: str) -> str:
    """Last <answer> written by the orchestrator, ignoring anything inside agent results or tool calls."""
    orchestrator_text = _TOOL_CALL.sub("", _TOOL_RESPONSE.sub("", response_text))
    answers = _ANSWER.findall(orchestrator_text)
    return answers[-1].strip() if answers else ""


def extract_harmony_code_from_response(response_text: str, validate_python_code, logger) -> Tuple[str, str, str]:
    """Same return contract as harmony_parser.minimal: (code, name, thought)."""
    answer = extract_final_answer(response_text)
    if answer:
        return "direct_answer", "direct_answer", answer
    # Same fallback as minimal.py when no <answer> is found: score the last turn's text.
    main_rank_print("No <answer> found in multi-turn episode; using the final turn text")
    return "direct_answer", "direct_answer", extract_final_turn(response_text)
