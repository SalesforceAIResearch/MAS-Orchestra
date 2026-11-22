"""
MAS-Orchestra building blocks exposed as verl multi-turn tools.

[ToolOrchestra step 2] These tools are the "environment" of the dynamic plan-execute loop.
The orchestrator emits a Hermes tool call (``<tool_call>{"name": "DebateAgent", ...}</tool_call>``),
verl's ``AsyncSGLangRollout._async_rollout_a_request`` dispatches it to ``execute`` below, and the
returned string is appended to the conversation as a ``<tool_response>`` (loss-masked) before the
orchestrator's next turn.

The agents themselves are the unchanged MAS-Orchestra blocks in ``agents/blocks_harmony`` reached
through ``AgentSystem.CoTAgent`` / ``SCAgent`` / ``DebateAgent`` / ``ReflexionAgent``, so prompts,
temperature, number of samples, debate rounds and reflection rounds are identical to single-shot
MAS-Orchestra.
"""

import ast
import asyncio
import json
import logging
import os
import time
from typing import Any, Optional, Tuple
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

from mas_r1_reasoner.agents.sampler.openrouter_completion_sampler import call_usage, new_usage

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SUB_TASK_TEMPLATE = (
    "Original task: {question}; Sub-task: {agent_input}. Please solve the sub-task first. "
    "If its result directly answers the original question, output that as the final answer. "
    "Otherwise, use the sub-task result as guidance to complete the original question and output "
    "the final answer to the original question."
)

_AGENT_SYSTEM = None
_RUNTIME_CONFIG = None
_SEMAPHORES = {}


def _ensure_agent_runtime(config: dict):
    """
    Rebuild the MAS-Orchestra agent globals inside the rollout worker process.

    The trainer sets these globals in the driver (``BaseDatasetProcessor._setup_global_variables_internal``),
    but tools run inside the SGLang rollout worker, which is a separate Ray process.
    """
    global _AGENT_SYSTEM, _RUNTIME_CONFIG
    if _AGENT_SYSTEM is not None:
        return _AGENT_SYSTEM

    from mas_r1_reasoner.agents.shared_vars import set_global
    from mas_r1_reasoner.data_precessor.BaseDatasetProcessor import BaseDatasetProcessor
    from mas_r1_reasoner.agents.agent_system import AgentSystem

    node_model = config["node_model"]
    mock_output = bool(config.get("mock_output", False))
    model_sampler_map = BaseDatasetProcessor.create_model_sampler_map(
        dict(config["model_sampler_map"]), mock_output=mock_output
    )
    if node_model not in model_sampler_map:
        raise ValueError(f"node_model '{node_model}' missing from tool model_sampler_map: {list(model_sampler_map)}")

    # Values mirror BaseDatasetProcessor._setup_global_variables_internal for math datasets.
    format_inst = lambda request_keys: (
        f"Reply EXACTLY with the following XML format.\n{str(request_keys)}\n"
        "DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n"
    )
    set_global("global_model_sampler_map", model_sampler_map)
    set_global("global_FORMAT_INST", format_inst)
    set_global("global_output_description", config.get(
        "output_description",
        "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything "
        "other than the integer answer; If the question is asked for more than numeric results, Return what "
        "the question asked and make sure the answer is complete.",
    ))
    set_global("global_cot_instruction", "Please think step by step and then solve the task.")
    set_global("global_debate_role", ["Math Professor", "Grade School Teacher"])
    set_global("global_max_round", int(config.get("max_round", 5)))
    set_global("global_max_sc", int(config.get("max_sc", 5)))
    set_global("global_node_model", node_model)
    set_global("global_use_igsm_prompt", False)
    set_global("global_web_search_type", "none")
    set_global("global_retrieval_method", "bm25")
    set_global("global_reasoning_effort", config.get("reasoning_effort", "low"))
    max_tokens = config.get("max_tokens", None)
    set_global("global_max_tokens", int(max_tokens) if max_tokens is not None else None)

    _RUNTIME_CONFIG = dict(config)
    _AGENT_SYSTEM = AgentSystem()
    return _AGENT_SYSTEM


def _get_semaphore(limit: int) -> asyncio.Semaphore:
    """One semaphore per event loop, shared by all agent tools in this worker."""
    loop = asyncio.get_event_loop()
    key = (id(loop), limit)
    if key not in _SEMAPHORES:
        _SEMAPHORES[key] = asyncio.Semaphore(limit)
    return _SEMAPHORES[key]


def _parse_debate_roles(raw: Any):
    if isinstance(raw, list):
        roles = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            roles = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                roles = json.loads(raw)
            except json.JSONDecodeError:
                roles = [r.strip() for r in raw.split(",") if r.strip()]
    else:
        roles = None
    if not isinstance(roles, list) or len(roles) < 2 or any(not isinstance(r, str) or len(r) < 2 for r in roles):
        return None
    return roles


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


class MASAgentTool(BaseTool):
    """Base class: one tool per MAS-Orchestra agent. Subclasses only set ``agent_method``."""

    agent_method: str = ""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._agent_system = _ensure_agent_runtime(config)
        self._instances = {}
        self._max_concurrent_calls = int(config.get("max_concurrent_calls", 64))
        self._timeout_s = float(config.get("timeout_s", 900))
        self._max_obs_chars = int(config.get("max_observation_chars", 6000))

    async def create(self, instance_id: Optional[str] = None, question: str = "", **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {"question": question, "num_calls": 0}
        return instance_id

    def _build_task_info(self, question: str, agent_input: str):
        from mas_r1_reasoner.agents.agent_system import Info

        # Same rule as the single-shot parser (rewards/utils/harmony_parser/minimal.py):
        # empty agent_input -> original question, otherwise prepend the original question as context.
        if not agent_input or not agent_input.strip() or agent_input.strip() == '""':
            content = question
        else:
            content = SUB_TASK_TEMPLATE.format(question=question, agent_input=agent_input)
        return Info("task", "User", content, None, None, None, -1, None)

    async def _run_agent(self, task_info, parameters: dict):
        method = getattr(self._agent_system, self.agent_method)
        return await method(agent_input=task_info, model=_RUNTIME_CONFIG["node_model"])

    def _format_observation(self, result) -> str:
        answer = getattr(result, "final_answer", None)
        content = getattr(result, "content", str(result))
        reasoning = content.rsplit("\n\nAnswer:", 1)[0] if "\n\nAnswer:" in content else content
        return (
            f"[{self.name} result]\n"
            f"Final answer from {self.name}: {answer if answer is not None else content}\n"
            f"Reasoning summary (truncated):\n{_truncate(reasoning, self._max_obs_chars)}"
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        state = self._instances.setdefault(instance_id, {"question": "", "num_calls": 0})
        state["num_calls"] += 1
        task_info = self._build_task_info(state["question"], str(parameters.get("agent_input", "") or ""))
        # Cost/latency of this call, consumed by the ToolOrchestra efficiency reward (rewards/utils/toolorchestra_reward.py).
        # Latency is measured after the concurrency semaphore, so queueing in our infrastructure is not counted.
        usage = new_usage()
        metrics = {"start": None, "end": None}
        usage_token = call_usage.set(usage)
        try:
            async with _get_semaphore(self._max_concurrent_calls):
                metrics["start"] = time.time()
                result = await asyncio.wait_for(self._run_agent(task_info, parameters), timeout=self._timeout_s)
            text, status = self._format_observation(result), "success"
        except asyncio.TimeoutError:
            text, status = f"[{self.name} error] The agent timed out after {int(self._timeout_s)}s.", "timeout"
        except ValueError as e:
            text, status = f"[{self.name} error] Invalid arguments: {e}", "invalid_args"
        except Exception as e:
            logger.warning(f"{self.name} failed: {type(e).__name__}: {e}")
            text, status = f"[{self.name} error] {type(e).__name__}: {str(e)[:500]}", "error"
        finally:
            call_usage.reset(usage_token)
        metrics["end"] = time.time()
        if metrics["start"] is None:
            metrics["start"] = metrics["end"]
        metrics.update({status: 1, **usage})
        return text, 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Outcome reward is computed by HarmonyRewardManager from the final <answer>.
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class CoTAgentTool(MASAgentTool):
    agent_method = "CoTAgent"


class SCAgentTool(MASAgentTool):
    agent_method = "SCAgent"


class ReflexionAgentTool(MASAgentTool):
    agent_method = "ReflexionAgent"


class DebateAgentTool(MASAgentTool):
    agent_method = "DebateAgent"

    async def _run_agent(self, task_info, parameters: dict):
        roles = _parse_debate_roles(parameters.get("debate_roles"))
        if roles is None:
            raise ValueError(
                'DebateAgent requires debate_roles as a list of at least 2 role names, '
                'e.g. ["Mathematics Professor", "Statistician"].'
            )
        return await self._agent_system.DebateAgent(
            agent_input=task_info, model=_RUNTIME_CONFIG["node_model"], debate_roles=roles
        )
