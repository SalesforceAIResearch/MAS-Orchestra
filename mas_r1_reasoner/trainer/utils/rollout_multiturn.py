"""
[ToolOrchestra step 2] Multi-turn (dynamic plan-execute) rollout for MAS-Orchestra.

Single-shot MAS-Orchestra generates one response containing the whole MAS plan, and the plan is executed
afterwards inside the reward manager. Multi-turn mode interleaves planning and execution instead:

    for turn in range(max_turns):                        # actor_rollout_ref.rollout.multi_turn.max_turns
        orchestrator generates <thinking> + tool calls   # loss_mask = 1
        if no tool call: episode ends (final <answer>)
        each tool call runs one MAS-Orchestra agent      # mas_r1_reasoner/tools/mas_agent_tools.py
        agent results are appended as <tool_response>   # loss_mask = 0
    reward = correctness of the final <answer>           # rewards/utils/harmony_parser/multiturn.py

The per-turn loop itself is verl's ``AsyncSGLangRollout._async_rollout_a_request``; this module builds its
inputs (chat messages + per-episode tool kwargs), calls it, and summarises the episodes.
"""

from collections import Counter
from typing import Dict, List

import numpy as np
from omegaconf import OmegaConf

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from mas_r1_reasoner.agents.common import main_rank_print

_TOOL_NAMES_CACHE: Dict[str, List[str]] = {}


def is_multi_turn_enabled(config) -> bool:
    multi_turn = config.actor_rollout_ref.rollout.get("multi_turn", None)
    return bool(multi_turn is not None and multi_turn.get("enable", False))


def get_tool_names(config) -> List[str]:
    """Tool (agent) names registered in the multi-turn tool config file."""
    path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
    if path not in _TOOL_NAMES_CACHE:
        tools_config = OmegaConf.load(path)
        _TOOL_NAMES_CACHE[path] = [t.tool_schema.function.name for t in tools_config.tools]
    return _TOOL_NAMES_CACHE[path]


def _object_array(items: list) -> np.ndarray:
    # np.array(list_of_lists) would build a 2-D array; DataProto needs one object per sample.
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def attach_multiturn_inputs(trainer_instance, gen_batch: DataProto, chat_templates: List[list], questions: List[str]) -> DataProto:
    """
    Add the non-tensor fields AsyncSGLangRollout.generate_sequences_with_tools requires:
      - raw_prompt:   chat messages (system + user) for each sample
      - tools_kwargs: {tool_name: {"create_kwargs": {"question": ...}}} for every agent tool, so each
                      episode gets its own tool instance that knows the original question
    """
    tool_names = get_tool_names(trainer_instance.config)
    tools_kwargs = [{name: {"create_kwargs": {"question": q}} for name in tool_names} for q in questions]
    gen_batch.non_tensor_batch["raw_prompt"] = _object_array(chat_templates)
    gen_batch.non_tensor_batch["tools_kwargs"] = _object_array(tools_kwargs)
    return gen_batch


def multi_turn_rollout_generation(trainer_instance, gen_batch: DataProto, is_validation: bool) -> DataProto:
    """
    Run the dynamic plan-execute rollout. Output has the standard verl multi-turn keys:
    prompts, responses, input_ids, attention_mask, position_ids, loss_mask (+ non-tensor messages).
    """
    rollout_config = trainer_instance.config.actor_rollout_ref.rollout
    meta_info = {
        "eos_token_id": trainer_instance.tokenizer.eos_token_id,
        "pad_token_id": trainer_instance.tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "multi_turn": True,
    }
    if is_validation:
        # Validation prompts are already repeated val_kwargs.n times in MASR1Trainer._validate;
        # the SGLang rollout then samples one episode per prompt.
        meta_info.update({"do_sample": rollout_config.val_kwargs.do_sample, "validate": True})
        gen_batch.meta_info = meta_info
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, trainer_instance.actor_rollout_wg.world_size)
        output = unpad_dataproto(trainer_instance.actor_rollout_wg.generate_sequences(gen_batch_padded), pad_size=pad_size)
    else:
        # Training: the rollout samples rollout.n episodes per prompt (GRPO group).
        # DataProto.pop shares meta_info with the source batch, so rebind instead of mutating it.
        gen_batch.meta_info = {**gen_batch.meta_info, **meta_info}
        output = trainer_instance.actor_rollout_wg.generate_sequences(gen_batch)

    output.meta_info["multi_turn"] = True
    output.meta_info["multi_turn_metrics"] = summarize_episodes(output, prefix="val" if is_validation else "train")
    main_rank_print(f"[multi-turn] {output.meta_info['multi_turn_metrics']}")
    return output


def _message_field(msg, field):
    return msg.get(field) if isinstance(msg, dict) else getattr(msg, field, None)


def summarize_episodes(output: DataProto, prefix: str) -> Dict[str, float]:
    """Turns, agent calls and agent usage per episode, logged as multi_turn/* metrics."""
    messages = output.non_tensor_batch.get("messages", None)
    if messages is None or len(messages) == 0:
        return {}

    turns, calls, no_tool_episodes = [], [], 0
    usage = Counter()
    for episode in messages:
        msgs = episode.get("messages", episode) if isinstance(episode, dict) else episode
        n_turns, n_calls = 0, 0
        for msg in msgs:
            if _message_field(msg, "role") != "assistant":
                continue
            n_turns += 1
            for call in _message_field(msg, "tool_calls") or []:
                function = _message_field(call, "function")
                usage[_message_field(function, "name")] += 1
                n_calls += 1
        turns.append(n_turns)
        calls.append(n_calls)
        no_tool_episodes += int(n_calls == 0)

    num_episodes = len(turns)
    metrics = {
        f"multi_turn/{prefix}_turns_mean": float(np.mean(turns)),
        f"multi_turn/{prefix}_turns_max": float(np.max(turns)),
        f"multi_turn/{prefix}_agent_calls_mean": float(np.mean(calls)),
        f"multi_turn/{prefix}_no_agent_fraction": no_tool_episodes / num_episodes,
    }
    total_calls = max(sum(usage.values()), 1)
    for name, count in usage.items():
        metrics[f"multi_turn/{prefix}_usage_{name}"] = count / total_calls
    return metrics
