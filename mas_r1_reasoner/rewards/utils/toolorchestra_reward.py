"""
ToolOrchestra reward (Su et al., 2025, "ToolOrchestra: Elevating Intelligence via Efficient Model and Tool
Orchestration", Sec. 3; reference implementation: ToolOrchestra/training/lead_agent/llm_agent/generation_quick3.py).

For every episode tau of a question, build the feature vector

    M(tau) = [ calls(tau, agent_1), ..., calls(tau, agent_k),   # how often each agent was used
               r_outcome(tau),                                   # 1 if the final <answer> is correct
               -cost(tau),                                       # $ spent: agent API calls + orchestrator tokens
               -latency(tau) ]                                   # seconds spent waiting on agents

normalise every feature with min-max over the GRPO group of the same question (a feature that is constant in
the group is dropped), and reward

    R(tau) = P . M_norm(tau)   if r_outcome(tau) = 1,   else 0

where P is the user preference vector (azr.mas_r1.toolorchestra_reward.preference). GRPO then standardises R
within the group as usual.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from mas_r1_reasoner.agents.common import main_rank_print


def reward_config(config) -> Optional[dict]:
    """The toolorchestra_reward block if it is enabled, else None."""
    block = config.azr.mas_r1.get("toolorchestra_reward", None)
    if block is None or not block.get("enable", False):
        return None
    from omegaconf import OmegaConf
    return OmegaConf.to_container(block, resolve=True)


def _busy_seconds(intervals: List[tuple]) -> float:
    """Length of the union of [start, end] intervals: parallel agent calls in one turn count once."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def episode_stats_from_tool_metrics(tool_metrics: Optional[dict]) -> dict:
    """Aggregate the per-call metrics that MASAgentTool.execute returns into per-episode statistics."""
    calls, intervals, agent_cost, agent_tokens = defaultdict(int), [], 0.0, 0
    for tool_name, call_metrics in (tool_metrics or {}).items():
        for m in call_metrics:
            calls[tool_name] += 1
            if m.get("start") is not None and m.get("end") is not None:
                intervals.append((float(m["start"]), float(m["end"])))
            agent_cost += float(m.get("cost", 0.0) or 0.0)
            agent_tokens += int(m.get("prompt_tokens", 0) or 0) + int(m.get("completion_tokens", 0) or 0)
    return {"calls": dict(calls), "agent_cost": agent_cost, "agent_tokens": agent_tokens, "latency": _busy_seconds(intervals)}


def attach_episode_stats(trainer_instance, question_results: Dict, generation_output, is_validation: bool) -> None:
    """
    Store episode statistics under the same {question}<<MySep>>{idx} keys that
    extraction.generate_and_extract_codes uses, so they travel with execution_stats through reformatting.
    """
    tool_metrics = generation_output.non_tensor_batch.get("tool_metrics", None)
    if tool_metrics is None:
        return
    response_count = defaultdict(int)
    for generation_idx in range(len(generation_output)):
        question = trainer_instance.processor.extract_math_question(generation_output, generation_idx)
        if is_validation:
            key = f"{question}<<MySep>>{generation_idx}"
        else:
            key = f"{question}<<MySep>>{response_count[question]}"
            response_count[question] += 1
        if key in question_results:
            question_results[key]["episode_stats"] = episode_stats_from_tool_metrics(tool_metrics[generation_idx])


def _orchestrator_cost(data_item, prices: dict) -> float:
    """Orchestrator tokens priced like a model API: prompt + agent results are input, generated turns are output."""
    prompt_length = data_item.batch["prompts"].shape[-1]
    attention = data_item.batch["attention_mask"]
    prompt_tokens = int(attention[:prompt_length].sum())
    response_tokens = int(attention[prompt_length:].sum())
    generated = int(data_item.batch["loss_mask"][prompt_length:].sum()) if "loss_mask" in data_item.batch else response_tokens
    input_tokens = prompt_tokens + (response_tokens - generated)
    return (input_tokens * prices.get("input", 0.0) + generated * prices.get("output", 0.0)) / 1e6


def compute_toolorchestra_reward(data, reward_tensor: torch.Tensor, correctness: List[float], cfg: dict,
                                 apply: bool, prefix: str):
    """
    Returns (reward_tensor, metrics). With apply=False (validation) the outcome reward is kept and only
    the cost / latency / usage metrics are reported.
    """
    n = len(data)
    execution_stats = data.non_tensor_batch.get("execution_stats", [None] * n)
    preference = cfg.get("preference", {})
    prices = cfg.get("orchestrator_price_per_million", {})

    episodes = []
    for i in range(n):
        stats = (execution_stats[i] or {}).get("episode_stats") or {"calls": {}, "agent_cost": 0.0, "agent_tokens": 0, "latency": 0.0}
        item = data[i]
        cost = stats["agent_cost"] + _orchestrator_cost(item, prices)
        episodes.append({
            "group": (execution_stats[i] or {}).get("question", i),
            "correct": float(correctness[i] or 0.0) > 0.5,
            "calls": stats["calls"],
            "cost": cost,
            "agent_cost": stats["agent_cost"],
            "latency": stats["latency"],
        })

    agents = sorted({a for e in episodes for a in e["calls"]} | {k for k in preference if k not in ("accuracy", "cost", "latency")})

    def features(e):
        f = {a: float(e["calls"].get(a, 0)) for a in agents}
        f.update({"accuracy": float(e["correct"]), "cost": -e["cost"], "latency": -e["latency"]})
        return f

    groups = defaultdict(list)
    for i, e in enumerate(episodes):
        groups[e["group"]].append(i)

    rewards = [0.0] * n
    for members in groups.values():
        feats = [features(episodes[i]) for i in members]
        lo = {k: min(f[k] for f in feats) for k in feats[0]}
        hi = {k: max(f[k] for f in feats) for k in feats[0]}
        for i, f in zip(members, feats):
            if not episodes[i]["correct"]:
                continue
            rewards[i] = sum(preference.get(k, 0.0) * (f[k] - lo[k]) / (hi[k] - lo[k]) for k in f if hi[k] > lo[k])

    if apply:
        prompt_length = data.batch["prompts"].shape[-1]
        reward_tensor = torch.zeros_like(reward_tensor)
        for i in range(n):
            valid_response_length = int(data.batch["attention_mask"][i, prompt_length:].sum())
            reward_tensor[i, max(valid_response_length - 1, 0)] = rewards[i]

    metrics = {
        f"toolorchestra/{prefix}_accuracy": float(np.mean([e["correct"] for e in episodes])),
        f"toolorchestra/{prefix}_cost_usd_mean": float(np.mean([e["cost"] for e in episodes])),
        f"toolorchestra/{prefix}_agent_cost_usd_mean": float(np.mean([e["agent_cost"] for e in episodes])),
        f"toolorchestra/{prefix}_latency_s_mean": float(np.mean([e["latency"] for e in episodes])),
        f"toolorchestra/{prefix}_cost_usd_total": float(np.sum([e["cost"] for e in episodes])),
    }
    for a in agents:
        metrics[f"toolorchestra/{prefix}_calls_{a}_mean"] = float(np.mean([e["calls"].get(a, 0) for e in episodes]))
    if apply:
        metrics[f"toolorchestra/{prefix}_reward_mean"] = float(np.mean(rewards))
        correct_rewards = [r for r, e in zip(rewards, episodes) if e["correct"]]
        if correct_rewards:
            metrics[f"toolorchestra/{prefix}_reward_correct_mean"] = float(np.mean(correct_rewards))
    main_rank_print(f"[ToolOrchestra reward] {metrics}")
    return reward_tensor, metrics
