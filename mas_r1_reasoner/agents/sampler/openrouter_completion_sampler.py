"""
OpenRouter sampler for the MAS-Orchestra sub-agents (e.g. openai/gpt-oss-120b).

Same prompting, retries and mock mode as TogetherCompletionSampler; only the transport differs:
OpenRouter's OpenAI-compatible endpoint, reasoning effort sent as {"reasoning": {"effort": ...}},
and optional provider routing (e.g. pin "together" to match the original MAS-Orchestra sub-agents).
"""

import os
import threading
from contextvars import ContextVar
from typing import Any, Optional

from openai import AsyncOpenAI

from mas_r1_reasoner.agents.sampler.together_completion_sampler import TogetherCompletionSampler

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-task usage accumulator: a tool sets a fresh dict before running an agent, and every OpenRouter call
# made by that agent (including calls in child tasks, which inherit the context) adds its usage to it.
call_usage: ContextVar[Optional[dict]] = ContextVar("openrouter_call_usage", default=None)


def new_usage() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}


class OpenRouterCompletionSampler(TogetherCompletionSampler):
    """Token usage and cost (USD) of every call are accumulated in OpenRouterCompletionSampler.usage."""

    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}
    _usage_lock = threading.Lock()

    def __init__(
        self,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.5,
        mock_output: bool = False,
        provider: Optional[dict] = None,
    ):
        self.api_key_name = "OPENROUTER_API_KEY"
        api_key = os.environ.get(self.api_key_name)
        if not api_key:
            if not mock_output:
                raise ValueError(f"{self.api_key_name} is not set (e.g. export OPENROUTER_API_KEY=$(cat ~/.openrouter_api_key))")
            api_key = "mock-no-api-calls"
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            timeout=300,
            max_retries=0,  # retries/backoff are handled in TogetherCompletionSampler.__call__
        )
        self.model = self._convert_to_basic_type(model)
        self.system_message = self._convert_to_basic_type(system_message)
        self.temperature = self._convert_to_basic_type(temperature)
        self.no_temperature_models = ['gpt-5-nano']
        self.mock_output = mock_output
        self.provider = self._to_container(provider)

    @staticmethod
    def _to_container(value: Any):
        if value is None:
            return None
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
        except ImportError:
            pass
        return dict(value)

    async def _create_completion(self, api_params: dict):
        params = dict(api_params)
        extra_body = {"usage": {"include": True}}
        reasoning_effort = params.pop("reasoning_effort", None)
        if reasoning_effort:
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if self.provider:
            extra_body["provider"] = self.provider
        if params.get("temperature") is None:
            params.pop("temperature", None)
        response = await self.client.chat.completions.create(**params, extra_body=extra_body)
        self._record_usage(response)
        if response.choices[0].message.content is None:
            # e.g. the reasoning used up max_tokens; the retry loop in __call__ handles the exception
            raise RuntimeError(f"OpenRouter returned no content (finish_reason={response.choices[0].finish_reason})")
        return response

    @classmethod
    def _record_usage(cls, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        delta = {
            "calls": 1,
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
            "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
            "cost": float(getattr(usage, "cost", 0.0) or 0.0),
        }
        task_usage = call_usage.get()
        with cls._usage_lock:
            for key, value in delta.items():
                cls.usage[key] += value
                if task_usage is not None:
                    task_usage[key] += value
