"""
Check the OpenRouter setup with one real gpt-oss-120b call through the same sampler the agents use.

Usage: OPENROUTER_API_KEY=$(cat ~/.openrouter_api_key) python scripts/check_openrouter.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mas_r1_reasoner.agents.shared_vars import set_global
from mas_r1_reasoner.agents.sampler.openrouter_completion_sampler import OpenRouterCompletionSampler


async def main():
    set_global("global_reasoning_effort", "low")
    set_global("global_max_tokens", None)
    sampler = OpenRouterCompletionSampler(
        model="openai/gpt-oss-120b",
        provider={"order": ["together"], "allow_fallbacks": True},
    )
    start = time.time()
    content = await sampler([{"role": "user", "content": "What is 17 * 23? Reply with <answer>NUMBER</answer> only."}])
    print(f"\nResponse ({time.time() - start:.1f}s): {content!r}")
    print(f"Usage: {OpenRouterCompletionSampler.usage}")
    if not content or "391" not in content:
        sys.exit("OpenRouter check FAILED (see the error messages above)")
    print("OpenRouter check OK")


if __name__ == "__main__":
    asyncio.run(main())
