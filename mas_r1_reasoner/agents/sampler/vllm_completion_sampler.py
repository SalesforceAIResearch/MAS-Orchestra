import base64
import time
import asyncio
from typing import Any

import openai
from openai import OpenAI, AsyncOpenAI, APITimeoutError

from dataclasses import dataclass, field
from typing import Any
import re
import json
from collections import OrderedDict
import os

from mas_r1_reasoner.agents.sampler.chat_common import SamplerBase, EvalResult, SingleEvalResult, Eval

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


# TODO: some error here

class VLLMCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        system_message: str | None = None,
        temperature: float = 0.5,
        model: str | None = None,
    ):        
    
        try:
            model_api_map = {
                'qwen-2.5-32b-instr': '8082'
            }
            self.api_key_name = "OPENAI_API_KEY"
            self.system_message = system_message
            self.temperature = temperature
            openai_api_key = "EMPTY"

            base_port = os.getenv("BASE_PORT")
            if base_port:
                openai_api_base = f"http://localhost:{base_port}/v1"
            else:
                openai_api_base = f"http://localhost:{model_api_map[model]}/v1"

            self.client = AsyncOpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            models = self.client.models.list()
            self.model = models.data[0].id     

        except Exception as e:
            print(f'warning VLLM: {e}')
            

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    async def __call__(self, message_list: MessageList, temperature=None, response_format=None) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        message_list[message_id]['content'] = str(message['content'])

                # Prepare parameters
                safe_temperature = float(temperature if temperature is not None else self.temperature)
                
                # Make API call
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=safe_temperature,
                    timeout=60
                )
                # print(f"✓ API request successful {response}")

                return response.choices[0].message.content

            except APITimeoutError as e:
                print(f"VLLM: Timeout error (trial {trial + 1}): {e}")
                if trial >= 4:
                    return ""
                await asyncio.sleep(2 ** trial)
                trial += 1

            except openai.BadRequestError as e:
                print(f"VLLM: Bad request error: {e}")
                return ""
                
            except Exception as e:
                print(f"VLLM: Error (trial {trial + 1}): {e}")
                # Only clear cache on persistent failures
                if trial >= 2:
                    print("VLLM: Clearing cache due to persistent failures")
                    self.available_services = {}
                    self.client_cache = {}
                if trial >= 4:
                    return ""
                await asyncio.sleep(2 ** trial)
                trial += 1