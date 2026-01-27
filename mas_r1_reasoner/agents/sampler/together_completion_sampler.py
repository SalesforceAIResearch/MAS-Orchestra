import base64
import time
import json
import asyncio
from typing import Any

import openai
from openai import APITimeoutError
from together import Together, AsyncTogether

from dataclasses import dataclass, field
from typing import Any
from mas_r1_reasoner.agents.sampler.chat_common import SamplerBase, EvalResult, SingleEvalResult, Eval
from mas_r1_reasoner.agents.shared_vars import get_global

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class TogetherCompletionSampler(SamplerBase):
    """
    Sample from Together AI's chat completion API
    
    Args:
        model: The model to use for completion
        system_message: Optional system message to prepend
        temperature: Sampling temperature (0.0 to 2.0)
        mock_output: If True, returns mock responses instead of calling the API
    """

    def __init__(
        self,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.5,
        mock_output: bool = False,
    ):
        self.api_key_name = "TOGETHER_API_KEY"
        self.client = AsyncTogether(timeout=300)
        # Note: Concurrency is now controlled at the RayAgentWorker level
        
        # Convert OmegaConf objects to basic Python types (these are loaded fron config, so we need to convert them to basic types)
        self.model = self._convert_to_basic_type(model)
        self.system_message = self._convert_to_basic_type(system_message)
        self.temperature = self._convert_to_basic_type(temperature)
        self.no_temperature_models = [
            'gpt-5-nano',  # OpenAI model that doesn't support temperature
            # Add Together AI models that don't support temperature if any
            # Note: Most Together AI models support temperature, but some might have issues
        ]
        self.mock_output = mock_output

    def _convert_to_basic_type(self, value: Any) -> Any:
        """Convert OmegaConf objects to basic Python types."""
        if value is None:
            return None
        
        # If it's already a basic type, return as-is
        if isinstance(value, (str, int, float, bool)):
            return value
        
        # If it's an OmegaConf object, convert to string
        
        try:
            from omegaconf import OmegaConf
            if hasattr(value, '__class__') and 'omegaconf' in str(value.__class__).lower():
                return str(value)
        except ImportError:
            pass
        
        # For other types, convert to string
        return str(value)

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


    async def __call__(self, message_list: MessageList, temperature=None, output_fields=None) -> str:
            
        # print(f"\n=== ChatCompletionSampler.__call__ Debug ===")
        # print(f"Model: {self.model}")
        # print(f"Temperature: {temperature if temperature is not None else self.temperature}")
        # print(f"System message: {self.system_message}")
        # print(f"Input message count: {len(message_list)}")
        
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
            print(f"Added system message, total messages: {len(message_list)}")
        
        trial = 0
        while True:
            print(f"\n--- Async Trial {trial + 1} ---")
            try:
                # Convert non-string content to strings
                # print("Converting message content to strings...")
                for message_id, message in enumerate(message_list):
                    if type(message['content']) != str:
                        original_type = type(message['content']).__name__
                        # print(f"  Converting message {message_id} content from {original_type} to string...")
                        message_list[message_id]['content'] = str(message['content'])
            
                # Ensure all parameters are JSON serializable
                safe_model = self._convert_to_basic_type(self.model)
                safe_temperature = self._convert_to_basic_type(temperature if temperature is not None else self.temperature)
                
                # Ensure temperature is a float
                if safe_temperature is not None:
                    try:
                        safe_temperature = float(safe_temperature)
                        # print(f"✓ Temperature converted to float: {safe_temperature} (type: {type(safe_temperature).__name__})")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Failed to convert temperature '{safe_temperature}' to float: {e}")
                
                print(f"  - Safe model: {safe_model} (type: {type(safe_model).__name__})")
                print(f"  - Safe temperature: {safe_temperature} (type: {type(safe_temperature).__name__})")
                print(f"  - Msg: {message_list}")

                if self.mock_output:
                    content = '<thinking>This is a mock output</thinking><answer>This is a mock answer</answer><correct>True</correct><feedback>This is a mock feedback</feedback>'
                else:
                    if safe_model in self.no_temperature_models:
                        safe_temperature = None
                        # For models that don't support temperature parameter

                    # Get reasoning_effort from global config
                    reasoning_effort = get_global("global_reasoning_effort")
                    if reasoning_effort is None:
                        raise ValueError("Reasoning effort is not set in the global config")
                    
                    # Get max_tokens from global config
                    max_tokens = get_global("global_max_tokens")
                    
                    # For models that support temperature parameter
                    # Build API call parameters
                    api_params = {
                        "model": safe_model,
                        "messages": message_list,
                        "reasoning_effort": reasoning_effort,  # together ai use this ('low', 'medium' or 'high') https://docs.together.ai/docs/gpt-oss#reasoning-effort-control
                        "temperature": safe_temperature,
                        "timeout": 300,
                    }
                    
                    # Only add max_tokens if it's set in the config
                    print(f"  - Max tokens: {max_tokens}, reasoning_effort: {reasoning_effort}")
                    if max_tokens is not None:
                        api_params["max_tokens"] = max_tokens
                        print(f"  - Add Max tokens: {max_tokens}")

                    response = await self.client.chat.completions.create(**api_params)
                    
                    print(f"✓ Async API request successful")
                    content = response.choices[0].message.content
                    # print(f"  - Answer: {content}")

                return content

            except APITimeoutError as e:
                print(f"\n✗ Together AI Async API Timeout Error (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                
                # For timeout errors, retry with exponential backoff
                exception_backoff = 2**trial
                print(f"  - Backoff time: {exception_backoff} seconds")
                print(f"  - Waiting {exception_backoff} seconds before retry...")
                await asyncio.sleep(exception_backoff)
                trial += 1
                
                if trial == 5:  # Max retries reached
                    print(f"\n✗ Max trials reached (5) - API Timeout persisted")
                    print(f"  - Final error type: {type(e).__name__}")
                    print(f"  - Final error message: {e}")
                    print(f"  - Returning empty string")
                    return ""

            except openai.BadRequestError as e:
                print(f"\n✗ Together AI Async Bad Request Error (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Error code: {getattr(e, 'code', 'N/A')}")
                print(f"  - Error status: {getattr(e, 'status', 'N/A')}")
                print(f"  - Error response: {getattr(e, 'response', 'N/A')}")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                print(f"  - Returning empty string")
                return ""
                
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(f"\n✗ Together AI Exception (Trial {trial + 1})")
                print(f"  - Error type: {type(e).__name__}")
                print(f"  - Error message: {e}")
                print(f"  - Error args: {e.args}")
                print(f"  - Backoff time: {exception_backoff} seconds")
                print(f"  - Model used: {self.model}")
                print(f"  - Message count: {len(message_list)}")
                print(f"  - Temperature: {temperature if temperature is not None else self.temperature}")
                
                # Check if it's a rate limit error
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"  - Detected rate limit error")
                elif "timeout" in str(e).lower() or "APITimeoutError" in str(type(e).__name__):
                    print(f"  - Detected timeout error")
                elif "connection" in str(e).lower():
                    print(f"  - Detected connection error")
                else:
                    print(f"  - Unknown error type")
                
                print(f"  - Waiting {exception_backoff} seconds before retry...")
                await asyncio.sleep(exception_backoff)
                trial += 1
                
                if trial == 5: # basically mean it is bad request after 5 trials
                    print(f"\n✗ Max trials reached (5)")
                    print(f"  - Final error type: {type(e).__name__}")
                    print(f"  - Final error message: {e}")
                    print(f"  - Returning empty string")
                    return ""                    
            # unknown error shall throw exception
