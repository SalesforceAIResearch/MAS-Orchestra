import string
import copy
import os
import random
import numpy as np
import re
import time
import json
import asyncio
from typing import Any, Optional, Dict, List
import torch
from verl import DataProto
from mas_r1_reasoner.agents.common import random_id, extract_xml
from mas_r1_reasoner.agents.shared_vars import get_global
from collections import namedtuple

# Import harmony block functions (regular)
from mas_r1_reasoner.agents.blocks_harmony.cot import CoTAgent as cot_func
from mas_r1_reasoner.agents.blocks_harmony.cot_sc import SCAgent as cot_sc_func
from mas_r1_reasoner.agents.blocks_harmony.llm_debate import DebateAgent as debate_func
from mas_r1_reasoner.agents.blocks_harmony.reflexion import ReflexionAgent as refinement_func
# Import IGSM harmony block functions
from mas_r1_reasoner.agents.blocks_harmony.igsm.cot import CoTAgent as cot_func_igsm
from mas_r1_reasoner.agents.blocks_harmony.igsm.cot_sc import SCAgent as cot_sc_func_igsm
from mas_r1_reasoner.agents.blocks_harmony.igsm.llm_debate import DebateAgent as debate_func_igsm
from mas_r1_reasoner.agents.blocks_harmony.igsm.reflexion import ReflexionAgent as refinement_func_igsm
# Note: WebSearch is imported dynamically in WebSearchAgent() method based on global_web_search_type

# Global Info namedtuple type used across the MAS-R1 system
Info = namedtuple('Info', ['name', 'author', 'content', 'msg', 'sub_tasks', 'agents', 'iteration_idx', 'final_answer'])


class LLMAgentBase():
    """
    LLM Agent Base class for MAS-R1 system.
    Handles prompt construction and LLM calls using independent LLM client.
    """
    

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model: str=None, temperature: int=None, system_prompt: str=None) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model or get_global("global_node_model")
        self.temperature = temperature
        # give each instance a unique id
        self.id = random_id()
        self.system_prompt = system_prompt

    def extract_pattern(self, msg):
        # pattern = r"\s*(.*?)\s*\n\nRelated original question"
        pattern = r"Given the above, answer the following question: \s*(.*?)\s*\n\n"

        # we use 'msg' from the message that contain both systme prompt and user prompt
        # msg[-1]['content'] means take the last user prompt's content
        sub_question = msg[-1]['content'] 
        match = re.search(pattern, sub_question, re.DOTALL)
        extracted_question = match.group(1)

        return extracted_question


    def generate_prompt(self, input_infos, instruction, is_sub_task=False) -> str:
        # Note: This is worker agent context management

        global_output_description = get_global("global_output_description")
        global_FORMAT_INST = get_global("global_FORMAT_INST")

        output_fields_and_description = '\n'.join([f"<{key}> [Your {key}.] </{key}>" if not 'answer' in key else f"<{key}> [Your {key}. {global_output_description}] </{key}>\n" for key in self.output_fields])

        ROLE_DESC = lambda role: f"You are a {role}."

        system_prompt = ROLE_DESC(self.role) + "\n\n" + global_FORMAT_INST(output_fields_and_description)
        
        # construct input infos text
        input_infos_text = ''
        prev_extracted_question = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, msg, _, _, iteration_idx, _) = input_info
            else:
                raise ValueError(f"input_info is not an Info object: {input_info}. This may cause hidden error, please update the input")
                
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task: 
                    input_infos_text += f'Related original question:\n\n{content}. \n\nRelated sub-task questions and answers:\n\n'
                else:
                    input_infos_text += f'{content}\n\n'
            elif iteration_idx != -1:
                if is_sub_task and msg is not None: 
                    extracted_question = self.extract_pattern(msg)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question
                    else:
                        input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'

                else:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                if is_sub_task and msg is not None: 
                    extracted_question = self.extract_pattern(msg)
                    if extracted_question != prev_extracted_question:
                        input_infos_text += f'### {extracted_question} \n\n ### {field_name} by {author}:\n{content}\n\n'
                        prev_extracted_question = extracted_question # we do not want to duplicate the prompt
                    else:
                        input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'
                else:
                    input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        if is_sub_task: 

            prompt = input_infos_text + f"""Given the above, answer the following question: {instruction}\n\n 
            
            If the question is too complicated or informaion is missing, you still need to give your best guess but add (1) an additional mark [TOO_HARD] in the next line of your final answer (2) information request or decomposison suggestion in the next line of the [TOO_HARD] mark, in the "answer" entry. In the "thinking", justify why you think so. Following the format below:
            
            "answer" entry: [Your best guess, e.g., 300]\n[TOO_HARD]\nSuggestion: [your suggestion]
            "thinking" entry:  [why you thinking is is too complicated or missing information. How to you arrive your best guess regardless]

            Otherwise, give your answer and thinking normally.

            "answer" entry: [your answer]
            "thinking" entry: [How do you arrive your answer]

            IMPORTANT: You need to give your best guess in both cases. Do not give [TOO_HARD] directly but always give your best guess first

            """


        else:
            prompt = input_infos_text + instruction
        return system_prompt, prompt

    async def query(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False) -> dict:

        def _pack_message(role: str, content: Any):
            return {"role": str(role), "content": content}

        system_prompt, prompt = self.generate_prompt(input_infos, instruction, is_sub_task=is_sub_task)

        if self.system_prompt is not None:
            assert False, "system_prompt is not supported for now"

        else:
            msg = [
                _pack_message(content=system_prompt, role="system"),
                _pack_message(content=prompt, role="user")]
            # use system prompt

        # print(f"[DEBUG] msg: {msg}")

        response_json = await self.get_response_from_agent(msg, self.output_fields)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, msg, None, None, iteration_idx, None)
            output_infos.append(info)
        
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False):
        # Note: This is now async
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx,  is_sub_task=is_sub_task)



    async def get_response_from_agent(self, msg: List[Dict[str, str]], output_fields: List[str]) -> Dict[str, str]:
        """
        Call the agent (LLM) using independent LLM client.
        
        Args:
            msg: List of message dictionaries with 'role' and 'content' keys
            output_fields: List of expected output fields
            temperature: Temperature for generation (optional)
            
        Returns:
            Dictionary with agent responses for each output field
        """
        # Use provided temperature or default
        temp = self.temperature
        
        # Use regular model sampler map
        model_sampler_map = get_global("global_model_sampler_map")
        if self.model not in model_sampler_map:
            raise RuntimeError(f"Model '{self.model}' not found in global_model_sampler_map. Available models: {list(model_sampler_map.keys())}")
        
        sampler = model_sampler_map[self.model]
        print(f"✓ Regular model sampler found for '{self.model}'")
        
        
        max_retries = 5
        debug_count = 0
        response_text = ""  # Initialize response_text to avoid reference error
        all_errors = []  # Track all errors for detailed reporting
        
        while debug_count < max_retries:  # Limit retries to 3 attempts
            debug_count += 1
            try:
                # print(f"\n--- Attempt {debug_count}/{max_retries} ---")
                # print(f"Calling sampler with model: {self.model}")
                # print(f"Message list: {msg}")
                # print(f"Temperature: {temp}")
                
                # Call the async sampler directly
                response_text = await sampler(msg, temp, output_fields)

                # Check if response_text is already valid JSON with required fields
                try:
                    response_json = json.loads(response_text)
                    if isinstance(response_json, dict):
                        # Check if all required fields are present
                        has_all_fields = all(field in response_json for field in output_fields)
                        if has_all_fields:
                            # Check if answer field is valid (same logic as XML validation)
                            if 'answer' in output_fields: # make it a bit more robust
                                answer_content = response_json.get('answer', '')
                                if not answer_content:
                                    # If answer field is empty, use thinking instead
                                    if 'thinking' in response_json:
                                        response_json['answer'] = response_json['thinking']
                                        print(f"✓ Answer field was empty, used thinking field instead")
                                    else:
                                        print("✗ Answer field is empty and no thinking field available")
                                        # Fall through to XML processing
                                        raise ValueError("Empty answer field with no thinking fallback")
                            
                            print(f"✓ Response is already valid JSON with required fields: {list(response_json.keys())}")
                            print(f"Parsed response: {response_json}")
                            print(f"{'='*50}\n")
                            return response_json
                except (json.JSONDecodeError, TypeError):
                    # Not valid JSON, continue with XML processing
                    print("Not a Json. Continue with XML processing")

                # print(f"Independent LLM response received!")
                # print(f"Response length: {len(response_text)} characters")
                # print(f"Response preview: {response_text[:200]}...")

                # Parse response into output fields using XML extraction
                response_dict = {}

                # Validate XML response instead of JSON
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
                if 'answer' in output_fields:
                    answer_content = extract_xml(response_text, 'answer')
                    if not answer_content or len(answer_content.strip()) == 0:
                        is_valid_response = False
                        print("✗ Answer field is empty")

                if is_valid_response:
                    print(f"Parsed response: {response_dict}")
                    print(f"{'='*50}\n")
                    return response_dict
                else: # TODO: we may not need it with reponse AI gurentee the output format
                    print(f'Invalid XML response. Required fields: {output_fields}, response: {response_text}, recall LLM with clearer instructions (Attempt {debug_count})')
                    
                    extra_instr = '\n'.join([f"<{key}> [Your {key}.] </{key}>" for key in self.output_fields])
                    msg[-1]['content'] += f'Reply EXACTLY with the following XML format.\n{extra_instr}\n\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!'

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

        # Raise error if all retries failed with detailed information
        error_summary = "\n".join(all_errors)
        raise RuntimeError(f"Failed to get valid response from agent after {debug_count} attempts. All retries failed.\n\nDetailed errors:\n{error_summary}\n\nFinal response text: {response_text if response_text else 'No response received'}")


class AgentSystem():
    """
    Agent System for executing generated MAS code.
    The generated code contains a forward function that gets inserted into this class.
    Uses simple namespace approach with setattr for contamination-free execution.
    """
    
    def __init__(self, agent_config: Dict[str, Any] = None) -> None:
        print(f"\n{'='*60}")
        print("INITIALIZING AGENT SYSTEM")
        print(f"{'='*60}")

        # Load global variables
        self.node_model = get_global("global_node_model")
        self.cot_instruction = get_global("global_cot_instruction") 
        # Not needed
        self.max_sc = get_global("global_max_sc")
        self.max_round = get_global("global_max_round")
        self.debate_role = get_global("global_debate_role")
        self.retrieval_method = get_global("global_retrieval_method")
        self.max_tokens = get_global("global_max_tokens")

        print(f"✓ Global variables loaded:")
        print(f"  - node_model: {self.node_model}")
        # print(f"  - cot_instruction: {self.cot_instruction}")
        print(f"  - max_sc: {self.max_sc} (type: {type(self.max_sc)})")
        print(f"  - max_round: {self.max_round}")
        print(f"  - debate_role: {self.debate_role}")
        print(f"  - retrieval_method: {self.retrieval_method}")
        print(f"  - max_tokens: {self.max_tokens}")
        print(f"{'='*60}\n")
        
        # Online search resources (loaded once for online web search)
        self.online_chat_model = None
        
        # Offline search resources (loaded once automatically since AgentSystem only initializes once)
        self.offline_corpus = None
        self.offline_chat_model = None
        self.offline_retriever = None  # BM25 or dense retriever
        
        # Hybrid cache: in-memory (fast) + SQLite (persistent, cross-worker)
        self.offline_summary_cache = {}  # In-memory cache for fast access
        self.offline_cache_db = None     # SQLite connection for persistent storage
        
        # Thread lock for cache safety (Ray workers can handle multiple tasks concurrently)
        import threading
        self.offline_cache_lock = threading.Lock()
        
        # Initialize web search resources based on mode
        self._initialize_web_search_resources()
    
    def _initialize_web_search_resources(self):
        """Initialize web search resources (online or offline) based on web_search_type"""
        try:
            from mas_r1_reasoner.agents.shared_vars import get_global
            from mas_r1_reasoner.agents.blocks_harmony.initialization import (
                initialize_online_resources,
                initialize_offline_resources
            )
            
            web_search_type = get_global("global_web_search_type")
            
            if web_search_type == "online":
                # Initialize online chat model for web search summarization
                initialize_online_resources(self)
            elif web_search_type == "offline":
                # Initialize offline resources (corpus, chat model, retriever, cache)
                initialize_offline_resources(self)
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not initialize web search resources: {e}")

    def set_instance_forward_function(self, forward_str: str):
        """
        Set the async forward function for this specific instance only.
        This prevents contamination by keeping functions isolated per instance.
        
        Args:
            forward_str: The async forward function code string
        """
        # print(f"\n{'='*50}")
        # print("AgentSystem: Setting instance forward function")
        # print(f"{'='*50}")
        # print(f"Code length: {len(forward_str)} characters")
        # print(f"Code preview: {forward_str[:200]}...")
        
        try:
            # Clean up the code string to handle indentation issues
            cleaned_code = forward_str.strip()
            
            namespace = {}
            exec(cleaned_code, globals(), namespace)
            names = list(namespace.keys())
            if len(names) != 1:  # Only the forward function
                error_msg = f"{len(names)} things in namespace. Expected 1 (forward function), but found: {names}. forward_str: {forward_str}. cleaned_code: {cleaned_code}"
                raise AssertionError(error_msg)
            
            # Find the forward function
            func = None
            for name, obj in namespace.items():
                if callable(obj):
                    func = obj
                    break
            
            if func is None:
                raise AssertionError(f"No callable forward function found in namespace. Available: {names}")
            
            # Bind the function directly to this instance
            bound_func = func.__get__(self, type(self))
            
            # Set the forward function as an instance method only (now async by default)
            setattr(self, "forward", bound_func)
            
            print(f"✓ Forward function successfully set as instance method!")
            # print(f"Function: {bound_func}")
                
        except Exception as e:
            print(f"✗ Error setting forward function: {e}")
            raise ValueError(f"Failed to set forward function: {e}")
        
        print(f"{'='*50}\n")

    # Majority voting function to select the most common answer
    def majority_voting(self, answers):
        from collections import Counter
        return Counter(answers).most_common(1)[0][0]
    
    # Harmony block functions - now async
    async def CoTAgent(self, agent_input, model):
        """Chain-of-Thought (CoT) approach"""
        # Dynamically get the correct function based on use_igsm_prompt
        use_igsm_prompt = get_global("global_use_igsm_prompt")
        if use_igsm_prompt:
            print(f"Using async IGSM CoTAgent")
            return await cot_func_igsm(self, agent_input, model)
        else:
            return await cot_func(self, agent_input, model)

    async def SCAgent(self, agent_input, model):
        """Self-Consistency with Chain-of-Thought (CoT_SC)"""
        # Dynamically get the correct function based on use_igsm_prompt
        use_igsm_prompt = get_global("global_use_igsm_prompt")
        if use_igsm_prompt:
            return await cot_sc_func_igsm(self, agent_input, model)
        else:
            return await cot_sc_func(self, agent_input, model)

    async def DebateAgent(self, agent_input, model, debate_roles):
        """LLM Debate (collaborative)"""
        # Dynamically get the correct function based on use_igsm_prompt
        use_igsm_prompt = get_global("global_use_igsm_prompt")
        if use_igsm_prompt:
            return await debate_func_igsm(self, agent_input, model, debate_roles)
        else:
            return await debate_func(self, agent_input, model, debate_roles)

    async def ReflexionAgent(self, agent_input, model):
        """Self-Refine (Reflexion)"""
        # Dynamically get the correct function based on use_igsm_prompt
        use_igsm_prompt = get_global("global_use_igsm_prompt")
        if use_igsm_prompt:
            return await refinement_func_igsm(self, agent_input, model)
        else:
            return await refinement_func(self, agent_input, model)

    async def WebSearchAgent(self, agent_input, model):
        """Async Web Search Agent with reasoning - uses online or offline based on global_web_search_type"""
        # Dynamically get the correct function based on current global setting
        web_search_type = get_global("global_web_search_type")
        
        if web_search_type == "offline":
            print(f"Using async offline web search")
            from mas_r1_reasoner.agents.blocks_harmony.web_search_offline import WebSearchOfflineAgent
            return await WebSearchOfflineAgent(self, agent_input, model)
        else:
            print(f"Using async online web search")
            from mas_r1_reasoner.agents.blocks_harmony.web_search import WebSearchAgent
            return await WebSearchAgent(self, agent_input, model)





    def make_final_answer(self, thinking, answer, sub_tasks=None, agents=None):
        name = thinking.name
        author = thinking.author
        msg = thinking.msg
        iteration_idx = thinking.iteration_idx

        if type(answer) == str:
            answer_content = answer
        else:
            answer_content = answer.content

        if agents is None: # this means sub_task is None, according to the propose prompt
            sub_tasks, agents = agents, sub_tasks

        if sub_tasks is None and agents is None:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', msg, None, None, iteration_idx, answer_content)
        elif agents is not None: # when remove decomposition, we still have agent output logged
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', msg, None, '\n'.join(agents), iteration_idx, answer_content)
        else:
            final_answer = Info(name, author, f'{thinking.content}\n\nAnswer:{answer_content}', msg, '\n'.join(sub_tasks), '\n'.join(agents), iteration_idx, answer_content)
            
        return final_answer