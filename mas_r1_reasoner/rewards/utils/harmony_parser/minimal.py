# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Extraction and generation utilities for Harmony MAS-R1 Trainer.
"""

from typing import Dict, Tuple, Any, List
from verl.protocol import DataProto
from mas_r1_reasoner.agents.common import main_rank_print, extract_xml
from mas_r1_reasoner.agents.shared_vars import get_global
import json
import ast


def parse_harmony_agent_response(response_text: str) -> str:
    """
    Parse harmony agent response and convert it to executable code.
    
    Args:
        response_text: The harmony response text containing agent definitions
        
    Returns:
        str: Generated code as a forward() function, or empty string if no agent found
    """
    # Check if agent_name is in the response
    agent_name = extract_xml(response_text, "agent_name")
    if not agent_name:
        main_rank_print("No agent_name found in harmony response, returning empty code")
        return ""
    
    try:
        # Extract all agent components
        agent_description = extract_xml(response_text, "agent_description")
        required_arguments_xml = extract_xml(response_text, "required_arguments")
        agent_output_id = extract_xml(response_text, "agent_output_id")
        no_decompose = get_global("global_no_decompose")
        #TODO: agent_output_id is not deal with yet
        #TODO: now only work for minimal
        
        # Parse the required_arguments directly
        try:
            # Debug: Print the raw required_arguments content
            main_rank_print(f"Raw required_arguments: {repr(required_arguments_xml)}")
            main_rank_print(f"required_arguments length: {len(required_arguments_xml)}")
            
            if not required_arguments_xml:
                main_rank_print("Failed to extract required_arguments from response")
                return ""
            
            # Parse required_arguments - extract all XML tags within it
            required_arguments = {}
            
            # Extract agent_input
            agent_input = extract_xml(required_arguments_xml, "agent_input")
            # agent_input can be:
            # - "" (two quotes as a string literal) -> means use original question, store as empty string
            # - empty string or not found -> means use original question, store as empty string
            # - actual text -> use as-is
            if agent_input == '""' or not agent_input or no_decompose: #if no_decompose, always use original question
                # Empty or "" means "use original question" - store as empty string
                required_arguments['agent_input'] = ''
            else:
                # Use the extracted value as-is (actual text)
                required_arguments['agent_input'] = agent_input
            
            # Extract debate_roles if present (for DebateAgent)
            debate_roles = extract_xml(required_arguments_xml, "debate_roles")
            if debate_roles:  # Only add if not empty
                # Parse debate_roles string representation of list into actual list
                parsed_roles = None
                try:
                    parsed_roles = ast.literal_eval(debate_roles)
                except (ValueError, SyntaxError):
                    # If ast.literal_eval fails, try json.loads as fallback
                    try:
                        parsed_roles = json.loads(debate_roles)
                    except json.JSONDecodeError:
                        main_rank_print(f"Failed to parse debate_roles as list: {debate_roles}")
                
                # Validate debate_roles is a proper list for DebateAgent
                if agent_name.lower() == 'debateagent':
                    if parsed_roles is None or not isinstance(parsed_roles, list):
                        raise ValueError(
                            f"DebateAgent requires debate_roles to be a valid list, got: {debate_roles}. "
                            f"Expected format: [\"Role1\", \"Role2\", ...]. "
                            f"Make sure to output debate_roles as a properly formatted list."
                        )
                    if len(parsed_roles) < 2:
                        raise ValueError(
                            f"DebateAgent requires at least 2 roles in debate_roles, got {len(parsed_roles)}: {parsed_roles}. "
                            f"Expected format: [\"Role1\", \"Role2\", ...]"
                        )
                    # Additional check: ensure all elements are strings (not single characters)
                    if any(not isinstance(role, str) or len(role) < 2 for role in parsed_roles):
                        raise ValueError(
                            f"DebateAgent debate_roles contains invalid role names: {parsed_roles}. "
                            f"Each role must be a meaningful string (not single characters). "
                            f"Got: {debate_roles}"
                        )
                    required_arguments['debate_roles'] = parsed_roles
                else:
                    # For non-DebateAgents, store as-is (shouldn't happen but be safe)
                    if parsed_roles is not None:
                        required_arguments['debate_roles'] = parsed_roles
                    else:
                        required_arguments['debate_roles'] = debate_roles
            
            # Construct agent_calls in the expected format using agent_name as the agent name
            agent_calls = [{
                'name': agent_name,
                'required_arguments': required_arguments
            }]
            
            main_rank_print(f"Constructed agent_calls: {json.dumps(agent_calls, indent=2)}")
            
        except Exception as e:
            main_rank_print(f"Failed to parse required_arguments XML: {required_arguments_xml}")
            main_rank_print(f"XML Parsing Error: {e}")
            return ""
        

        # Generate the forward function code
        global_node_model = get_global('global_node_model')

        code_lines = [
            "async def forward(self, original_task_info):",
            "    \"\"\"Generated harmony agent forward function\"\"\"",
            "    # Agent configuration",
            f"    agent_name = \"\"\"{agent_name}\"\"\"",
            f"    agent_description = \"\"\"{agent_description}\"\"\"",
            f"    agent_output_id = \"\"\"{agent_output_id}\"\"\"",
            "",
            "    # Agent calls configuration",
            f"    agent_calls = {repr(agent_calls)}",
            "",
            "    # Execute agent calls based on agent name",
            "    agent_name = agent_calls[0]['name'] if agent_calls else ''",
            "    agent_name_lower = agent_name.lower()",
            "    if agent_name_lower == 'scagent':",
            "        # Execute SCAgent",
            "        for agent_call in agent_calls:",
            "            if agent_call['name'] == 'SCAgent':",
            "                args = agent_call['required_arguments']",
            "                agent_input = args.get('agent_input')",
            "                # Use original_task_info if agent_input is empty, otherwise combine",
            "                if not agent_input or not agent_input.strip():",
            "                    task_info = original_task_info",
            "                else:",
            "                    combined_content = f'Original task: {original_task_info.content}; Sub-task: {agent_input}. Please solve the sub-task first. If its result directly answers the original question, output that as the final answer. Otherwise, use the sub-task result as guidance to complete the original question and output the final answer to the original question.'",
            "                    task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
            "                result = await self.SCAgent(",
            "                    agent_input=task_info,",
            f"                    model='{global_node_model}'",
            "                )",
            "                return result",
            "    elif agent_name_lower == 'cotagent':",
            "        # Execute CoTAgent",
            "        for agent_call in agent_calls:",
            "            if agent_call['name'] == 'CoTAgent':",
            "                args = agent_call['required_arguments']",
            "                agent_input = args.get('agent_input')",
            "                # Use original_task_info if agent_input is empty, otherwise combine",
            "                if not agent_input or not agent_input.strip():",
            "                    task_info = original_task_info",
            "                else:",
            "                    combined_content = f'Original task: {original_task_info.content}; Sub-task: {agent_input}. Please solve the sub-task first. If its result directly answers the original question, output that as the final answer. Otherwise, use the sub-task result as guidance to complete the original question and output the final answer to the original question.'",
            "                    task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
            "                result = await self.CoTAgent(",
            "                    agent_input=task_info,",
            f"                    model='{global_node_model}'",
            "                )",
            "                return result",
            "    elif agent_name_lower == 'reflexionagent':",
            "        # Execute ReflexionAgent",
            "        for agent_call in agent_calls:",
            "            if agent_call['name'] == 'ReflexionAgent':",
            "                args = agent_call['required_arguments']",
            "                agent_input = args.get('agent_input')",
            "                # Use original_task_info if agent_input is empty, otherwise combine",
            "                if not agent_input or not agent_input.strip():",
            "                    task_info = original_task_info",
            "                else:",
            "                    combined_content = f'Original task: {original_task_info.content}; Sub-task: {agent_input}. Please solve the sub-task first. If its result directly answers the original question, output that as the final answer. Otherwise, use the sub-task result as guidance to complete the original question and output the final answer to the original question.'",
            "                    task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
            "                result = await self.ReflexionAgent(",
            "                    agent_input=task_info,",
            f"                    model='{global_node_model}'",
            "                )",
            "                return result",
            "    elif agent_name_lower == 'debateagent':",
            "        # Execute DebateAgent",
            "        for agent_call in agent_calls:",
            "            if agent_call['name'] == 'DebateAgent':",
            "                args = agent_call['required_arguments']",
            "                agent_input = args.get('agent_input')",
            "                # Use original_task_info if agent_input is empty, otherwise combine",
            "                if not agent_input or not agent_input.strip():",
            "                    task_info = original_task_info",
            "                else:",
            "                    combined_content = f'Original task: {original_task_info.content}; Sub-task: {agent_input}. Please solve the sub-task first. If its result directly answers the original question, output that as the final answer. Otherwise, use the sub-task result as guidance to complete the original question and output the final answer to the original question.'",
            "                    task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
            "                result = await self.DebateAgent(",
            "                    agent_input=task_info,",
            f"                    model='{global_node_model}',",
            "                    debate_roles=args.get('debate_roles')",
            "                )",
            "                return result",
            "    else:",
            "        # Unknown agent type - raise error",
            "        raise ValueError(f\"Unknown agent name: {agent_name}. Supported agent names: 'CoTAgent', 'SCAgent', 'ReflexionAgent', 'DebateAgent' (case insensitive)\")",
        ]
        
        generated_code = "\n".join(code_lines)
        
        # Log the generated code after parsing
        main_rank_print(f"\n{'='*80}")
        main_rank_print(f"GENERATED CODE FOR AGENT: {agent_name}")
        main_rank_print(f"{'='*80}")
        main_rank_print(generated_code)
        main_rank_print(f"{'='*80}\n")
        
        return generated_code
        
    except Exception as e:
        main_rank_print(f"Error parsing harmony agent response: {e}")
        return ""

def extract_harmony_code_from_response(response_text: str, validate_python_code, logger) -> Tuple[str, str, str]:
    """
    Extract code from harmony response, handling both direct answers and agent definitions.
    
    Args:
        response_text: The harmony response text
        validate_python_code: Function to validate Python code
        logger: Logger instance
        
    Returns:
        Tuple of (code, name, thought)
    """    
    try:
        # First check if there's an agent definition
        agent_name = extract_xml(response_text, "agent_name")
        
        if agent_name:
            # Parse the agent response and generate code
            code = parse_harmony_agent_response(response_text)
            name = agent_name
            thought = extract_xml(response_text, "thinking")
            
            # Check if parsing was successful (code is not empty and valid)
            if code and code.strip():
                main_rank_print(f"Successfully parsed agent response for {agent_name}")
            else:
                main_rank_print(f"Agent parsing failed for {agent_name} - empty code generated. response_text: {response_text}")
            
            return code, name, thought
        else:
            # No agent found, check if we have a direct answer
            answer = extract_xml(response_text, "answer")
            if answer:
                # Direct answer found - return "direct_answer" as code to track direct answers
                main_rank_print("Found direct answer in harmony response")
                return "direct_answer", "direct_answer", answer
            else:
                # No agent and no answer - parsing failed
                main_rank_print(f"No agent_name or answer found in harmony response - parsing failed. response_text: {response_text}")
                return "direct_answer", "direct_answer", response_text
                
    except Exception as e:
        main_rank_print(f"Error parsing harmony response: {e}")
        return e, "Error parsing harmony response", "Error parsing harmony response"




