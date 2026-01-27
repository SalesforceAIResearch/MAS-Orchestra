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
Extraction and generation utilities for Harmony MAS-R1 Trainer - Medium level.
Handles multi-agent systems with graph-based execution.
"""

from typing import Dict, Tuple, Any, List
from verl.protocol import DataProto
from mas_r1_reasoner.agents.common import main_rank_print, extract_xml
from mas_r1_reasoner.agents.shared_vars import get_global
import json
import ast
import re
from collections import defaultdict, deque


def extract_all_agents(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract all agent definitions from the response text.
    
    Args:
        response_text: The harmony response text containing multiple agent definitions
        
    Returns:
        List of agent dictionaries with id, name, description, and required_arguments
    """
    agents = []
    
    # Find all <agent>...</agent> blocks
    agent_pattern = r'<agent>(.*?)</agent>'
    agent_matches = re.findall(agent_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for agent_block in agent_matches:
        try:
            agent_id = extract_xml(agent_block, "agent_id")
            agent_name = extract_xml(agent_block, "agent_name")
            agent_description = extract_xml(agent_block, "agent_description")
            required_arguments_xml = extract_xml(agent_block, "required_arguments")
            
            if not agent_id or not agent_name:
                main_rank_print(f"Skipping agent block - missing agent_id or agent_name: {agent_block[:100]}")
                continue
            
            # Parse required_arguments
            required_arguments = {}
            
            # Extract agent_input
            agent_input = extract_xml(required_arguments_xml, "agent_input")
            # Empty string or "" means use original question
            if agent_input == '""' or not agent_input:
                required_arguments['agent_input'] = ''
            else:
                required_arguments['agent_input'] = agent_input
            
            # Extract debate_roles if present (for DebateAgent)
            debate_roles = extract_xml(required_arguments_xml, "debate_roles")
            if debate_roles:
                parsed_roles = None
                try:
                    parsed_roles = ast.literal_eval(debate_roles)
                except (ValueError, SyntaxError):
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
            
            agents.append({
                'agent_id': agent_id,
                'agent_name': agent_name,
                'agent_description': agent_description,
                'required_arguments': required_arguments
            })
            
        except Exception as e:
            main_rank_print(f"Error parsing agent block: {e}")
            continue
    
    return agents


def extract_edges(response_text: str) -> List[Tuple[str, str]]:
    """
    Extract all edges from the response text.
    
    Args:
        response_text: The harmony response text containing edge definitions
        
    Returns:
        List of (from_id, to_id) tuples
    """
    edges = []
    
    # Find the <edge>...</edge> block
    edge_block = extract_xml(response_text, "edge")
    if not edge_block:
        return edges
    
    # Find all <from> and <to> pairs
    from_pattern = r'<from>(.*?)</from>'
    to_pattern = r'<to>(.*?)</to>'
    
    from_matches = re.findall(from_pattern, edge_block, re.DOTALL | re.IGNORECASE)
    to_matches = re.findall(to_pattern, edge_block, re.DOTALL | re.IGNORECASE)
    
    # Match from and to pairs sequentially
    for from_id, to_id in zip(from_matches, to_matches):
        edges.append((from_id.strip(), to_id.strip()))
    
    return edges


def build_dependency_graph(agents: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Build a dependency graph from agents and edges.
    
    Args:
        agents: List of agent dictionaries
        edges: List of (from_id, to_id) tuples
        
    Returns:
        Dictionary mapping agent_id to list of dependent agent_ids
    """
    graph = defaultdict(list)
    
    # Initialize all agents in the graph
    for agent in agents:
        agent_id = agent['agent_id']
        if agent_id not in graph:
            graph[agent_id] = []
    
    # Add edges
    for from_id, to_id in edges:
        graph[from_id].append(to_id)
    
    return graph


def topological_sort(agents: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Perform topological sort to get execution order of agents.
    
    Args:
        agents: List of agent dictionaries
        edges: List of (from_id, to_id) tuples
        
    Returns:
        List of agent_ids in execution order
    """
    # Calculate in-degree for each agent
    in_degree = {agent['agent_id']: 0 for agent in agents}
    graph = defaultdict(list)
    
    for from_id, to_id in edges:
        graph[from_id].append(to_id)
        if to_id in in_degree:
            in_degree[to_id] += 1
    
    # Initialize queue with agents that have no incoming edges
    queue = deque([agent_id for agent_id, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        agent_id = queue.popleft()
        result.append(agent_id)
        
        # Decrease in-degree for dependent agents
        for dependent_id in graph[agent_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)
    
    # Check if all agents were processed (no cycles)
    if len(result) != len(agents):
        main_rank_print("Warning: Cycle detected in agent dependency graph")
    
    return result


def find_sink_agents(agents: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> List[str]:
    """
    Find sink agents (agents with no outgoing edges).
    
    Args:
        agents: List of agent dictionaries
        edges: List of (from_id, to_id) tuples
        
    Returns:
        List of sink agent_ids
    """
    has_outgoing = set(from_id for from_id, _ in edges)
    all_agents = set(agent['agent_id'] for agent in agents)
    sink_agents = all_agents - has_outgoing
    return list(sink_agents)


def validate_graph(agents: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> None:
    """
    Validate the agent graph structure according to harmony constraints.
    Raises ValueError if the graph is invalid.
    
    Args:
        agents: List of agent dictionaries
        edges: List of (from_id, to_id) tuples
        
    Raises:
        ValueError: If any validation constraint is violated
        
    Validation Rules:
        1. Node consistency: Every <from> and <to> must reference a valid <agent_id>
        2. Directionality: Edges are directed: data flows from <from> → <to>
        3. Connectivity: Every agent must be connected directly or indirectly to the main flow
        4. Start node(s): At least one agent must have no incoming edge
        5. Sink node: Exactly one agent with no outgoing edge
        6. No undefined edges: It is invalid to reference an agent in <from> or <to> that was not declared
        7. No cycles: Cycles are NOT allowed - graph must be a Directed Acyclic Graph (DAG)
        8. Parallelism allowed: Multiple agents may have the same <from> or <to> (fan-out/fan-in)
        9. Unambiguous sink: The parser will reject graphs with multiple sinks
        10. Order-independent: The XML order of edges does not need to follow execution order
    """
    if not agents:
        raise ValueError("Graph validation failed: No agents defined")
    
    # Get all agent IDs
    all_agent_ids = set(agent['agent_id'] for agent in agents)
    
    # RULE 1 & 6: Node consistency - every edge must reference valid agent_ids
    for from_id, to_id in edges:
        if from_id not in all_agent_ids:
            raise ValueError(
                f"Graph validation failed (Rule 1/6): Edge references undefined agent '{from_id}' in <from>. "
                f"Valid agent_ids: {sorted(all_agent_ids)}"
            )
        if to_id not in all_agent_ids:
            raise ValueError(
                f"Graph validation failed (Rule 1/6): Edge references undefined agent '{to_id}' in <to>. "
                f"Valid agent_ids: {sorted(all_agent_ids)}"
            )
    
    # RULE 4: Start nodes - at least one agent must have no incoming edge
    incoming_edges = set(to_id for _, to_id in edges)
    start_nodes = all_agent_ids - incoming_edges
    if not start_nodes:
        raise ValueError(
            f"Graph validation failed (Rule 4): No start node found. "
            f"At least one agent must have no incoming edges (entry points). "
            f"All agents have incoming edges: {sorted(all_agent_ids)}"
        )
    
    # RULE 5 & 9: Sink node - exactly one agent must have no outgoing edge
    sink_agents = find_sink_agents(agents, edges)
    if len(sink_agents) == 0:
        raise ValueError(
            f"Graph validation failed (Rule 5): No sink node found. "
            f"Exactly one agent must have no outgoing edges (final answer producer). "
            f"All agents have outgoing edges."
        )
    if len(sink_agents) > 1:
        raise ValueError(
            f"Graph validation failed (Rule 9): Multiple sink nodes found: {sorted(sink_agents)}. "
            f"Must have exactly one sink node (final answer producer). "
            f"Add a final 'collector' agent if you need to merge multiple endpoints."
        )
    
    # RULE 3: Connectivity - every agent must be reachable from start nodes
    # Build adjacency graph (both forward and backward for full connectivity check)
    graph = defaultdict(set)
    reverse_graph = defaultdict(set)
    for from_id, to_id in edges:
        graph[from_id].add(to_id)
        reverse_graph[to_id].add(from_id)
    
    # BFS from all start nodes to find reachable agents (forward)
    visited_forward = set()
    queue = deque(start_nodes)
    while queue:
        node = queue.popleft()
        if node in visited_forward:
            continue
        visited_forward.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited_forward:
                queue.append(neighbor)
    
    # BFS from sink to find agents that can reach sink (backward)
    visited_backward = set()
    queue = deque(sink_agents)
    while queue:
        node = queue.popleft()
        if node in visited_backward:
            continue
        visited_backward.add(node)
        for neighbor in reverse_graph.get(node, []):
            if neighbor not in visited_backward:
                queue.append(neighbor)
    
    # Agents must be reachable from start AND must reach the sink
    connected_agents = visited_forward & visited_backward
    isolated_agents = all_agent_ids - connected_agents
    
    if isolated_agents:
        raise ValueError(
            f"Graph validation failed (Rule 3): Isolated agents detected: {sorted(isolated_agents)}. "
            f"Every agent must be connected directly or indirectly to the main flow. "
            f"Connected agents: {sorted(connected_agents)}"
        )
    
    # RULE 7: Cycle detection - cycles are NOT allowed
    # Use DFS to detect cycles and find the cycle path
    def find_cycle_dfs(node, visited, rec_stack, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                cycle_path = find_cycle_dfs(neighbor, visited, rec_stack, path[:])
                if cycle_path:
                    return cycle_path
            elif neighbor in rec_stack:
                # Found a cycle - return the cycle path
                cycle_start_idx = path.index(neighbor)
                return path[cycle_start_idx:] + [neighbor]
        
        rec_stack.discard(node)
        return None
    
    visited = set()
    for node in all_agent_ids:
        if node not in visited:
            cycle_path = find_cycle_dfs(node, set(), set(), [])
            if cycle_path:
                cycle_str = " -> ".join(cycle_path)
                raise ValueError(
                    f"Graph validation failed (Rule 7): Cycle detected in agent graph. "
                    f"Cycles are NOT allowed. Found cycle: {cycle_str}. "
                    f"Please restructure your graph to be a Directed Acyclic Graph (DAG)."
                )
    
    # All validations passed
    main_rank_print(f"✅ Graph validation passed: {len(agents)} agents, {len(edges)} edges, "
                   f"{len(start_nodes)} start node(s), 1 sink node")


def parse_harmony_agent_response(response_text: str) -> str:
    """
    Parse harmony agent response and convert it to executable code for medium level.
    Handles both single agent and multi-agent graphs.
    
    Args:
        response_text: The harmony response text containing agent definitions
        
    Returns:
        str: Generated code as a forward() function, or empty string if no agent found
    """
    try:
        # Extract all agents
        agents = extract_all_agents(response_text)
        
        if not agents:
            main_rank_print("No agents found in harmony response")
            return ""
        
        main_rank_print(f"Found {len(agents)} agent(s)")
        
        # Extract edges
        edges = extract_edges(response_text)
        main_rank_print(f"Found {len(edges)} edge(s)")
        
        # Check if this is a single agent case (no edges)
        if len(agents) == 1 and len(edges) == 0:
            main_rank_print("Single agent case detected (similar to minimal)")
            return generate_single_agent_code(agents[0])
        
        # Check for invalid case: multiple agents but no edges
        if len(agents) > 1 and len(edges) == 0:
            raise ValueError(
                f"Graph validation failed: Found {len(agents)} agents but no edges. "
                f"Multiple agents require edges to define execution flow. "
                f"Agent IDs: {[a['agent_id'] for a in agents]}"
            )
        
        # Multi-agent case with graph execution
        main_rank_print("Multi-agent graph case detected")
        
        # Validate graph structure before code generation
        main_rank_print("Validating agent graph structure...")
        validate_graph(agents, edges)
        
        return generate_multi_agent_code(agents, edges)
        
    except Exception as e:
        main_rank_print(f"Error parsing harmony agent response: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_single_agent_code(agent: Dict[str, Any]) -> str:
    """
    Generate code for a single agent execution (similar to minimal).
    
    Args:
        agent: Agent dictionary
        
    Returns:
        Generated code string
    """
    agent_name = agent['agent_name']
    agent_description = agent['agent_description']
    agent_id = agent['agent_id']
    required_arguments = agent['required_arguments']
    
    global_node_model = get_global('global_node_model')
    
    agent_calls = [{
        'name': agent_name,
        'required_arguments': required_arguments
    }]
    
    code_lines = [
        "async def forward(self, original_task_info):",
        "    \"\"\"Generated harmony agent forward function - single agent (async)\"\"\"",
        f"    agent_name = \"\"\"{agent_name}\"\"\"",
        f"    agent_description = \"\"\"{agent_description}\"\"\"",
        f"    agent_id = \"\"\"{agent_id}\"\"\"",
        "",
        f"    agent_calls = {json.dumps(agent_calls, indent=4)}",
        "",
        "    # Execute single agent",
        "    agent_name_lower = agent_name.lower()",
        "    args = agent_calls[0]['required_arguments']",
        "    agent_input = args.get('agent_input', '')",
        "",
        "    # Prepare task_info",
        "    if not agent_input or not agent_input.strip():",
        "        task_info = original_task_info",
        "    else:",
        "        combined_content = f'{agent_input}'",
        "        task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
        "",
        "    # Execute based on agent type",
        "    if agent_name_lower == 'cotagent':",
        f"        result = await self.CoTAgent(agent_input=task_info, model='{global_node_model}')",
        "        return result",
        "    elif agent_name_lower == 'scagent':",
        f"        result = await self.SCAgent(agent_input=task_info, model='{global_node_model}')",
        "        return result",
        "    elif agent_name_lower == 'reflexionagent':",
        f"        result = await self.ReflexionAgent(agent_input=task_info, model='{global_node_model}')",
        "        return result",
        "    elif agent_name_lower == 'debateagent':",
        "        debate_roles = args.get('debate_roles')",
        f"        result = await self.DebateAgent(agent_input=task_info, model='{global_node_model}', debate_roles=debate_roles)",
        "        return result",
        "    elif agent_name_lower == 'websearchagent':",
        f"        result = await self.WebSearchAgent(agent_input=task_info, model='{global_node_model}')",
        "        return result",
        "        return result",
        "    else:",
        "        raise ValueError(f\"Unknown agent name: {agent_name}\")",
    ]
    
    generated_code = "\n".join(code_lines)
    
    main_rank_print(f"\n{'='*80}")
    main_rank_print(f"GENERATED CODE FOR SINGLE AGENT: {agent_name}")
    main_rank_print(f"{'='*80}")
    main_rank_print(generated_code)
    main_rank_print(f"{'='*80}\n")
    
    return generated_code


def generate_multi_agent_code(agents: List[Dict[str, Any]], edges: List[Tuple[str, str]]) -> str:
    """
    Generate code for multi-agent graph execution.
    
    Args:
        agents: List of agent dictionaries
        edges: List of (from_id, to_id) tuples
        
    Returns:
        Generated code string
    """
    global_node_model = get_global('global_node_model')
    
    # Get execution order
    execution_order = topological_sort(agents, edges)
    
    # Find sink agents (final outputs)
    sink_agents = find_sink_agents(agents, edges)
    
    # Create agent lookup
    agent_lookup = {agent['agent_id']: agent for agent in agents}
    
    main_rank_print(f"Execution order: {execution_order}")
    main_rank_print(f"Sink agents: {sink_agents}")
    
    code_lines = [
        "async def forward(self, original_task_info):",
        "    \"\"\"Generated harmony agent forward function - multi-agent graph (async)\"\"\"",
        "    import re",
        "    ",
        "    # Agent configurations",
        f"    agents_config = {json.dumps(agents, indent=4)}",
        f"    edges = {json.dumps(edges, indent=4)}",
        f"    execution_order = {json.dumps(execution_order, indent=4)}",
        f"    sink_agents = {json.dumps(sink_agents, indent=4)}",
        "",
        "    # Store agent results",
        "    agent_results = {}",
        "",
        "    # Execute agents in topological order",
        "    for agent_id in execution_order:",
        "        # Get agent config",
        "        agent_config = next(a for a in agents_config if a['agent_id'] == agent_id)",
        "        agent_name = agent_config['agent_name']",
        "        agent_description = agent_config['agent_description']",
        "        args = agent_config['required_arguments']",
        "        agent_input = args.get('agent_input', '')",
        "",
        "        # Substitute ${agent_id} references with actual results",
        "        if agent_input:",
        "            for prev_agent_id, prev_result in agent_results.items():",
        "                # Replace ${agent_id} with the actual result content (Info.content)",
        "                pattern = r'\\$\\{' + re.escape(prev_agent_id) + r'\\}'",
        "                # Use lambda to avoid re.sub() interpreting backslashes in replacement",
        "                agent_input = re.sub(pattern, lambda m: str(prev_result.content), agent_input)",
        "",
        "        # Prepare task_info",
        "        if not agent_input or not agent_input.strip():",
        "            task_info = original_task_info",
        "        else:",
        "            # For multi-agent, prepend original question as context",
        "            combined_content = f'{agent_input}.'", #TODO: slightly different from minimal
        "            task_info = Info('task', 'User', combined_content, None, None, None, -1, None)",
        "",
        "        # Execute agent based on type",
        "        agent_name_lower = agent_name.lower()",
        "        result = None",
        "",
        "        if agent_name_lower == 'cotagent':",
        f"            result = await self.CoTAgent(agent_input=task_info, model='{global_node_model}')",
        "        elif agent_name_lower == 'scagent':",
        f"            result = await self.SCAgent(agent_input=task_info, model='{global_node_model}')",
        "        elif agent_name_lower == 'reflexionagent':",
        f"            result = await self.ReflexionAgent(agent_input=task_info, model='{global_node_model}')",
        "        elif agent_name_lower == 'debateagent':",
        "            debate_roles = args.get('debate_roles')",
        f"            result = await self.DebateAgent(agent_input=task_info, model='{global_node_model}', debate_roles=debate_roles)",
        "        elif agent_name_lower == 'websearchagent':",
        f"            result = await self.WebSearchAgent(agent_input=task_info, model='{global_node_model}')",
        "        else:",
        "            raise ValueError(f\"Unknown agent name: {agent_name}\")",
        "",
        "        # Store result",
        "        agent_results[agent_id] = result",
        "",
        "    # Return result from sink agent(s)",
        "    if len(sink_agents) == 1:",
        "        return agent_results[sink_agents[0]]",
        "    else:",
        "        # Multiple sink agents - combine results",
        "        combined_result = '\\n\\n'.join([f'{agent_id}: {agent_results[agent_id]}' for agent_id in sink_agents])",
        "        return combined_result",
    ]
    
    generated_code = "\n".join(code_lines)
    
    main_rank_print(f"\n{'='*80}")
    main_rank_print(f"GENERATED CODE FOR MULTI-AGENT GRAPH")
    main_rank_print(f"{'='*80}")
    main_rank_print(generated_code)
    main_rank_print(f"{'='*80}\n")
    
    return generated_code


def extract_harmony_code_from_response(response_text: str, validate_python_code, logger) -> Tuple[str, str, str]:
    """
    Extract code from harmony response, handling both single agent and multi-agent graphs.
    
    Args:
        response_text: The harmony response text
        validate_python_code: Function to validate Python code
        logger: Logger instance
        
    Returns:
        Tuple of (code, name, thought)
    """    
    try:
        # Try to extract agents
        agents = extract_all_agents(response_text)
        
        if agents:
            # Parse the agent response and generate code
            code = parse_harmony_agent_response(response_text)
            
            # Use first agent name or "multi_agent" if multiple
            if len(agents) == 1:
                name = agents[0]['agent_name']
            else:
                name = "multi_agent_graph"
            
            thought = extract_xml(response_text, "thinking")
            
            # Check if parsing was successful
            if code and code.strip():
                main_rank_print(f"Successfully parsed agent response for {name}")
            else:
                main_rank_print(f"Agent parsing failed for {name} - empty code generated. response_text: {response_text}")
            
            return code, name, thought
        else:
            # No agents found, check if we have a direct answer
            answer = extract_xml(response_text, "answer")
            if answer:
                main_rank_print("Found direct answer in harmony response")
                return "direct_answer", "direct_answer", answer
            else:
                # No agents and no answer - parsing failed
                main_rank_print(f"No agents or answer found in harmony response - parsing failed. response_text: {response_text}")
                return "direct_answer", "direct_answer", response_text
                
    except Exception as e:
        main_rank_print(f"Error parsing harmony response: {e}")
        import traceback
        traceback.print_exc()
        return str(e), "Error parsing harmony response", "Error parsing harmony response"
