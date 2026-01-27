"""
CoTAgent with think_tool
"""

import inspect


async def CoTAgent(self, agent_input, model: str) -> str:
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    from pydantic import BaseModel, Field
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool
    from langchain_together import ChatTogether
    from datetime import datetime
    from typing import Any, List, Dict, Optional
    import os
    
    max_iterations = 20
    max_tool_calls_per_iter = None
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"
    
    # Extract the query from agent_input
    query = agent_input.content

    # Store configuration in a container
    context = {
        "chat_model": None,
    }


    # Research system prompt (adapted from open_deep_research)
    research_system_prompt = """You are a research assistant conducting research on the user's input topic.

        <Task>
        Your job is to use tools to gather information about the user's input topic.
        You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
        </Task>

        <Available Tools>
        You have access to two main tools:
        1. **think_tool**: For reflection and strategic planning during research

        **CRITICAL: Use think_tool to reflect on reasoning results and plan next steps. Do not call think_tool with any other tools. It should be to reflect on the results of the previous reasoning results.**
        </Available Tools>

        <Instructions>
        Think like a human researcher. Follow these steps:

        1. **Read the question carefully** - What problem is the user trying to solve?
        2. **Solve each problem separately* - Call tools to solve each problem. Do not solve all at once
        4. **Narrower the thinking as you gather information** - Fill in the gaps
        5. **Stop when you can answer confidently** - Don't keep solving for perfection
        </Instructions>

        <Hard Limits>
        **Must use think tool**:
        - You must use think_tool for each sub-problem. If there are 10 problems, for example, you must use think_tool at least 10 times.
        - In the 10 problems example, likely you will call more than 10 think_tool as you may have more detailed reflection or verification with the think_tool
        - **Always stop**: After 5 search tool calls if you cannot find the right sources

        **Stop Criteria**:
        - Do not stop until you have a complete answer to each problem and with the corresponing tool calls.
        - No need to worry about the resources limit.
        </Hard Limits>


        <Show Your Thinking>
        After each reasoning step, use think_tool to analyze the results:
        - What key information did I find?
        - What's missing?
        - Do I have enough to answer the question comprehensively?
        - Should I think more or provide my answer?
        </Show Your Thinking>
    """

    # Initialize think_tool for reflection (from open_deep_research)
    @tool(description="Strategic reflection tool for research planning")
    async def think_tool(reflection: str) -> str:
        """Tool for strategic reflection on research progress and decision-making.
        
        Use this tool after each reasoning step to analyze results and plan next steps systematically.
        This creates a deliberate pause in the research workflow for quality decision-making.
        
        When to use:
        - After receiving reasoning step results: What key information did I find?
        - Before deciding next steps: Do I have enough to answer comprehensively?
        - When assessing research gaps: What specific information am I still missing?
        - Before concluding research: Can I provide a complete answer now?
        
        Reflection should address:
        1. Analysis of current findings - What concrete information have I gathered?
        2. Gap assessment - What crucial information is still missing?
        3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
        4. Strategic decision - Should I continue searching or provide my answer?
        
        Args:
            reflection: Your detailed reflection on research progress, findings, gaps, and next steps
            
        Returns:
            Confirmation that reflection was recorded for decision-making
        """
        print(f"\n💭 THINKING: {reflection[:200]}{'...' if len(reflection) > 200 else ''}")
        return f"Reflection recorded: {reflection}"
    
    # Use pre-initialized chat model from AgentSystem (initialized once per worker)
    # If not available (e.g., sync mode), initialize it here
    if hasattr(self, 'offline_chat_model') and self.offline_chat_model is not None:
        chat_model = self.offline_chat_model
        print(f"🧠 Using pre-initialized chat model from AgentSystem (OFFLINE MODE)")
    else:
       raise RuntimeError("Chat model not initialized. Please initialize the chat model first.")
    
    # Store the model in context so web_search tool can access it for summarization
    context["chat_model"] = chat_model
    
    # Bind tools to model (LangChain handles format automatically - Harmony for GPT-OSS, native for others!)
    # Include both search and think_tool (following open_deep_research pattern)
    model_with_tools = chat_model.bind_tools([think_tool])
    
    # Prepare system prompt
    system_prompt = research_system_prompt
    
    # Initialize conversation
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research Question: {query}\n\nPlease conduct searches and provide a comprehensive answer with sources. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero")
    ]
    
    # Tool calling loop
    import asyncio
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        # Call model (LangChain handles everything - including Harmony format for GPT-OSS!)
        response = await model_with_tools.ainvoke(messages)
        messages.append(response)

        print(f"  - Msg: {messages}")
       
        # Check if model wants to use tools
        if not response.tool_calls:
            # No tool calls - this is the final answer
            print("No tool calls - final answer received")
            return response.content
        
        # Limit tool calls to prevent explosion (only if max_tool_calls_per_iter is set)
        if max_tool_calls_per_iter is not None and len(response.tool_calls) > max_tool_calls_per_iter:
            print(f"   ⚠️  Limiting tool calls: {len(response.tool_calls)} → {max_tool_calls_per_iter}")
            response.tool_calls = response.tool_calls[:max_tool_calls_per_iter]
        
        # Execute tool calls
        print(f"Executing {len(response.tool_calls)} tool call(s)...")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  - {tool_name}({tool_args})")
            
            # Execute the tool

            if tool_name == "think_tool":
                reflection = tool_args.get("reflection", "")
                tool_result = await think_tool.ainvoke(reflection)
            else:
                tool_result = f"Unknown tool: {tool_name}"
            
            # Add tool result to messages (LangChain handles the format)
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))
    
    # Max iterations reached - return last response
    print(f"Max iterations ({max_iterations}) reached")
    final_response = await model_with_tools.ainvoke(messages)
    return final_response.content




# Export for mas_r1 framework compatibility
func_string = inspect.getsource(CoTAgent)

COT = {
    "desciption": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought Agent (CoTAgent)",
    "required_arguments": {
        "agent_input": "The input for the CoTAgent. This is the task question for the CoTAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question."
    },
    "implementation": """{func_string}""".format(func_string=func_string)
}