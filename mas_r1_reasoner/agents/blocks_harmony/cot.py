
import inspect
from typing import Any, Union, List, Dict, Tuple


# %%%%%%%%%%%%%%%%%%%% COT %%%%%%%%%%%%%%%%%%%%
async def CoTAgent(self, agent_input, model: str):
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"

    
    # Basic setting
    temperature = 0.5
    
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    cot_instruction = "Please think step by step and then solve the task."

    # Instantiate a new LLM specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature)

    # Get the response from the CoT tool
    thinking, answer = await cot_agent([agent_input], cot_instruction)
    final_answer = self.make_final_answer(thinking, answer)
    
    # Return only the final answer
    return final_answer   

func_string = inspect.getsource(CoTAgent)

COT = {
    "desciption": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought Agent (CoTAgent)",
    "required_arguments": {
        "agent_input": "The input for the CoTAgent. This is the task question for the CoTAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question."
    },
    "implementation": """{func_string}""".format(func_string=func_string)
}