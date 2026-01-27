
import inspect
from typing import Any, Union, List, Dict, Tuple

#TODO: not done, consider langchain

# %%%%%%%%%%%%%%%%%%%% COT %%%%%%%%%%%%%%%%%%%%
async def PythonAgent(self, agent_input, model: str):
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"

    
    # Basic setting
    temperature = 0.5
    
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    instruction = "Please solve the task via writing and executing python code."

    # Instantiate a new LLM specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    search_agent = LLMAgentBase(['thinking', 'answer'], 'Python LLM', model=model, temperature=temperature)

    # Get the response from the CoT tool
    thinking, answer = await search_agent([agent_input], instruction)
    final_answer = self.make_final_answer(thinking, answer)
    
    # Return only the final answer
    return final_answer   

func_string = inspect.getsource(PythonAgent)

Python = {
    "desciption": "The PythonAgent allows models to write and run Python code in a sandboxed environment to solve complex problems in domains like data analysis, coding, and math. Use it for: (1) Processing files with diverse data and formatting; (2)Generating files with data and images of graphs; (3) Writing and running code iteratively to solve problems—for example, a model that writes code that fails to run can keep rewriting and running that code until it succeeds",
    "name": "Python Code Interpreter Agent (PythonAgent)",
    "required_arguments": {
        "agent_input": "The input for the PythonAgent. This is the task question for the PythonAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question."
    },
    "implementation": """{func_string}""".format(func_string=func_string)
}