import inspect
from typing import Any, Union, List, Dict, Tuple

# %%%%%%%%%%%%%%%%%%%% relexion (generator-evaluator) %%%%%%%%%%%%%%%%%%%%
async def ReflexionAgent(self, agent_input, model: str):
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"

    # Basic setting
    temperature = 0.5
    max_reflection_round = 5

    # Instruction for initial reasoning
    initial_instruction = "Please think step by step and then solve the task. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"

    # Instruction for reflecting on previous attempts and feedback to improve
    reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature)

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Given all caculation are done mod 23 and any parameter that wasn't mentioned in the problem statement is by default zero. Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."

    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic LLM', model=model, temperature=temperature)
        
    # Initial attempt
    cot_inputs = [agent_input]
    thinking, answer = await cot_agent(cot_inputs, initial_instruction, 0)

    for i in range(max_reflection_round):
        # Get feedback and correct status from the critic
        feedback, correct = await critic_agent([agent_input, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = await cot_agent(cot_inputs, reflect_instruction, i + 1)

    final_answer = self.make_final_answer(thinking, answer)

    return final_answer


func_string = inspect.getsource(ReflexionAgent)


Reflexion = {
    "desciption": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution. Best for complex problems that benefit from self-correction.",
    "name": "Self-Refine (Reflexion)",
    "required_arguments": {
        "agent_input": "The input for the ReflexionAgent. This is the task question for the ReflexionAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question."
    },
    "implementation": """{func_string}""".format(func_string=func_string)
}



