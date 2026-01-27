import inspect
from typing import Any, Union, List, Dict, Tuple


# %%%%%%%%%%%%%%%%%%%% CoT_SC %%%%%%%%%%%%%%%%%%%%

async def SCAgent(self, agent_input, model: str):
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"

    # Basic setting
    temperature = 0.5
    num_repeated_samples = 5

    # Instruction for step-by-step reasoning
    cot_instruction = "Please think step by step and then solve the task. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"

    # Instruction for step-by-step reasoning
    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature) for _ in range(num_repeated_samples)]
    
    thinking_mapping = {}
    answer_mapping = {}
    possible_answers = []
    for i in range(num_repeated_samples):
        thinking, answer = await cot_agents[i]([agent_input], cot_instruction)
        possible_answers.append(answer.content)
        thinking_mapping[answer.content] = thinking
        answer_mapping[answer.content] = answer

    # Ensembling the answers from multiple CoT agents
    answer = self.majority_voting(possible_answers)

    thinking = thinking_mapping[answer]
    answer = answer_mapping[answer]

    final_answer = self.make_final_answer(thinking, answer)

    return final_answer  

func_string = inspect.getsource(SCAgent)

COT_SC = {"desciption": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoTAgent) agents to produce a more accurate final answer through ensembling. Best for problems where you want high confidence through consensus.",
          "name": "Self-Consistency with Chain-of-Thought (SCAgent)",
            "required_arguments": {
                "agent_input": "The input for the SCAgent. This is the task question for the SCAgent to solve. If left empty (\"\"), the parser will automatically replace it with the original question."
            },
            "implementation": """{func_string}""".format(func_string=func_string)
              }