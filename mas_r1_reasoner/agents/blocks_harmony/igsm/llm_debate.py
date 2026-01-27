import inspect
from typing import Any, Union, List, Dict, Tuple


# %%%%%%%%%%%%%%%%%%%% LLM-Debate (collabrative) %%%%%%%%%%%%%%%%%%%%


async def DebateAgent(self, agent_input, model: str, debate_roles: List[str]):
    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info
    
    # Validate that agent_input is an Info object
    assert isinstance(agent_input, Info), f"agent_input must be an Info object, got {agent_input}"

    # Basic setting
    temperature = 0.5
    max_debate_round = 5

    # Instruction for initial reasoning
    debate_initial_instruction =  "Please think step by step and then solve the task. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"

    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Put your thinking process in the 'thinking' field and the updated answer in the 'answer' field. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"

    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate LLM', model=model, role=role, temperature=temperature) for role in debate_roles]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer. Put your thinking process in the 'thinking' field and the final answer in the 'answer' field. All caculation are done mod 23. Any parameter that wasn't mentioned in the problem statement is by default zero"


    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision LLM', model=model, temperature=temperature)

    all_thinking = [[] for _ in range(max_debate_round)]
    all_answer = [[] for _ in range(max_debate_round)]

    # Perform debate rounds
    for r in range(max_debate_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = await debate_agents[i]([agent_input], debate_initial_instruction)
            else:
                input_infos = [agent_input] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = await debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # Make the final decision based on all debate results and solutions
    thinking, answer = await final_decision_agent([agent_input] + all_thinking[max_debate_round-1] + all_answer[max_debate_round-1], final_decision_instruction)
    final_answer = self.make_final_answer(thinking, answer)

    return final_answer

func_string = inspect.getsource(DebateAgent)

LLM_debate = {
    "desciption": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks. Best for problems that benefit from multiple perspectives.",
    "name": "LLM Debate (DebateAgent)",
    "required_arguments": {
        "agent_input": "The input for the DebateAgent. This is the task question or context for the DebateAgent to solve. If left empty (\"\") the parser will automatically replace it with the original question.",
        "debate_roles": "A list of roles (musst be more than one) for the DebateAgent (e.g., 'Mathematics Professor', 'Statistician'). Each role represents a distinct perspective and viewpoint."
    },
    "implementation": """{func_string}""".format(func_string=func_string)

}