
import json
from mas_r1_reasoner.agents.prompts.code_archive import util_code, wrong_implementation

EXAMPLE = {
    "thought": """
    **Insights:**\nProvide your reasoning for the next effective block or architecture (an architecture may contain multiple blocs), along with an explanation of the overall concept behind the design. 

    **Task and Sub-tasksin the given code**:\n This corresponding to the step 1. You need to extract its sub-tasks. The task and sub-tasks should be the same as the given code and should not be changed during the whole block or architecture design.

    **Updated Block/Architeure:**\nThis corresponding to the step 2. You updated block or architcutre given the task or sub-tasks in the given code (no changes should be made to the task or sub-tasks in the given code). Explain how this architecture can solve each of the resulting sub-task
    "**Implementation:**describe the implementation step by step."
    """,
    "name": "Name of your designed block or architecture",

    "code": """def forward(self, taskInfo):
    from collections import Counter # must have this and always make sure you import everything needed
    # Your code here. IMPORTANT  
    # (1) If the given code uses single block, you need to determine whether to use the same block or other single block. If the given code uses decomposition, follow the instructions below.
    # (2) You cannot call the existing architecture from the archive but you have to implment it from the code in the available blocks archive
    # for example:
    # you cannot call 'COT' but you have to implement it eactly the same as the code in the available blocks archive. 
    # name an agent 'reflexion' or 'debate' or 'cot_sc' also do not implement the blocks. Make sure you ACTUALLY IMPLEMENET them
    # for example, You need to impement the for-loop in COT-SC, LLM Debate and Reflextion
    # You should only change how they connect but not the function inside (setting and instrction can be different)

    # (3) To creat an agent, call the LLMAgentBase, must have 'model=self.node_model' in the parameters
    # the return of the call is always the same as the output fields,
    # for example: 
    # reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent', model=self.node_model)

    # (4) Since the agent is handling with a sub_question made by question decomposition, use 'is_sub_task=True' when calling the agent.

    # (5) You need to specify the sub-task dependencies by specify the dependent sub-tasks in the instruction. You also need to give each sub-task an ID. This means you need to state clearly what the current sub-task is based on and also give all required information (thinking and answer) in the input task info
    # for example:
    # cot_instruction = ("Sub-task 3: Based on the output of sub-task 1 and sub-task 2.....")
    # thinking, answer = reasoning_agent([taskInfo] + [thinking1, answer1, thiking2, answer2, ..., ], cot_instruction, is_sub_task=True)


    # (6) You need to keep tack of sub-task output, for each sub-task output, please append it to a list named `sub_tasks` (please initialize it as `sub_tasks = []` at the beining of your function) so that we can keep track of the performance of each sub-task. When you do self.make_final_answer, please also add `sub_tasks` as the second last parameterss 
    # for example: 
    # sub_tasks = []
    # ...
    # thinking, answer = reasoning_agent([taskInfo] + [sub-task thinking, sub-task answer], reasoning_instruction, is_sub_task=True)
    # sub_tasks.append(f'the output of sub task n: thinking {thinking.content}; final answer: {answer.content}')

    # (7) You need to keep track of agent, for each agent inside the sub-architecture (or block or node) output, please append it to a list named `agents` (please initialize it as `agents = []` at the beining of your function) so that we can keep track of the performance of each agents. It should contain the agent name (if setting is important to identify the agent, please include as well, e.g., round, ID), agent purpose, and the thinking and output of the agent. When you do self.make_final_answer, please also add `agents` as the last parameterss 
    # Takes debate as an example: 
    # debate_instruction = ("Sub-task i: Based on the output of...")
    # max_round = ...(the max round you determine)
    # debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in ...(the role you design)]
    # all_thinking = []
    # all_answer = []
    # for r in range(max_round):
    #     round_thinking = []
    #     round_answer = []
    #     for i, agent in enumerate(debate_agents):
    #         if r == 0:
    #             t, a = agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    #             agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of..., thinking: {t.content}; answer: {a.content}')
    #         else:
    #             t, a = agent([taskInfo, thinking1, answer1] + all_thinking[r-1], debate_instruction, is_sub_task=True)
    #             agents.append(f'Debate agent {agent.id}, round {r}, on the purpose of..., thinking: {t.content}; answer: {a.content}')
    #         round_thinking.append(t)
    #         round_answer.append(a)
    #     all_thinking.append(round_thinking)
    #     all_answer.append(round_answer)
    # final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    # thinking2, answer2 = final_decision_agent(
    #     [taskInfo] + all_thinking[-1] + all_answer[-1],
    #     "Sub-task i: Based on the output of...",
    #     is_sub_task=True
    # )
    # agents.append(f'Final Decision agent, on the purpose of..., thinking: {thinking2.content}; answer: {answer2.content}')
    # sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")

    # Take reflexion as another example:
    # cot_reflect_instruction = "Sub-task i: Based on the output of..."
    # cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)

    # # Instruction for providing feedback and correcting the answer
    # critic_instruction = "Sub-task i: Based on the output of...,"
    # critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    # N_max = ...(the max round you determine) # Maximum number of attempts
    # # Initial attempt
    # cot_inputs = [taskInfo]
    # thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0, is_sub_task=True)
    # agents.append(f'CoT agent {cot_agent.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')

    # for i in range(N_max):
    #     # Get feedback and correct status from the critic
    #     feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i, is_sub_task=True)
    #     agents.append(f'Critic agent {critic_agent.id}, on the purpose of..., thinking: {feedback.content}; answer: {correct.content}')
    #     if correct.content == 'True':
    #         break
            
    #     # Add feedback to the inputs for the next iteration
    #     cot_inputs.extend([thinking, answer, feedback])

    #     # Reflect on previous attempts and refine the answer
    #     thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
    #     agents.append(f'CoT agent {cot_agent.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')
    # sub_tasks.append(f"Sub-task i output: thinking - {thinking.content}; answer - {answer.content}")

    # Take self-consistency as another example:
    # cot_instruction = 'Sub-task i: Based on the output of...'
    # # Initialize multiple CoT agents with a higher temperature for varied reasoning
    # cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent',  model=self.node_model, temperature=0.5) for _ in range(N)]
    
    # thinking_mapping = {}
    # answer_mapping = {}
    # possible_answers = []
    # for i in range(N):
    #     thinking, answer = cot_agents[i]([taskInfo], cot_instruction, is_sub_task=True)
    #     agents.append(f'CoT agent {cot_agents.id}, on the purpose of..., thinking: {thinking.content}; answer: {answer.content}')
    #     possible_answers.append(answer.content)
    #     thinking_mapping[answer.content] = thinking
    #     answer_mapping[answer.content] = answer
    # answer = majority_voting(possible_answers)

    # thinking = thinking_mapping[answer]
    # answer = answer_mapping[answer]
    # sub_tasks.append(f"Sub-task i output: thinking - {thinking.content}; answer - {answer.content}")

    # (8) Put the saved sub_tasks and agents to the final elf.make_final_answer. Make sure you have `is_sub_task=True` when calling an agent, and keep track of `sub_task`, `agents`, include `sub_task` dependency and detailed steps for the sub-task ('Sub-task i: based on sub-task,') in the sub-task instruction, and actually implmenet the blocks by yourself (for-loop if COT-SC, Debate and Reflextion).

    final_answer = self.make_final_answer(thinking, answer, sub_tasks, agents)
    # Return only the final answer
    return final_answer
"""
}


base = f"""# Overview
You are an expert machine learning researcher testing various agentic systems. Given a set of blocks in the archive and the question, The main goal here is to update the selected block in the giveb code to BETTER solve the question, WITHOUT changing the task or sub-tasks in the given code, but only changing the architecture or block. Note that architecture can contain multiple blocks, and block mean a LLM that use for specifical objectives by specifclaied setting (instruction, tempreture...)


Your objective is to 

Step 1: Determine the sub-task in the given code. Read the given code carefully, you need to identify the sub-tasks and task in the given code. 
Step 2: Given the task and sub-tasks, design connections between existing blocks to adress the task and sub-tasks. During the deisgn, you should NEVER change the tasks or sub-tasks in the given code.

You need to design a good structure to solve each of the sub-tasks. More specifically, you should structure the architecture as a multi-layered network. Each available block in the archive serves as a node, while connections between them act as edges, forming a structured hierarchy of interactions. Additionally, you must determine the number of layers in the network.

For example, if the available blocks are 'COT, COT_SC, Reflexion, LLM_debate' and you determine that there can be 3 layers. There are 3 resulting sub-task from (1) sub-task 1, sub-task 2, sub-task 1, sub-task 3:

Example Setup

Resulting sub-tasks:
sub-task 1, sub-task 2, sub-task 3, sub-task 4

Available architectures:
COT, COT_SC, Reflexion, LLM_debate

Network with 3 Layers:

Layer 1: COT  COT_SC  Reflexion  LLM_debate  
Layer 2: COT  COT_SC  Reflexion  LLM_debate   
Layer 3: COT  COT_SC  Reflexion  LLM_debate  

Connection Strategies:

1. Linear Connection: Directly link two block to pass information forward.
Example: [COT] (address sub-task 1) -> [LLM_debate] (address sub-task 2) (Single connection and exit)

2. Multi-Layer Connection: An block can appear in multiple layers, forming deeper reasoning structures.
Example: [COT] (address sub-task 1) -> [LLM_debate] (address sub-task 2) -> [COT -> Reflexion] (address sub-task 3) (COT appears in both Layer 1 and Layer 3) (the whole [COT -> Reflexion] is a sub-task architecture that aims to address sub-task 3)

Your aim is to design an optimal block connection that can performe well on each of the sub-task. Your code should implment the available blocks given in the archive (the 'code' field of blocks) as it-is without modication: Do not propose new blocks or modify existing ones and only change the connections between the given blocks, but block setting like instruction, tempreture are allowed to modify


{util_code}

# Output Instruction and Example:
The first field should be ("thought"), and it should capture your thought process for and it should capture your thought process for reconnecting the available blocks in achived. 

In the "thought" field, include the following:

(1) **Sub-task and Task in the given code**: This corresponding to the step 1. You need to determine the sub-tasks in the given code.

(2) **Updated Single Block/Architecture**:  This corresponding to the step 2. Sepcfically, you need to design a good structure to solve each of the sub-tasks. You must not change the tasks or sub-tasks in the given code. You should ONLY update the architecture or block.

Given the sub-task 1, sub-task 2, ... sub-task n extracted from given code in stepo 1, design connections between existing blocks to adress each of them. describe your reasoning and the overall concept behind the connection design and finally detail the implementation steps. All connection must betweene available blocks in the archive and no new blocks can be made. The format must strickly follow: 

Use '->' for connection. for example, 'CoT (address sub-task 1) (available block name) -> LLM debate (address sub-task 2) (another available block name)' means connect the CoT block and the LLM debate block to address sub-task 1 and sub-task 2 correspondingly.

The second field ("name") corresponds to the name of your design. 

Finally, the last field ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project (so do not implement any other part), so please implement complete, reliable, reusable code snippets. You cannot call the available blocks (e.g., COT, COT-SC) by its name, but must implement them as it is in the achive. If the block is handling a sub-task, add 'sub_task=True' when calling the block.

Here is an example of the output format for the new slection or new connected block architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. DON'T try to use some function that doesn't exisit.
Also, it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of “Infos”. You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

{wrong_implementation}


# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to better solve the question by re-connecting or re-selecting the available block in archived. Do not try to propose new block or modify the available block, and only change the connection or selection. Block setting like instruction, tempreture are allowed to modify
Observe the discovered blocka carefully and think about what insights, lessons, or stepping stones can be learned from them.
You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the new connection or selection.

Below is the question to solve:\n\n[QUESTION]\n\n
Below is the given code:\n\n[CODE]
"""