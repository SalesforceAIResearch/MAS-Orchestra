from mas_r1_reasoner.agents.prompts.code_archive import util_code, wrong_implementation

EXAMPLE = {
    "thought": """
    **Insights:**\nProvide your reasoning for each step, along with an explanation of the overall concept behind the design. 

    **No decomposition or decomposition**:\n This corresponding to the step 1. Sepcfically, you need to first determine whether you need to propose decomposition or you can use directly solve the task. Carefully consider the features of each block, the question, and the answer before making your decision. Output "No decomposition" or "decomposition".

    **Decomposion:**\n This corresponding to the step 1.2.1, so you only need to do this if you select decomposition. Sepcfically, you decompose the given task into multiple manageable sub-tasks. Please explain 
    (1) How do you make sure the sub-task is easier enough and thus solavable by the given or proposed blocks and agents; 
    (2) Justify how these sub-tasks can achieve the final answer to the orginal questions.

    **CoT to solve each sub-task:**\n This corresponding to the step 1.2.2, so you only need to do this if you select decomposition and after finishing step 1.2.1. Sepcfically given the resulting sub-task 1, sub-task 2, ... sub-task n from step 1.2.1, always use CoT to address each of them.

    "**Implementation:**\n describe the implementation step by step."
    """,
    "name": "Name of your No decomposition or decomposition design",

    "code": """def forward(self, taskInfo):
    from collections import Counter # must have this and always make sure you import everything needed
    # Your code here. IMPORTANT  
    # (1) If you select No decomposition, you need to implement the CoT from the code in the available blocks archive. If you select decomposition, follow the instructions below.
    # (2) You cannot call the existing architecture from the archive but you have to implment it from the code in the available blocks archive
    # for example:
    # you cannot call 'COT' but you have to implement it eactly the same as the code in the available blocks archive. Make sure you ACTUALLY IMPLEMENET them
    # You should always use CoT to address each of the sub-task.

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

    # Take CoT as an example:
    # cot_instruction = "Sub-task i: Based on the output of..."
    # cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5)

    # (8) Put the saved sub_tasks and agents to the final elf.make_final_answer. Make sure you have `is_sub_task=True` when calling an agent, and keep track of `sub_task`, `agents`, include `sub_task` dependency and detailed steps for the sub-task ('Sub-task i: based on sub-task,') in the sub-task instruction, and actually implmenet the blocks by yourself (for-loop if CoT_SC, Debate and Reflextion).

    final_answer = self.make_final_answer(thinking, answer, sub_tasks, agents)
    # Return only the final answer
    return final_answer
"""
}

base = f"""# Overview
You are an expert machine learning researcher testing various agentic systems. Given a set of blocks in the archive and the question, The main goal here is to determine whether you need to decompose the question in order to solve it. Note that block mean a LLM that use for specifical objectives by specifclaied setting (instruction, tempreture...)

Your objective is to 

Step 1: Determine whether you need to propose a decomposition or whether you can directly use one of the given blocks to solve the entire task. In the available blocks archive, you will see the blocks along with their outputs and answers for the given question. Carefully consider the features of each block, the question, and the answer, and then determine whether a single block is already sufficient to solve the entire task or whether you need to further decompose the question to achieve a better answer using the blocks.

Step 1.1: If you decided not to decompose, directly copy the answer to the 'code' field, in the format of `block_name:answer`. block_name is one of: CoT, CoT_SC, Debate, Refine. For example, if the answer is 10 with CoT, you should write <code>CoT:10</code> in the 'code' field. In this case, you do not need to write any code as the code is the same as the available blocks archive and you already have the answer.

Step 1.2.1: If and only if you select decomposition, you need to perform task decomposition (that is, if you select no decomposition, you do not need to do task decomposition). Specfically, decompose the give question significantly so that the sub-architecture (or node or block) can perform each of the sub-tasks. The output should be sub-task 1, sub-task 2, ... sub-task n. Do not solve the task for the sub-architecture and do not leak the expected answer in your sub-task description/instruction/question (a short-cut like 'output exactly the following...' is also leakage and should be avoided). Instead, decompose the task that easy enough for the sub-architecture to solve. You need to justify how these sub-tasks can achieve the final answer to the orginal questions.

Make sure 

(a) Include sub-task ID  and 'Based on (task i)' in the instruction of a sub-task. 
For example, 
Similarly, if sub-task 2 requires the output of task 1, then sub-task 2's instruction should be
'Sub-task 2: Based on the outputs from sub-task 1, ....(origin sub-task 1's instruction)'

Similarly, if sub-task 3 requires the output of task 1 and 2, then sub-task 3's instruction should be
'Sub-task 3: Based on the outputs from sub-task 1 and 2, ....(origin sub-task 3's instruction)'
This helps each sub-task connects to its prerequisite sub-tasks so that there is enough information to solve it.

(b) Each sub-task should be specific and detailed enough to solve and to help achieve the final answer to the given question. The output should be helpful to solve the next sub-task. You need to include details steps (but not the answer) to the sub-task 
For example,
`Sub-task 3: Based on the output of sub-task 1 and 2....`
You can see it clearly states 'based on what sub-tasks'

(c) The answer to the last sub-task should be the same as the answer to the final question, so that the architecture successfully solve the complext question by solveing each of the sub-task. 

(d) Decomposition itself should not be included in the architecture as the question has been decomposed. Do not assign one block to perform all the sub-tasks (if you put all decomposed sub-tasks into a single instruction for an block, it is very wrong). Instead, assign different block to address each of the sub-task instead.

Step 1.2.2: If and only if you select decomposition, and finish step 1.2.1, Given the resulting sub-task 1, sub-task 2, ... sub-task n, always use CoT to address each of them.


IMPORTANT:

Your code should implment the available blocks given in the archive (the 'code' field of blocks) as it-is without modication: Do not propose new blocks or modify existing ones.

{util_code}

# Output Instruction and Example:
The first field should be "thought", and it should capture your thought process for and it should capture your thought process for the above steps. 

In the "thought" field, include the following:

(1) **No decomposition or decomposition **: This corresponding to the step 1. Sepcfically, If you select not to decompose, you need to do step 1.1 and then output: 'No decomposition'. If you select decomposition, you need to do step 1.2.1 and 1.2.2, there output insructions are as follows.


(2) **Decomposion**:. This corresponding to the step 1.2.1, so you only need to do this if you select decomposition (This is, if you select No decomposition, you do not need to do task decomposition). Sepcfically, you decompose the given task into multiple manageable sub-tasks. Explain in details how do you decompose the question and how such decomposition is complete for solving the original task and each sub-task is more solvable and easier than the original task.

(3) **CoT to solve each sub-task**: This corresponding to the step 1.2.2, so you only need to do this if you select decomposition and after finishing step 1.2.1. Sepcfically given the resulting sub-task 1, sub-task 2, ... sub-task n from step 1.2.1, always use CoT to address each of them. IMPORTANT: DO NOT mix this step with step 1.1. In step 1,1, you select no decomposition and you need to use CoT in the given archive. but here, you select decomposition and has already decomposed the task into sub-tasks, and you always use CoT to address the sub-task.

The second field ("name") corresponds to the name of your No decomposition or decomposition design. 

Finally, the last field ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project (so do not implement any other part), so please implement complete, reliable, reusable code snippets. You cannot call the available blocks (e.g., COT, CoT_SC) by its name, but must implement them as it is in the achive. If the block is handling a sub-task, add 'sub_task=True' when calling the block.

Here is an example of the output format for the No decomposition or decomposition:

[EXAMPLE]

1. You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. DON'T try to use some function that doesn't exisit.
2. it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Infos.content. 
3. DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

{wrong_implementation}


# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to solve the question without decomposition or decomposing the question into sub-tasks and using CoT to address each of them. Do not try to propose new block or modify the available block, but block setting like instruction, tempreture are allowed to modify.
Observe the discovered blocka carefully and think about what insights, lessons, or stepping stones can be learned from them.
You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the no decomposition or decomposition.

Below is the question to solve:\n\n[QUESTION]
Below is the available blocks and corresponding output (which may be cut off using '...' due to the length limit) and answer for the question:\n\n

CoT: [CoT]\n\n
CoT_SC: [CoT_SC]\n\n
Debate: [Debate]\n\n
Refine: [Refine]\n\n
"""

# if determine to use, no need to execute. sub-agent will use it as the entire task but still able to change the architecture and block setting
# the 'thinking' could be very long
# TODO: block.py is not updated yet
# TODO: the problem is, judge is very hard
# Perhaps we shold do SFT?
#TODO: LoRA does not speed up the execution time.

