prompt = '''
# Overview
You are an expert machine learning researcher testing various agentic systems. Given a set of blocks in the archive and the question. Note that architecture can contain multiple blocks, and block mean a LLM that use for specifical objectives by specifclaied setting (instruction, tempreture...)

The main goal here is to design a single block without decomposition or decomposition with CoT block to solve the question.

Your objective is to 

Step 1: Determine whether you need to propose decomposition with CoT block or you can simply use the ONE of the given blocks to solve the entire task. 

Step 1.1: If you select "single block", you need to select from CoT, CoT-SC, Debate, or Reflexion in the given archive. The features of each block is described in the given archive (the 'thought' field of each block). Think carefully about the features of each block and the question to determine which block is the most suitable to solve the question.

Step 1.2.1: If and only if you select decomposition, you need to perform task decomposition (that is, if you select block, you do not need to do task decomposition). Specfically, decompose the give question significantly so that the sub-architecture (or node or block) can perform each of the sub-tasks. The output should be sub-task 1, sub-task 2, ... sub-task n. Do not solve the task for the sub-architecture and do not leak the expected answer in your sub-task description/instruction/question (a short-cut like 'output exactly the following...' is also leakage and should be avoided). Instead, decompose the task that easy enough for the sub-architecture to solve. You need to justify how these sub-tasks can achieve the final answer to the orginal questions.

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

Step 1.2.2: If and only if you select decomposition, and finish step 1.2.1, Given the resulting sub-task 1, sub-task 2, ... sub-task n, always use CoT to address each of them.


IMPORTANT:

1. Decomposition itself should not be included in the architecture as the question has been decomposed at step (1). Do not assign one block to perform all the sub-tasks (if you put all decomposed sub-tasks into a single instruction for an block, it is very wrong). Instead, assign different block to address each of the sub-task instead.

2. Your code should implment the available blocks given in the archive (the 'code' field of blocks) as it-is without modication: Do not propose new blocks or modify existing ones.


# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'prompt', 'sub_tasks', 'agents', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.
{str(request_keys)}
DO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!
"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    """
    Function to get JSON response from GPT model.
    
    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.
    
    Returns:
    - dict: The JSON response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase():
    """
    Attributes:
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model=None, temperature=None) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature
        # give each instance a unique id
        self.id = random_id()
        

    def generate_prompt(self, input_infos, instruction, is_sub_task=False) -> str:

        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. {global_output_description}" for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "

" + FORMAT_INST(output_fields_and_description)

        # print('is_sub_task: ',is_sub_task)
        

        # construct input infos text
        input_infos_text = ''
        prev_prompt = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, prompt, _, _, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                if is_sub_task: 
                    # input_infos_text += f'Giving the original question: 

{content}

, and below sub-task questions and answers, please solve the sub-task: {instruction}

Sub-task questions and answers (if any):

'
                    input_infos_text += f'{instruction}

Previous sub-task questions and answers (if any):

'
                else:
                    # continue # TODO: make sure it can deal with sub-tasks
                    input_infos_text += f'{content}

'
            elif iteration_idx != -1:
                if is_sub_task and prompt is not None and prompt != prev_prompt: 
                    # print('prompt: ',prompt)
                    # pattern = r"please solve the sub-task:\s*(.*?)\s*

Sub-task questions and answers"
                    pattern = r"\s*(.*?)\s*

Previous sub-task questions"

                    sub_question = prompt[-1]['content']
                    match = re.search(pattern, sub_question, re.DOTALL)                                        
                    input_infos_text += f'### {match.group(1)} 

 ### {field_name} #{iteration_idx + 1} by {author}:
{content}

'
                    prev_prompt = prompt
                else:
                    input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:
{content}

'
            else:
                if is_sub_task and prompt is not None and prompt != prev_prompt: 
                    # print('prompt: ',prompt)
                    pattern = r"\s*(.*?)\s*

Previous sub-task questions"
                    sub_question = prompt[-1]['content']
                    match = re.search(pattern, sub_question, re.DOTALL)
                    input_infos_text += f'### {match.group(1)} 

 ### {field_name} by {author}:
{content}

'
                    prev_prompt = prompt # we do not want to duplicate the prompt
                else:
                    input_infos_text += f'### {field_name} by {author}:
{content}

'

        if is_sub_task: 
            prompt = input_infos_text # instruction (sub-task in above)
        else:
            prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False) -> dict:

        system_prompt, prompt = self.generate_prompt(input_infos, instruction, is_sub_task=is_sub_task)

        prompt = [
            _pack_message(content=system_prompt, role="system"),
            _pack_message(content=prompt, role="user")]
        # use system prompt

        response_json = get_json_response_from_gpt(prompt, self.model)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, prompt, None, None, iteration_idx, None)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1, is_sub_task=False):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx,  is_sub_task=is_sub_task)


class AgentSystem:
    """
    Fill in your code here.
    """
    def forward(self, taskInfo) -> Union[Info, str]:
        """
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Info): Your FINAL Answer. Return namedtuple Info returned from self.make_final_answer.
        """
        pass
```
# Block Archive
Here is the archive of the available blocks:

[{"thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.", "name": "Chain-of-Thought (CoT)", "code": "def forward(self, taskInfo):\n    # Instruction for the Chain-of-Thought (CoT) approach\n    # It is an important practice that allows the LLM to think step by step before solving the task.\n    cot_instruction = self.cot_instruction\n\n    # Instantiate a new LLM agent specifically for CoT\n    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.\n    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent',  model=self.node_model, temperature=0.5)\n\n    # Prepare the inputs for the CoT agent\n    # The input should be a list of Info, and the first one is often the taskInfo\n    cot_agent_inputs = [taskInfo]\n\n    # Get the response from the CoT agent\n    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)\n    final_answer = self.make_final_answer(thinking, answer)\n    \n    # Return only the final answer\n    return final_answer   \n"},
{"thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.", "name": "Self-Consistency with Chain-of-Thought (CoT-SC)", "code": "def forward(self, taskInfo):\n    # Instruction for step-by-step reasoning\n    cot_instruction = self.cot_instruction\n    N = self.max_sc # Number of CoT agents\n\n    # Initialize multiple CoT agents with a higher temperature for varied reasoning\n    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent',  model=self.node_model, temperature=0.5) for _ in range(N)]\n\n    # Majority voting function to select the most common answer\n    from collections import Counter\n    def majority_voting(answers):\n        return Counter(answers).most_common(1)[0][0]\n    \n    thinking_mapping = {}\n    answer_mapping = {}\n    possible_answers = []\n    for i in range(N):\n        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)\n        possible_answers.append(answer.content)\n        thinking_mapping[answer.content] = thinking\n        answer_mapping[answer.content] = answer\n\n    # Ensembling the answers from multiple CoT agents\n    answer = majority_voting(possible_answers)\n    print('possible_answers: ',possible_answers)\n\n    thinking = thinking_mapping[answer]\n    answer = answer_mapping[answer]\n\n    final_answer = self.make_final_answer(thinking, answer)\n\n    return final_answer  \n"},
{"thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.", "name": "Self-Refine (Reflexion)", "code": "def forward(self, taskInfo):\n    # Instruction for initial reasoning\n    cot_initial_instruction = self.cot_instruction\n\n    # Instruction for reflecting on previous attempts and feedback to improve\n    cot_reflect_instruction = \"Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.\"\n    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5)\n\n    # Instruction for providing feedback and correcting the answer\n    critic_instruction = \"Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.\"\n    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.5)\n    \n    N_max = self.max_round # Maximum number of attempts\n    \n    # Initial attempt\n    cot_inputs = [taskInfo]\n    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)\n\n    for i in range(N_max):\n        # Get feedback and correct status from the critic\n        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)\n        if correct.content == 'True':\n            break\n            \n        # Add feedback to the inputs for the next iteration\n        cot_inputs.extend([thinking, answer, feedback])\n\n        # Reflect on previous attempts and refine the answer\n        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)\n\n    final_answer = self.make_final_answer(thinking, answer)\n\n    return final_answer\n"},
{"thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.", "name": "LLM Debate (Debate)", "code": "def forward(self, taskInfo):\n    # Instruction for initial reasoning\n    debate_initial_instruction = self.cot_instruction\n\n    # Instruction for debating and updating the solution based on other agents' solutions\n    debate_instruction = \"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Put your thinking process in the 'thinking' field and the updated answer in the 'answer' field. \"\n    \n    # Initialize debate agents with different roles and a moderate temperature for varied reasoning\n    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent',  model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]\n\n    # Instruction for final decision-making based on all debates and solutions\n    final_decision_instruction = \"Given all the above thinking and answers, reason over them carefully and provide a final answer. Put your thinking process in the 'thinking' field and the final answer in the 'answer' field.\"\n    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent',  model=self.node_model, temperature=0.5)\n\n    max_round = self.max_round # Maximum number of debate rounds\n    all_thinking = [[] for _ in range(max_round)]\n    all_answer = [[] for _ in range(max_round)]\n\n    # Perform debate rounds\n    for r in range(max_round):\n        for i in range(len(debate_agents)):\n            if r == 0:\n                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)\n            else:\n                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]\n                thinking, answer = debate_agents[i](input_infos, debate_instruction)\n            all_thinking[r].append(thinking)\n            all_answer[r].append(answer)\n    \n    # Make the final decision based on all debate results and solutions\n    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)\n    final_answer = self.make_final_answer(thinking, answer)\n\n    return final_answer\n"}]



# Output Instruction and Example:
The first field should be "thought", and it should capture your thought process for and it should capture your thought process for the above steps. 

In the "thought" field, include the following:

(1) **Single Block (CoT, CoT-SC, Debate, or Reflexion) or Decomposition with CoT Block **: This corresponding to the step 1. Sepcfically, If you select single block, you need to do step 1.1 and then output the block you selected (Single Block: (one of CoT, CoT-SC, Debate, Reflexion based on your judgement)). If you select decomposition with CoT Block, you need to do step 1.2.1 and 1.2.2, there output insructions are as follows.


(2) **Decomposion**:. This corresponding to the step 1.2.1, so you only need to do this if you select decomposition with CoT Block (This is, if you select single block, you do not need to do task decomposition). Sepcfically, you decompose the given task into multiple manageable sub-tasks. Explain in details how do you decompose the question and how such decomposition is complete for solving the original task and each sub-task is more solvable and easier than the original task.

(3) **CoT to solve each sub-task**: This corresponding to the step 1.2.2, so you only need to do this if you select decomposition with CoT Block and after finishing step 1.2.1. Sepcfically given the resulting sub-task 1, sub-task 2, ... sub-task n from step 1.2.1, always use CoT to address each of them. IMPORTANT: DO NOT mix this step with step 1.1. In step 1,1, you select block and you need to select from one of CoT, CoT-SC, Debate, Reflexion in the given archive. but here, you select decomposition with CoT Block and has already decomposed the task into sub-tasks, and you always use CoT to address the sub-task.

The second field ("name") corresponds to the name of your single block or decomposition. 

Finally, the last field ("code") corresponds to the exact "forward()" function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project (so do not implement any other part), so please implement complete, reliable, reusable code snippets. You cannot call the available blocks (e.g., COT, COT-SC) by its name, but must implement them as it is in the achive. If the block is handling a sub-task, add 'sub_task=True' when calling the block.

Here is an example of the output format for the single block or decomposition with CoT Block:

{"thought": "\n    **Insights:**\nProvide your reasoning for each step, along with an explanation of the overall concept behind the design. \n\n    **Single Block without Decomposition or Decomposition with CoT Block**:\n This corresponding to the step 1. Sepcfically, you need to first determine whether you need to propose decomposition or you can use the existing block to solve the task. You need to carefully consider the features of each block and the question to determine which block is the most suitable to solve the question. Output \"Single Block: (the block you selected)\" or \"decomposition with CoT Block\".\n\n    **Decomposion:**\n This corresponding to the step 1.2.1, so you only need to do this if you select decomposition with CoT Block. Sepcfically, you decompose the given task into multiple manageable sub-tasks. Please explain \n    (1) How do you make sure the sub-task is easier enough and thus solavable by the given or proposed blocks and agents; \n    (2) Justify how these sub-tasks can achieve the final answer to the orginal questions.\n\n    **CoT to solve each sub-task:**\n This corresponding to the step 1.2.2, so you only need to do this if you select decomposition with CoT Block and after finishing step 1.2.1. Sepcfically given the resulting sub-task 1, sub-task 2, ... sub-task n from step 1.2.1, always use CoT to address each of them.\n\n    \"**Implementation:**\n describe the implementation step by step.\"\n    ", "name": "Name of your selected single block or designed decomposition", "code": "def forward(self, taskInfo):\n    from collections import Counter # must have this and always make sure you import everything needed\n    # Your code here. IMPORTANT  \n    # (1) If you select single block, you need to implement the single block from the code in the available blocks archive. If you select decomposition with CoT Block, follow the instructions below.\n    # (2) You cannot call the existing architecture from the archive but you have to implment it from the code in the available blocks archive\n    # for example:\n    # you cannot call 'COT' but you have to implement it eactly the same as the code in the available blocks archive. Make sure you ACTUALLY IMPLEMENET them\n    # You should always use CoT to address each of the sub-task.\n\n    # (3) To creat an agent, call the LLMAgentBase, must have 'model=self.node_model' in the parameters\n    # the return of the call is always the same as the output fields,\n    # for example: \n    # reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent', model=self.node_model)\n\n    # (4) Since the agent is handling with a sub_question made by question decomposition, use 'is_sub_task=True' when calling the agent.\n\n    # (5) You need to specify the sub-task dependencies by specify the dependent sub-tasks in the instruction. You also need to give each sub-task an ID. This means you need to state clearly what the current sub-task is based on and also give all required information (thinking and answer) in the input task info\n    # for example:\n    # cot_instruction = (\"Sub-task 3: Based on the output of sub-task 1 and sub-task 2.....\")\n    # thinking, answer = reasoning_agent([taskInfo] + [thinking1, answer1, thiking2, answer2, ..., ], cot_instruction, is_sub_task=True)\n\n\n    # (6) You need to keep tack of sub-task output, for each sub-task output, please append it to a list named `sub_tasks` (please initialize it as `sub_tasks = []` at the beining of your function) so that we can keep track of the performance of each sub-task. When you do self.make_final_answer, please also add `sub_tasks` as the second last parameterss \n    # for example: \n    # sub_tasks = []\n    # ...\n    # thinking, answer = reasoning_agent([taskInfo] + [sub-task thinking, sub-task answer], reasoning_instruction, is_sub_task=True)\n    # sub_tasks.append(f'the output of sub task n: thinking {thinking.content}; final answer: {answer.content}')\n\n    # (7) You need to keep track of agent, for each agent inside the sub-architecture (or block or node) output, please append it to a list named `agents` (please initialize it as `agents = []` at the beining of your function) so that we can keep track of the performance of each agents. It should contain the agent name (if setting is important to identify the agent, please include as well, e.g., round, ID), agent purpose, and the thinking and output of the agent. When you do self.make_final_answer, please also add `agents` as the last parameterss \n\n    # Take CoT as an example:\n    # cot_instruction = \"Sub-task i: Based on the output of...\"\n    # cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5)\n\n    # (8) Put the saved sub_tasks and agents to the final elf.make_final_answer. Make sure you have `is_sub_task=True` when calling an agent, and keep track of `sub_task`, `agents`, include `sub_task` dependency and detailed steps for the sub-task ('Sub-task i: based on sub-task,') in the sub-task instruction, and actually implmenet the blocks by yourself (for-loop if COT-SC, Debate and Reflextion).\n\n    final_answer = self.make_final_answer(thinking, answer, sub_tasks, agents)\n    # Return only the final answer\n    return final_answer\n"}

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. DON'T try to use some function that doesn't exisit.
Also, it could be helpful to set the LLM's role and temperature to further control the LLM's response. Note that the LLMAgentBase() will automatically parse the output and return a list of "Infos". You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.


## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0, None, None, -1, None)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0, None, None, -1, None)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '
'.join(all_thinking)
aggregated_answers = '
'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', None, None, None, 0, None)
```

5. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer   
```
You MUST return final_answer returned by ```final_answer = self.make_final_answer(thinking, answer)```, instead of answer only.

6. This is WRONG when handling sub-tasks: ```
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
```
You MUST add sub_task=True, when handling a sub question made by question decomposition

7. This is WRONG when handling sub-tasks: ```
reasoning_instruction = '...'
```
You MUST clealy states what sub task it is, it can be solved based on the output of what sub-tasks and what are the steps to solve the sub-tasks

8. This is WRONG: ```
cot_sc_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Self-Consistency Agent', model=self.node_model, temperature=0.5)
thinking3, answer3 = cot_sc_agent([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction, is_sub_task=True)
```
You MUST ACTUALLY IMPLEMENT the achitecture (here self-consistency). Name an agent to 'sc' does not implment the structure (e.g. you will need to actually implmemnt the for-loop).
CORRECT example: ```
    from collections import Counter
    def majority_voting(answers):
        return Counter(answers).most_common(1)[0][0]

    thinking_mapping = {}
    answer_mapping = {}
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.append(answer.content)
        thinking_mapping[answer.content] = thinking
        answer_mapping[answer.content] = answer

    # Ensembling the answers from multiple CoT agents
    answer = majority_voting(possible_answers)
```

9. This is WRONG: ```
reflexion_agent = LLMAgentBase(['thinking', 'answer'], 'Reflexion Agent', model=self.node_model, temperature=0.5)
thinking3, answer3 = reflexion_agent([taskInfo, thinking1, answer1, thinking2, answer2], reflexion_instruction, is_sub_task=True)
```
You MUST ACTUALLY IMPLEMENT the achitecture (here reflexion). Name an agent to 'reflexion' does not implment the structure (e.g. you will need to actually implmemnt the for-loop).
CORRECT example: ```
    N_max = self.max_round # Maximum number of attempts
    
    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
```

10. This is WRONG: ```
debate_agent = LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, temperature=0.5)
thinking3, answer3 = debate_agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction, is_sub_task=True)
```
You MUST ACTUALLY IMPLEMENT the achitecture (here debate). Name an agent to 'debate' does not implment the structure (e.g. you will need to actually implmemnt the for-loop).
CORRECT example: ```
for r in range(max_round):
    for i in range(len(debate_agents)):
        if r == 0:
            thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
        else:
            input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
            thinking, answer = debate_agents[i](input_infos, debate_instruction)
        all_thinking[r].append(thinking)
        all_answer[r].append(answer)
```
You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_instruction = 'Sub-task i: Based on the output of sub-task i and j, ....'
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
final_answer = self.make_final_answer(thinking, answer)

# Return only the final answer
return final_answer   
```



# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to solve the question by selecting one of the available blocks or decomposing the question into sub-tasks and using CoT to address each of them. Do not try to propose new block or modify the available block, but block setting like instruction, tempreture are allowed to modify.
Observe the discovered blocka carefully and think about what insights, lessons, or stepping stones can be learned from them.
You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the single-block or decomposition with CoT Block.

Below is the question to solve:\n\n[QUESTION]
'''