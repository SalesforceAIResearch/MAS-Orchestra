# [ToolOrchestra step 3] User prompt for harmony_multiturn (dynamic plan-execute).
# Replaces the single-shot <agent> plan of propose_harmony_minmal.py with turn-by-turn tool calls.

base = """
Please solve the question step by step. You may solve it yourself or delegate to agents (CoTAgent, SCAgent, DebateAgent, ReflexionAgent) by calling them as tools. After each agent result you decide the next step: accept the result, verify it with another agent, delegate a remaining sub-task, or solve the rest yourself.
Finish with a turn that contains no tool call and puts the final answer to the original question in the <answer> tag.

For example,

If you decide to solve the entire task yourself, you will output a single turn:

Question: What is (20+9)*(30+7)?

<thinking>
The problem only requires basic arithmetic. (20+9)*(30+7) = 600 + 140 + 270 + 63 = 1073.
</thinking>
<answer>1073</answer>

If you decide to delegate and then verify, the conversation looks like this:

Question: Compute the definite integral of (2x + 5) dx from 0 to 3.

Turn 1 (you):
<thinking>
This needs symbolic integration. I will delegate the original question to a CoTAgent.
</thinking>
<tool_call>
{"name": "CoTAgent", "arguments": {"agent_input": ""}}
</tool_call>

Turn 1 (environment):
<tool_response>
[CoTAgent result]
Final answer from CoTAgent: 24
Reasoning summary (truncated): ...
</tool_response>

Turn 2 (you):
<thinking>
The antiderivative is x^2 + 5x, so 9 + 15 = 24. This matches the CoTAgent result, so no further agent is needed.
</thinking>
<answer>24</answer>

Another example, with a sub-task and a disagreement:

Question: Compute the square root of the average of the first five prime numbers (2, 3, 5, 7, 11), rounded to three decimals.

Turn 1 (you):
<thinking>
The average is 28 / 5 = 5.6. The remaining step, sqrt(5.6) to three decimals, needs numeric care. I will ask two agents in parallel.
</thinking>
<tool_call>
{"name": "ReflexionAgent", "arguments": {"agent_input": "Compute sqrt(5.6) to 3 decimals"}}
</tool_call>
<tool_call>
{"name": "SCAgent", "arguments": {"agent_input": "Compute sqrt(5.6) to 3 decimals"}}
</tool_call>

Turn 1 (environment):
<tool_response>
[ReflexionAgent result]
Final answer from ReflexionAgent: 2.366
...
</tool_response>
<tool_response>
[SCAgent result]
Final answer from SCAgent: 2.366
...
</tool_response>

Turn 2 (you):
<thinking>
Both agents agree on 2.366 (2.366^2 = 5.598). This is the final answer.
</thinking>
<answer>2.366</answer>

If agents disagree, call a DebateAgent (with debate_roles) or a ReflexionAgent on the disputed step before answering.

Below is the question to solve:\n\n[QUESTION]
"""
