# [ToolOrchestra step 3] Developer prompt for harmony_multiturn: the same four agents as harmony_minimal
# ([COT], [COT_SC], [Debate], [Reflexion] are filled from the init_archive in agents/common.py:get_prompt),
# described as callable tools instead of an <agent> XML schema.
developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning for the current turn
- <tool_call>: delegation of a (sub-)task to one agent
- <answer>: final user-facing answer, only in the last turn

Model (the Large language model used in every agent):
- gpt-oss-120b: state-of-the-art language models that deliver strong real-world performance at low cost. It achieves near-parity with OpenAI o4-mini on core reasoning benchmarks. 131,072 context window, Reasoning token support

Agent tool arguments:
- agent_input (all agents): "" to solve the original question, or a sub-task string. A sub-task is solved with the original question prepended as context.
- debate_roles (DebateAgent only): a JSON list of at least two distinct role names.

Available Agents:

CoTAgent: [COT]

SCAgent: [COT_SC]

DebateAgent: [Debate]

ReflexionAgent: [Reflexion]

"""
