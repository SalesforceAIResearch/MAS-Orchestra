developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <edge>: definition of edges

Model (the Large lanaguge model used in agent:
- gpt-oss-120b: state-of-the-art language models that deliver strong real-world performance at low cost. outperform similarly sized open models on reasoning tasks, demonstrate strong tool use capabilities, and are optimized for efficient deployment on consumer hardware. It was trained using a mix of reinforcement learning and techniques informed by OpenAI's most advanced internal models, including o3 and other frontier systems. It achieves near-parity with OpenAI o4-mini on core reasoning benchmarks. It also perform strongly on tool use, few-shot function calling, CoT reasoning (as seen in results on the Tau-Bench agentic evaluation suite) and HealthBench (even outperforming proprietary models like OpenAI o1 and GPT-4o). 131,072 context window, 131,072 max output tokens, Jun 01, 2024 knowledge cutoff, Reasoning token support

MASness Levels:
- medium: one or more agents delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_id>...</agent_id> (a unique id for the agent)
    <agent_name>...</agent_name> (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent)
    <agent_description>...</agent_description>
    <required_arguments> (make sure all required parameters are set. Must follow XML format)
        <...>...</...>
        <...>...</...>
    </required_arguments>
</agent>

Edge Schema (single block; all fields required. Each pair defines a directed link: output of <from> → input of <to>. List ALL links here; use exactly one <edge> block per solution):
<edge>
    <from>...</from> (the source agent_id)
    <to>...</to> (the target agent_id)
</edge>

Available Agents:

CoTAgent: [COT]

SCAgent: [COT_SC]

DebateAgent: [Debate]

ReflexionAgent: [Reflexion]

"""
