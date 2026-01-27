developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <answer>: final user-facing answer

Model (the Large lanaguge model used in agent:
- gpt-4.1-nano: Fastest, most cost-efficient version of GPT-4.1. GPT-4.1 nano excels at instruction following and tool calling. It features a 1M token context window, and low latency without a reasoning step. 1,047,576 context window. 32,768 max output tokens. Jun 01, 2024 knowledge cutoff
- gpt-oss-120b: state-of-the-art language models that deliver strong real-world performance at low cost. outperform similarly sized open models on reasoning tasks, demonstrate strong tool use capabilities, and are optimized for efficient deployment on consumer hardware. It was trained using a mix of reinforcement learning and techniques informed by OpenAI's most advanced internal models, including o3 and other frontier systems. It achieves near-parity with OpenAI o4-mini on core reasoning benchmarks. It also perform strongly on tool use, few-shot function calling, CoT reasoning (as seen in results on the Tau-Bench agentic evaluation suite) and HealthBench (even outperforming proprietary models like OpenAI o1 and GPT-4o). 131,072 context window, 131,072 max output tokens, Jun 01, 2024 knowledge cutoff, Reasoning token support

MASness Levels:
- minimal: direct solve or at most one agent
- medium:  one or more agents delegation
- high: complex multi-agent delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_name>...</agent_name> (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent)
    <agent_description>...</agent_description>
    <required_arguments> (make sure all required parameters are set. Must follow XML format)
        <...>...</...>
        <...>...</...>
    </required_arguments>
    <agent_output_id>
    ...
    </agent_output_id>
</agent>

Available Agents:

CoTAgent: [COT]

SCAgent: [COT_SC]

DebateAgent: [Debate]

ReflexionAgent: [Reflexion]

"""
