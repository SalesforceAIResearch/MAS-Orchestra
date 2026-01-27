developer_prompt = """

Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <edge>: definition of edges

Model (the Large lanaguge model used in agent:
- gpt-oss-120b: state-of-the-art language models that deliver strong real-world performance at low cost. outperform similarly sized open models on reasoning tasks, demonstrate strong tool use capabilities, and are optimized for efficient deployment on consumer hardware. It was trained using a mix of reinforcement learning and techniques informed by OpenAI's most advanced internal models, including o3 and other frontier systems. It achieves near-parity with OpenAI o4-mini on core reasoning benchmarks. It also perform strongly on tool use, few-shot function calling, CoT reasoning (as seen in results on the Tau-Bench agentic evaluation suite) and HealthBench (even outperforming proprietary models like OpenAI o1 and GPT-4o). 131,072 context window, 131,072 max output tokens, Jun 01, 2024 knowledge cutoff, Reasoning token support
- qwen2.5_7b_instr: latest series of Qwen large language models (2024). For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2: Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains. Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots. Long-context Support up to 128K tokens and can generate up to 8K tokens. Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more. Context Length: 131,072 tokens

MASness Levels:
- medium: one or more agents delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_id>...</agent_id> (a unique id for the agent)
    <agent_name>...</agent_name> (select one of the agents: CoTAgent)
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

"""
