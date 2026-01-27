"""
Online Search Initialization Module

Handles initialization of online web search resources including:
- Chat model for LLM-based summarization of web pages
"""

import os
from mas_r1_reasoner.agents.shared_vars import get_global


def initialize_online_resources(agent_system):
    """
    Initialize online search resources for AgentSystem.
    
    This function is called once during AgentSystem initialization if web_search_type == "online".
    It sets up the chat model used for summarizing web pages retrieved from live search APIs.
    
    Args:
        agent_system: AgentSystem instance to initialize resources for
    """
    print(f"🔧 AgentSystem: Initializing online chat model for web search...")
    
    chat_model = _initialize_chat_model(agent_system.node_model)
    if chat_model:
        agent_system.online_chat_model = chat_model


def _initialize_chat_model(node_model):
    """Initialize ChatTogether model for webpage summarization in online mode."""
    try:
        from langchain_together import ChatTogether
        
        # Map model names to their full API names (matching grpo_trainer.yaml model_sampler_map)
        # This ensures consistency with BaseDatasetProcessor and model samplers
        model_mapping = {
            "gpt-4o": "gpt-4o",
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-nano": "gpt-4.1-nano",
            "gpt-5-nano": "gpt-5-nano",
            "llama-3.3-70b-instr": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "qwen-2.5-32b-instr": "qwen-2.5-32b-instr",
            "qwen-2.5-7b-instr": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen-2.5-72b-instr": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        }
        
        # Parse model name: if it already has a prefix, use as-is; otherwise map it
        if "/" in node_model:
            # Model already has a prefix (e.g., "openai/gpt-oss-120b")
            model_name = node_model
        else:
            # Try to find in mapping, or add openai/ prefix as fallback
            model_name = model_mapping.get(node_model, f"openai/{node_model}")
        
        # Get reasoning_effort from global config
        reasoning_effort = get_global("global_reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = "low"  # Fallback default
        
        # Configure for online search (matching web_search.py settings exactly)
        chat_model_kwargs = {
            "model": model_name,
            "temperature": 0.5,
            "together_api_key": os.getenv("TOGETHER_API_KEY"),
            "reasoning_effort": reasoning_effort,
            "timeout": 300,
            "max_tokens": 120000,  # Matching web_search.py
            "max_retries": 3,
        }
        
        chat_model = ChatTogether(**chat_model_kwargs)
        print(f"   ✅ Online chat model initialized: {model_name}")
        return chat_model
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not initialize online chat model: {e}")
        return None

