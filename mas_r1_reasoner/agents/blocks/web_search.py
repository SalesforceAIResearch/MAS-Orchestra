

import inspect

async def forward(self, agent_input) -> str:
    """Forward function that wraps WebSearchAgent from blocks_harmony.
    
    This is used by the eval_building_blocks system which expects a 'forward' function.
    It dynamically selects online or offline web search based on global_web_search_type.
    """
    from mas_r1_reasoner.agents.shared_vars import get_global
    
    # Dynamically get the correct function based on current global setting
    web_search_type = get_global("global_web_search_type")
    
    if web_search_type == "offline":
        print(f"Using offline web search")
        from mas_r1_reasoner.agents.blocks_harmony.web_search_offline import WebSearchOfflineAgent
        return await WebSearchOfflineAgent(self, agent_input, self.node_model)
    else:
        print(f"Using online web search")
        from mas_r1_reasoner.agents.blocks_harmony.web_search import WebSearchAgent as OnlineSearchAgent
        return await OnlineSearchAgent(self, agent_input, self.node_model)


func_string = inspect.getsource(forward)

WebSearch = {
    "thought": "Web search allows models to access up-to-date information from the internet and provide answers with sourced citations.",
    "name": "Web Search Agent (WebSearchAgent)",
    "code": """{func_string}""".format(func_string=func_string)
}

# the code will be executed