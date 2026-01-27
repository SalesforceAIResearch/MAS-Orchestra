import inspect


# %%%%%%%%%%%%%%%%%%%% LLM-Debate (collabrative) %%%%%%%%%%%%%%%%%%%%


async def forward(self, taskInfo):
    """Forward function that wraps DebateAgent from blocks_harmony.
    
    This is used by the eval_building_blocks system which expects a 'forward' function.
    """
    from mas_r1_reasoner.agents.blocks_harmony.llm_debate import DebateAgent
    return await DebateAgent(self, taskInfo, self.node_model, self.debate_role)

func_string = inspect.getsource(forward)

LLM_debate = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "LLM Debate (Debate)",
    "code": """{func_string}""".format(func_string=func_string)

}