
import inspect


# %%%%%%%%%%%%%%%%%%%% COT %%%%%%%%%%%%%%%%%%%%
async def forward(self, taskInfo):
    """Forward function that wraps CoTAgent from blocks_harmony.
    
    This is used by the eval_building_blocks system which expects a 'forward' function.
    """
    from mas_r1_reasoner.agents.blocks_harmony.cot import CoTAgent
    return await CoTAgent(self, taskInfo, self.node_model)   

func_string = inspect.getsource(forward)

COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought (CoT)",
    "code": """{func_string}""".format(func_string=func_string)
}

