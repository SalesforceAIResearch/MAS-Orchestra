import inspect

# %%%%%%%%%%%%%%%%%%%% relexion (generator-evaluator) %%%%%%%%%%%%%%%%%%%%
async def forward(self, taskInfo):
    """Forward function that wraps ReflexionAgent from blocks_harmony.
    
    This is used by the eval_building_blocks system which expects a 'forward' function.
    """
    from mas_r1_reasoner.agents.blocks_harmony.reflexion import ReflexionAgent
    return await ReflexionAgent(self, taskInfo, self.node_model)


func_string = inspect.getsource(forward)


Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.",
    "name": "Self-Refine (Reflexion)",
    "code": """{func_string}""".format(func_string=func_string)
}



