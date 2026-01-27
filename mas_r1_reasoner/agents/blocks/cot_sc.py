import inspect


# %%%%%%%%%%%%%%%%%%%% CoT_SC %%%%%%%%%%%%%%%%%%%%

async def forward(self, taskInfo):
    """Forward function that wraps SCAgent from blocks_harmony.
    
    This is used by the eval_building_blocks system which expects a 'forward' function.
    """
    from mas_r1_reasoner.agents.blocks_harmony.cot_sc import SCAgent
    return await SCAgent(self, taskInfo, self.node_model)  

func_string = inspect.getsource(forward)

COT_SC = {"thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.",
          "name": "Self-Consistency with Chain-of-Thought (CoT_SC)",
            "code": """{func_string}""".format(func_string=func_string)
              }