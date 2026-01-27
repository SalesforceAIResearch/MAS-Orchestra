global global_output_description 
global global_max_workers
global global_task_queue
global global_score_compute
global global_max_round
global global_max_sc
global global_debate_role
global global_cot_instruction
global global_node_model
global global_answers
global global_questions
global global_use_oracle_verifier
global global_judge_path
global global_reponse_path
global global_example_id
global global_n
global global_response_dict
global global_dataset
global global_instance_id
global global_code_snippet
global global_shorten_context
global global_format_choice
global global_merge_context
global global_COST_TOTAL
global global_no_decompose
global global_no_meta_reward
global global_decompose_only
global global_architecture_only
global global_architecture_only_sequential
global global_model_sampler_map
global global_ref_model_sampler_map
global global_shared_agent_system
global global_init_archive
global global_enable_tree_architecture
global global_include_blocks
global global_add_judge
global global_multiply_processes
global global_max_ray_workers
global global_problem_type
global global_eval_building_blocks
global global_dataset_name
global global_use_llm_judge
global global_web_search_type
global global_retrieval_method
global global_max_concurrent
global global_use_igsm_prompt
global global_igsm_variant
global global_use_long_horizon
global global_reasoning_effort
global global_max_tokens

# Declare your globals here (optional initial values)
global_vars = [
    "global_output_description",
    "global_max_workers",
    "global_task_queue",
    "global_score_compute",
    "global_max_round",
    "global_max_sc",
    "global_debate_role",
    "global_cot_instruction",
    "global_node_model",
    "global_answers",
    "global_questions",
    "global_use_oracle_verifier",
    "global_judge_path",
    "global_reponse_path",
    "global_example_id",
    "global_n",
    "global_response_dict",
    "global_dataset",
    "global_instance_id",
    "global_code_snippet",
    "global_FORMAT_INST",
    "global_model_sampler_map",
    "global_ref_model_sampler_map",
    "global_shorten_context",
    "global_merge_context",
    "global_format_choice",
    "global_COST_TOTAL",
    "global_no_decompose",
    "global_no_meta_reward",
    "global_decompose_only",
    "global_architecture_only",
    "global_architecture_only_sequential",
    "global_shared_agent_system",
    "global_init_archive",
    "global_enable_tree_architecture",
    "global_include_blocks",
    "global_add_judge",
    "global_multiply_processes",
    "global_known_prompt",
    "global_max_ray_workers",
    "global_problem_type",
    "global_eval_building_blocks",
    "global_dataset_name",
    "global_use_llm_judge",
    "global_web_search_type",
    "global_retrieval_method",
    "global_max_concurrent",
    "global_use_igsm_prompt",
    "global_igsm_variant",
    "global_use_long_horizon",
    "global_reasoning_effort",
    "global_max_tokens"
]

# Optionally initialize to None
for var in global_vars:
    globals()[var] = None

def set_global(name, value):
    if name in global_vars:
        globals()[name] = value
    else:
        raise NameError(f"{name} is not a recognized global variable.")

def get_global(name):
    if name in global_vars:
        return globals()[name]
    else:
        raise NameError(f"{name} is not a recognized global variable.")

def add_to_global_cost(cost):
    if "global_COST_TOTAL" in global_vars:
        globals()["global_COST_TOTAL"] += cost
    else:
        raise NameError("global_COST_TOTAL is not a recognized global variable.")