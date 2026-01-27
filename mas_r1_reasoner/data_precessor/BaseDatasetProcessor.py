import torch
import copy
from typing import Dict, Tuple, Callable
from verl import DataProto
from mas_r1_reasoner.agents.sampler.chat_completion_sampler import ChatCompletionSampler
from mas_r1_reasoner.agents.sampler.together_completion_sampler import TogetherCompletionSampler
from mas_r1_reasoner.agents.sampler.vllm_completion_sampler import VLLMCompletionSampler
from mas_r1_reasoner.agents.shared_vars import set_global, get_global
from mas_r1_reasoner.agents.common import main_rank_print, get_prompt
from mas_r1_reasoner.agents.agent_system import AgentSystem, LLMAgentBase, Info


class BaseDatasetProcessor:
    """Base class for dataset-specific processors"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    @staticmethod
    def create_model_sampler_map(model_sampler_map_config, mock_output=False):
        """
        Create model_sampler_map from configuration.
        This is a shared function that can be used by both BaseDatasetProcessor and Ray workers.
        
        Args:
            model_sampler_map_config: Dictionary containing sampler configurations
            mock_output: Whether to use mock output for samplers
            
        Returns:
            Dictionary mapping model names to sampler instances
        """
        model_sampler_map = {}
        
        # Check if model_sampler_map_config is empty or None
        if not model_sampler_map_config:
            raise ValueError("No model_sampler_map specified in agent_config. Please specify at least one model in the model_sampler_map configuration.")
        
        for model_name, sampler_config in model_sampler_map_config.items():
            # Convert DictConfig to dict if needed
            if hasattr(sampler_config, '_content'):
                sampler_config = sampler_config._content
            
            if model_name == "ref":
                raise ValueError("Reference model is not supported in the model_sampler_map configuration.")
                
            elif isinstance(sampler_config, dict):
                sampler_type = sampler_config.get('type', 'ChatCompletionSampler')
                if sampler_type == 'ChatCompletionSampler':
                    # Create synchronous sampler
                    model_name_to_use = sampler_config.get('model', model_name)
                    if not model_name_to_use:
                        raise ValueError(f"Model name not specified for sampler '{model_name}'. Please specify a 'model' field in the sampler configuration.")
                    
                    model_sampler_map[model_name] = ChatCompletionSampler(
                        model=model_name_to_use,
                        temperature=sampler_config.get('temperature', 0.5),
                        system_message=sampler_config.get('system_message', None),
                        mock_output=mock_output
                    )
                elif sampler_type == 'TogetherCompletionSampler':
                    # Create Together AI sampler
                    model_name_to_use = sampler_config.get('model', model_name)
                    if not model_name_to_use:
                        raise ValueError(f"Model name not specified for sampler '{model_name}'. Please specify a 'model' field in the sampler configuration.")
                    
                    model_sampler_map[model_name] = TogetherCompletionSampler(
                        model=model_name_to_use,
                        temperature=sampler_config.get('temperature', 0.5),
                        system_message=sampler_config.get('system_message', None),
                        mock_output=mock_output
                    )
                elif sampler_type == 'VLLMCompletionSampler':
                    # Create VLLM sampler
                    model_name_to_use = sampler_config.get('model', model_name)
                    if not model_name_to_use:
                        raise ValueError(f"Model name not specified for sampler '{model_name}'. Please specify a 'model' field in the sampler configuration.")
                    
                    model_sampler_map[model_name] = VLLMCompletionSampler(
                        model=model_name_to_use,
                        temperature=sampler_config.get('temperature', 0.5),
                        system_message=sampler_config.get('system_message', None)
                    )
                else:
                    raise ValueError(f"Unsupported sampler type '{sampler_type}' for model '{model_name}'. Only 'ChatCompletionSampler', 'TogetherCompletionSampler', and 'VLLMCompletionSampler' are supported.")
            else:
                # Default sampler - but require a valid model name
                if not model_name or model_name == "":
                    raise ValueError("Model name cannot be empty. Please specify a valid model name in the model_sampler_map configuration.")
                
                model_sampler_map[model_name] = ChatCompletionSampler(
                    model=model_name,
                    temperature=0.5,
                    mock_output=mock_output
                )
        
        return model_sampler_map
    
    def _create_model_sampler_map(self, agent_config, mock_output=False):
        """Helper method to create model_sampler_map using the shared function"""
        model_sampler_map_config = agent_config.get('model_sampler_map', {})
        return self.create_model_sampler_map(model_sampler_map_config, mock_output)    

    def build_task_info(self, question: str) -> dict:
        """Build task-specific information for code execution"""
        try:
            # Extract question from data format
            # Get hyperparameters from agent configuration
            agent_config = self.trainer.mas_r1_config.get('agent', {})
            
            # Global variables are now set up once in the trainer before this method is called
            # No need to set them up again here (optimization)

            task_info = Info('task', 'User', question, None, None, None, -1, None)

            # Log the final task_info
            # main_rank_print(f"\n{'='*80}")
            # main_rank_print(f"FINAL TASK_INFO")
            # main_rank_print(f"{'='*80}")
            # main_rank_print(f"task_info: {task_info}")
            # main_rank_print(f"task_info type: {type(task_info)}")
            # main_rank_print(f"task_info length: {len(task_info)}")
            # main_rank_print(f"question in task_info: {task_info[2]}")
            # main_rank_print(f"question length: {len(task_info[2]) if task_info[2] else 0}")
            # main_rank_print(f"{'='*80}\n")

            return task_info
            
        except Exception as e:
            main_rank_print(f"ERROR building task info: {e}")
            raise RuntimeError(f"Failed to build task info: {e}")

    def prepare_batch(self, gen_batch: DataProto, templates: Dict[str, str]) -> DataProto:
        """Prepare batch for code generation for data"""
        raise NotImplementedError("prepare_batch is not implemented")

    def setup_global_variables(self, agent_config: dict, mas_r1_config: dict = None, config: dict = None):
        """Set up global variables needed by AgentSystem before initialization"""
        main_rank_print(f"\n{'='*60}")
        main_rank_print(" DATASET PROCESSOR: SETTING UP GLOBAL VARIABLES")
        main_rank_print(f"{'='*60}")
        
        # Use the shared internal method
        result = self._setup_global_variables_internal(agent_config, mas_r1_config, config)
        
        
        main_rank_print(f"✓  Global variables set:")
        main_rank_print(f"  - global_max_sc: {result['max_sc']}")
        main_rank_print(f"  - global_max_round: {result['max_round']}")
        main_rank_print(f"  - global_node_model: {result['node_model']}")
        main_rank_print(f"  - global_model_sampler_map: {len(result['model_sampler_map'])} samplers")
        main_rank_print(f"  - global_decompose_only: {result['decompose_only']}")
        main_rank_print(f"  - global_architecture_only: {result['architecture_only']}")
        main_rank_print(f"  - global_architecture_only_sequential: {result['architecture_only_sequential']}")
        main_rank_print(f"  - global_enable_tree_architecture: {result['enable_tree_architecture']}")
        main_rank_print(f"  - global_init_archive: {result['init_archive']}")
        main_rank_print(f"  - global_problem_type: {result['problem_type']}")
        main_rank_print(f"  - global_dataset_name: {result['dataset_name']}")
        main_rank_print(f"  - global_no_decompose: {result['no_decompose']}")
        main_rank_print(f"  - global_use_llm_judge: {result['use_llm_judge']}")
        main_rank_print(f"  - global_web_search_type: {result['web_search_type']}")
        main_rank_print(f"  - global_retrieval_method: {result['retrieval_method']}")
        main_rank_print(f"  - global_max_concurrent: {result['max_concurrent']}")
        main_rank_print(f"  - global_use_igsm_prompt: {result['use_igsm_prompt']}")
        main_rank_print(f"  - global_use_long_horizon: {result['use_long_horizon']}")
        main_rank_print(f"  - global_reasoning_effort: {result['reasoning_effort']}")
        main_rank_print(f"  - global_max_tokens: {result['max_tokens']}")
        main_rank_print(f"{'='*60}\n")

        return result

 
    def _setup_global_variables_internal(self, agent_config: dict, mas_r1_config: dict = None, config: dict = None):
        """Internal method to set up global variables - shared between build_task_info and setup_global_variables"""
        
        # Extract values from agent config
        max_round = agent_config.get('max_round', 5)
        max_sc = agent_config.get('max_sc', 5)
        multiply_processes = mas_r1_config.get('multiply_processes')
        node_model = agent_config.get('model_name', 'gpt-4o')
        decompose_only = agent_config.get('decompose_only', False)
        architecture_only = agent_config.get('architecture_only', False)
        architecture_only_sequential = agent_config.get('architecture_only_sequential', False)
        # Get enable_tree_architecture from mas_r1_config if available, otherwise default to False
        enable_tree_architecture = mas_r1_config.get('enable_tree_architecture', False)
        # Get mock_output from agent config, default to False
        mock_output = mas_r1_config.get('mock_output', False)
        # Get include_blocks from agent config, default to False
        include_blocks = agent_config.get('include_blocks', False)
        # Get add_judge from agent config, default to False
        add_judge = mas_r1_config.get('add_judge', False)
        # Get eval_building_blocks from mas_r1_config, default to False
        eval_building_blocks = mas_r1_config.get('eval_building_blocks', False)
        # Get known_prompt from mas_r1_config, default to None
        known_prompt = mas_r1_config.get('known_prompt', None)
        # Get max_ray_workers from mas_r1_config, default to 48
        max_ray_workers = mas_r1_config.get('max_ray_workers', 48)
        # Get problem_type from config, default to None
        problem_type = config.azr.get('problem_type')
        # Get dataset_name from config, default to None
        dataset_name = config.azr.get('dataset_name')
        # Get no_decompose from mas_r1_config, default to False
        no_decompose = mas_r1_config.get('no_decompose', False)

        use_llm_judge = mas_r1_config.get('use_llm_judge', False)
        
        # Get web_search_type from agent_config, default to "online"
        # Can be "online" (live internet) or "offline" (BrowseComp-Plus corpus)
        web_search_type = agent_config.get('web_search_type', 'online')
        
        # Get retrieval_method from agent_config, default to "bm25"
        # Can be "bm25" or "dense"
        retrieval_method = agent_config.get('retrieval_method', 'bm25')
        
        # Get max_concurrent from agent_config, default to 32
        max_concurrent = agent_config.get('max_concurrent', 32)

        # Get use_long_horizon from mas_r1_config, default to False
        use_long_horizon = mas_r1_config.get('use_long_horizon', False)

        # Get reasoning_effort from agent_config, default to "low"
        reasoning_effort = agent_config.get('reasoning_effort', 'low')

        # Get max_tokens from agent_config, default to None
        max_tokens = agent_config.get('max_tokens', None)

        # Check if data files contain "igsm" (case insensitive)
        use_igsm_prompt = False
        igsm_variant = None  # Will be 'breadth', 'depth', 'horizon', 'parallel', or 'combine'
        try:
            if config is not None and hasattr(config, 'data'):
                data_config = config.data
                # Check train_files for IGSM variant detection
                train_files = getattr(data_config, 'train_files', None)
                
                # Detect variant from train_files only
                if train_files is not None:
                    train_files_str = str(train_files).lower()
                    if 'igsm' in train_files_str:
                        use_igsm_prompt = True
                        main_rank_print(f"✓  Detected IGSM dataset in train_files: {train_files}")
                        # Detect variant from train_files
                        if 'combine' in train_files_str:
                            igsm_variant = 'combine'
                            main_rank_print(f"✓  Detected IGSM variant: combine")
                        elif 'parallel' in train_files_str:
                            igsm_variant = 'parallel'
                            main_rank_print(f"✓  Detected IGSM variant: parallel")
                        elif 'breadth' in train_files_str:
                            igsm_variant = 'breadth'
                            main_rank_print(f"✓  Detected IGSM variant: breadth")
                        elif 'depth' in train_files_str:
                            igsm_variant = 'depth'
                            main_rank_print(f"✓  Detected IGSM variant: depth")
                        elif 'horizon' in train_files_str:
                            igsm_variant = 'horizon'
                            main_rank_print(f"✓  Detected IGSM variant: horizon")
        except Exception as e:
            main_rank_print(f"⚠️  Error checking for IGSM dataset: {e}")
            # If there's any error, fall back to default behavior (False)
            pass

        # Convert string values to boolean if needed
        if isinstance(decompose_only, str):
            decompose_only = decompose_only.lower() in ['true', '1', 'yes', 'on']
        if isinstance(architecture_only, str):
            architecture_only = architecture_only.lower() in ['true', '1', 'yes', 'on']
        if isinstance(architecture_only_sequential, str):
            architecture_only_sequential = architecture_only_sequential.lower() in ['true', '1', 'yes', 'on']
        if isinstance(enable_tree_architecture, str):
            enable_tree_architecture = enable_tree_architecture.lower() in ['true', '1', 'yes', 'on']
        if isinstance(mock_output, str):
            mock_output = mock_output.lower() in ['true', '1', 'yes', 'on']
        if isinstance(include_blocks, str):
            include_blocks = include_blocks.lower() in ['true', '1', 'yes', 'on']
        if isinstance(add_judge, str):
            add_judge = add_judge.lower() in ['true', '1', 'yes', 'on']
        if isinstance(eval_building_blocks, str):
            eval_building_blocks = eval_building_blocks.lower() in ['true', '1', 'yes', 'on']
        if isinstance(no_decompose, str):
            no_decompose = no_decompose.lower() in ['true', '1', 'yes', 'on']
        if isinstance(use_llm_judge, str):
            use_llm_judge = use_llm_judge.lower() in ['true', '1', 'yes', 'on']
        if isinstance(use_long_horizon, str):
            use_long_horizon = use_long_horizon.lower() in ['true', '1', 'yes', 'on']
        
        # Validate reasoning_effort is valid (can be 'low', 'medium', or 'high')
        valid_reasoning_efforts = ['low', 'medium', 'high']
        if reasoning_effort not in valid_reasoning_efforts:
            main_rank_print(f"⚠️  Invalid reasoning_effort '{reasoning_effort}', defaulting to 'low'")
            reasoning_effort = 'low'
        
        # Validate web_search_type is valid
        if web_search_type not in ['online', 'offline']:
            main_rank_print(f"⚠️  Invalid web_search_type '{web_search_type}', defaulting to 'online'")
            web_search_type = 'online'
            
        # Validate retrieval_method is valid
        if retrieval_method not in ['bm25', 'dense']:
            main_rank_print(f"⚠️  Invalid retrieval_method '{retrieval_method}', defaulting to 'bm25'")
            retrieval_method = 'bm25'
            
        init_archive = agent_config.get('init_archive', ['COT', 'COT_SC', 'Reflexion', 'LLM_debate'])
        
        # Validate that init_archive is a list and convert if needed
        if not isinstance(init_archive, list):
            main_rank_print(f"init_archive should be a list, got {type(init_archive)}: {init_archive}. Converting to list.")
            # Convert to list - handle different input types
            if hasattr(init_archive, '_content') or hasattr(init_archive, 'to_container'):
                # Handle OmegaConf objects (ListConfig, DictConfig, etc.)
                try:
                    if hasattr(init_archive, 'to_container'):
                        init_archive = init_archive.to_container()
                    elif hasattr(init_archive, '_content'):
                        init_archive = init_archive._content
                    else:
                        init_archive = list(init_archive)
                except Exception:
                    raise ValueError(f"Failed to convert OmegaConf object to list: {init_archive}")
            else:
                raise ValueError(f"init_archive should be a list, got {type(init_archive)}: {init_archive}. Please provide a list of strings.")

            main_rank_print(f"init_archive converted to {init_archive}")

        # Create model_sampler_map using shared function
        model_sampler_map = self._create_model_sampler_map(agent_config, mock_output)
        
        # Define Math-specific global variables
        FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following XML format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n"""

        if dataset_name == 'multiple_choice': #TODO: here we do not seprate training and testing (but we cannot do that when compute score)
            output_description = "If the question is asked for a choice, Return ONLY a single letter (A, B, C, or D) that corresponds to the correct answer and DO NOT return anything other than the single letter; If the question is asked for more than single letter, Return what the question asked and make sure the answer is complete."
        else:
            output_description = "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete."
        cot_instruction = "Please think step by step and then solve the task."
        debate_role = ['Math Professor', 'Grade School Teacher']
        
        # Set LoRA flag in global variables for agent system to use
        lora_rank = config.actor_rollout_ref.model.get('lora_rank', 0)
        lora_enabled = lora_rank > 0
        
        # Set all global variables
        set_global("global_model_sampler_map", model_sampler_map)
        set_global("global_FORMAT_INST", FORMAT_INST)
        set_global("global_output_description", output_description)
        set_global("global_cot_instruction", cot_instruction)
        set_global("global_debate_role", debate_role)
        set_global("global_max_round", int(max_round))
        set_global("global_max_sc", int(max_sc))
        set_global("global_node_model", node_model)
        set_global("global_decompose_only", decompose_only)
        set_global("global_architecture_only", architecture_only)
        set_global("global_architecture_only_sequential", architecture_only_sequential)
        set_global("global_enable_tree_architecture", enable_tree_architecture)
        set_global("global_init_archive", init_archive)
        set_global("global_include_blocks", include_blocks)
        set_global("global_add_judge", add_judge)
        set_global("global_eval_building_blocks", eval_building_blocks)
        set_global("global_multiply_processes", multiply_processes)
        set_global("global_known_prompt", known_prompt)
        set_global("global_max_ray_workers", int(max_ray_workers))
        set_global("global_problem_type", problem_type)
        set_global("global_dataset_name", dataset_name)
        set_global("global_no_decompose", no_decompose)
        set_global("global_use_llm_judge", use_llm_judge)
        set_global("global_web_search_type", web_search_type)
        set_global("global_retrieval_method", retrieval_method)
        set_global("global_max_concurrent", int(max_concurrent))
        set_global("global_use_igsm_prompt", use_igsm_prompt)
        set_global("global_igsm_variant", igsm_variant)
        set_global("global_use_long_horizon", use_long_horizon)
        set_global("global_reasoning_effort", reasoning_effort)
        set_global("global_max_tokens", int(max_tokens) if max_tokens is not None else None)
        
        return {
            'max_round': max_round,
            'max_sc': max_sc,
            'node_model': node_model,
            'model_sampler_map': model_sampler_map,
            'decompose_only': decompose_only,
            'architecture_only': architecture_only,
            'architecture_only_sequential': architecture_only_sequential,
            'enable_tree_architecture': enable_tree_architecture,
            'init_archive': init_archive,
            'include_blocks': include_blocks,
            'add_judge': add_judge,
            'eval_building_blocks': eval_building_blocks,
            'multiply_processes': multiply_processes,
            'known_prompt': known_prompt,
            'max_ray_workers': max_ray_workers,
            'problem_type': problem_type,
            'dataset_name': dataset_name,
            'no_decompose': no_decompose,
            'use_llm_judge': use_llm_judge,
            'web_search_type': web_search_type,
            'retrieval_method': retrieval_method,
            'max_concurrent': max_concurrent,
            'use_igsm_prompt': use_igsm_prompt,
            'use_long_horizon': use_long_horizon,
            'reasoning_effort': reasoning_effort,
            'max_tokens': max_tokens
        }