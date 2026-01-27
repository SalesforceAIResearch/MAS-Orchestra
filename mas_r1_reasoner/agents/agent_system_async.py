import string
import copy
import os
import random
import numpy as np
import re
import time
import asyncio
from typing import Any, Optional, Dict, List
import torch
import ray
from verl import DataProto
from mas_r1_reasoner.agents.shared_vars import get_global, set_global
from collections import namedtuple
from tqdm import tqdm

# Import the base classes from the original agent_system.py
from mas_r1_reasoner.agents.agent_system import LLMAgentBase, AgentSystem, Info
from mas_r1_reasoner.agents.shared_vars import get_global, set_global


def initialize_ray_if_needed():
    """Initialize Ray if not already initialized"""
    if not ray.is_initialized():
        print("🔧 Initializing Ray...")
        ray.init(
            num_cpus=os.cpu_count(),
            ignore_reinit_error=True,
            log_to_driver=False  # Reduce log noise
        )
        print("✅ Ray initialized successfully")
    else:
        print("✓ Ray already initialized (using existing Ray instance)")


def shutdown_ray_if_needed():
    """Shutdown Ray if it's initialized (use with caution - only if you initialized it)"""
    if ray.is_initialized():
        print("⚠️  Shutting down Ray (use with caution if Ray was initialized elsewhere)...")
        ray.shutdown()
        print("✅ Ray shutdown complete")
    else:
        print("✓ Ray not initialized, nothing to shutdown")


@ray.remote(max_concurrency=128)
class RayAgentWorker:
    """Ray remote worker for executing agent tasks asynchronously"""
    
    def __init__(self):
        self.agent_system = None
        self.global_vars_initialized = False
        print(f"🔧 Ray Worker {ray.get_runtime_context().get_node_id()}: Initializing...")
    
    def initialize_with_globals(self, global_vars: Dict[str, Any]):
        """Initialize global variables in this worker process"""
        print(f"🔧 Ray Worker: Setting up global variables...")
        
        # Set simple global variables
        for var_name, var_value in global_vars.items():
            if not var_name.endswith('_config') and not var_name.endswith('_template'):
                set_global(var_name, var_value)
        
        # Recreate complex objects from configurations
        self._recreate_complex_globals(global_vars)
        
        self.global_vars_initialized = True
        print(f"✅ Ray Worker: Global variables initialized!")
        return True
    
    def _recreate_complex_globals(self, global_vars: Dict[str, Any]):
        """Recreate complex global variables from serializable configurations"""
        # 1. Recreate model_sampler_map from configuration
        if "global_model_sampler_map_config" in global_vars and global_vars["global_model_sampler_map_config"]:
            model_sampler_map = self._recreate_model_sampler_map(global_vars["global_model_sampler_map_config"])
            set_global("global_model_sampler_map", model_sampler_map)
            print(f"✓ Recreated global_model_sampler_map with {len(model_sampler_map)} samplers")
            print(f"  Available models: {list(model_sampler_map.keys())}")
        
        # 2. Recreate FORMAT_INST from template
        if "global_FORMAT_INST_template" in global_vars:
            format_template = global_vars["global_FORMAT_INST_template"]
            format_inst = lambda request_keys: format_template.format(request_keys=str(request_keys))
            set_global("global_FORMAT_INST", format_inst)
            print("✓ Recreated global_FORMAT_INST lambda function")
    
    def _recreate_model_sampler_map(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Recreate model_sampler_map from serializable configurations"""
        from mas_r1_reasoner.data_precessor.BaseDatasetProcessor import BaseDatasetProcessor
        
        # Extract mock_output setting from the first sampler config (assuming all samplers have same setting)
        #TODO: one may also make mock_output a global variable
        mock_output = False
        if configs:
            first_config = next(iter(configs.values()))
            mock_output = first_config.get('mock_output', False)
        
        # Use the existing create_model_sampler_map method with preserved mock_output setting
        model_sampler_map = BaseDatasetProcessor.create_model_sampler_map(configs, mock_output=mock_output)
        
        print(f"✓ Recreated {len(model_sampler_map)} samplers with mock_output={mock_output}")
        
        return model_sampler_map
    
    def initialize_agent_system(self):
        """Initialize AgentSystem once per worker"""
        if self.agent_system is None:
            if not self.global_vars_initialized:
                raise RuntimeError("Global variables must be initialized before creating AgentSystem")
            
            print(f"🔧 Ray Worker: Initializing AgentSystem...")
            
            # Verify that global_model_sampler_map is available
            try:
                model_sampler_map = get_global("global_model_sampler_map")
                if model_sampler_map:
                    print(f"✓ global_model_sampler_map available with {len(model_sampler_map)} samplers")
                    print(f"  Models: {list(model_sampler_map.keys())}")
                else:
                    print("⚠️  Warning: global_model_sampler_map is None")
            except Exception as e:
                print(f"⚠️  Warning: Could not access global_model_sampler_map: {e}")
            
            self.agent_system = AgentSystem()
            print(f"✅ Ray Worker: AgentSystem initialized!")
        return True
    
    async def execute_single_task_async(self, code: str, task_info: Dict[str, Any], timeout: int = None) -> tuple[str, bool, str]:
        """Execute a single task asynchronously with timeout handling"""
        try:
            # Ensure global variables are initialized
            if not self.global_vars_initialized:
                raise RuntimeError("Global variables not initialized in Ray worker")
            
            # Initialize agent system if needed
            if self.agent_system is None:
                self.initialize_agent_system()
            
            # Set the forward function for THIS EXECUTION ONLY
            self.agent_system.set_instance_forward_function(code)
            
            # Execute the async forward function
            if timeout:
                result = await asyncio.wait_for(
                    self.agent_system.forward(task_info),
                    timeout=timeout
                )
            else:
                result = await self.agent_system.forward(task_info)
            
            # Convert result to string if needed
            if isinstance(result, dict):
                result_str = str(result)
            elif hasattr(result, 'final_answer'): # TODO: here we care about the final answer, not the content
                result_str = str(result.final_answer)
                if hasattr(result, 'name') and result.name == 'error':
                    return result_str, False, f"Agent execution failed: {result_str}"
            else:
                result_str = str(result)
            
            return result_str, True, ""
            
        except asyncio.TimeoutError:
            return "", False, f"Task timed out after {timeout} seconds"
        except Exception as e:
            return "", False, str(e)
    
    def execute_single_task_sync(self, code: str, task_info: Dict[str, Any], timeout: int = None) -> tuple[str, bool, str]:
        """Synchronous wrapper for execute_single_task_async"""
        return asyncio.run(self.execute_single_task_async(code, task_info, timeout))



class AsyncAgentSystem(AgentSystem):
    """
    Async AgentSystem for executing generated MAS code with batch parallelization using Ray.
    Uses Ray remote actors for distributed async execution.
    """
    
    def __init__(self, agent_config: Dict[str, Any] = None, auto_init_ray: bool = False) -> None:
        super().__init__(agent_config)
        self.workers = []
        self.worker_pool_size = 0
                
        # Collect global variables from the main process
        self.global_vars = self._collect_global_variables()
        
        # Only initialize Ray if explicitly requested (since it's usually initialized in main_mas_r1.py)
        if auto_init_ray:
            initialize_ray_if_needed()
        elif not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Please initialize Ray first or set auto_init_ray=True")
        
        print(f"✓ AsyncAgentSystem initialized for Ray-based async batch parallelization")
        print(f"✓ Collected {len(self.global_vars)} global variables for Ray workers")
        
        # Verify that model_sampler_map was properly collected
        if "global_model_sampler_map_config" in self.global_vars and self.global_vars["global_model_sampler_map_config"]:
            config = self.global_vars["global_model_sampler_map_config"]
            print(f"✓ Model sampler config collected with {len(config)} models: {list(config.keys())}")
        else:
            print("⚠️  Warning: No model_sampler_map configuration collected")
    
    def _collect_global_variables(self) -> Dict[str, Any]:
        """Collect all global variables that need to be passed to Ray workers"""
        global_vars = {}
        
        # List of simple global variables that can be serialized
        simple_globals = [
            "global_output_description",
            "global_cot_instruction", 
            "global_debate_role",
            "global_max_round",
            "global_max_sc",
            "global_node_model",
            "global_decompose_only",
            "global_architecture_only",
            "global_architecture_only_sequential",
            "global_enable_tree_architecture",
            "global_init_archive",
            "global_include_blocks",
            "global_add_judge",
            "global_multiply_processes",
            "global_known_prompt",
            "global_max_ray_workers",
            "global_problem_type",
            "global_web_search_type",
            "global_retrieval_method",
            "global_max_concurrent",
            "global_use_igsm_prompt",
            "global_reasoning_effort",
            "global_max_tokens"
        ]
        
        # Collect simple global variables
        for var_name in simple_globals:
            try:
                value = get_global(var_name)
                global_vars[var_name] = value
            except NameError:
                print(f"⚠️  Warning: Global variable {var_name} not found, using None")
                global_vars[var_name] = None
        
        # Handle complex variables that need special treatment
        # 1. model_sampler_map - collect configuration instead of objects
        try:
            model_sampler_map = get_global("global_model_sampler_map")
            if model_sampler_map:
                # Store the configuration for recreating samplers in workers
                global_vars["global_model_sampler_map_config"] = self._extract_sampler_configs(model_sampler_map)
            else:
                global_vars["global_model_sampler_map_config"] = None
        except NameError:
            print("⚠️  Warning: global_model_sampler_map not found")
            global_vars["global_model_sampler_map_config"] = None
        
        # 2. FORMAT_INST - store as string template instead of lambda
        try:
            format_inst = get_global("global_FORMAT_INST")
            if format_inst and callable(format_inst):
                # Convert lambda to string template
                global_vars["global_FORMAT_INST_template"] = "Reply EXACTLY with the following XML format.\\n{request_keys}\\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\\n\\n"
            else:
                global_vars["global_FORMAT_INST_template"] = "Reply EXACTLY with the following XML format.\\n{request_keys}\\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\\n\\n"
        except NameError:
            print("⚠️  Warning: global_FORMAT_INST not found")
            global_vars["global_FORMAT_INST_template"] = "Reply EXACTLY with the following XML format.\\n{request_keys}\\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\\n\\n"
        
        return global_vars
    
    def _extract_sampler_configs(self, model_sampler_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract serializable configuration from model_sampler_map"""
        configs = {}
        
        for model_name, sampler in model_sampler_map.items():
            if hasattr(sampler, '__class__'):
                sampler_type = sampler.__class__.__name__
                config = {
                    'type': sampler_type,
                    'model': getattr(sampler, 'model', model_name),
                    'temperature': getattr(sampler, 'temperature', 0.5),
                    'system_message': getattr(sampler, 'system_message', None),
                    'mock_output': getattr(sampler, 'mock_output', False)  # Preserve mock_output setting
                }
                configs[model_name] = config
            else:
                # Fallback for unknown sampler types
                configs[model_name] = {
                    'type': 'ChatCompletionSampler',
                    'model': model_name,
                    'temperature': 0.5,
                    'system_message': None,
                    'mock_output': False
                }
        
        return configs
    
    def _initialize_worker_pool(self, num_workers: int):
        """Initialize Ray worker pool if not already done or if size changed"""
        if len(self.workers) != num_workers:
            # Clean up existing workers
            if self.workers:
                for worker in self.workers:
                    ray.kill(worker)
            
            # Create new worker pool and initialize with global variables
            self.workers = [RayAgentWorker.remote() for _ in range(num_workers)]
            
            # Initialize all workers with global variables
            init_futures = []
            for worker in self.workers:
                future = worker.initialize_with_globals.remote(self.global_vars)
                init_futures.append(future)
            
            # Wait for all workers to be initialized
            ray.get(init_futures)
            
            self.worker_pool_size = num_workers
            print(f"✓ Initialized {num_workers} Ray workers with global variables")
    
    async def execute_mas_batch_async(self, codes: List[str], task_infos: List[Dict[str, Any]], timeout: int = None) -> List[tuple[str, bool, str]]:
        """
        Execute multiple codes in parallel using Ray async workers.
        
        Args:
            codes: List of Python code strings containing the forward functions
            task_infos: List of dictionaries containing task information
            timeout: Execution timeout in seconds per execution
            
        Returns:
            List of tuples (result, success, error_message)
        """
        print(f"\n{'='*50}")
        print(f"EXECUTING {len(codes)} CODES IN PARALLEL WITH RAY")
        print(f"TIMEOUT: {timeout} seconds for execution")
        print(f"{'='*50}")
        
        # Ensure we have the same number of codes and task_infos
        if len(codes) != len(task_infos):
            raise ValueError(f"Number of codes ({len(codes)}) must match number of task_infos ({len(task_infos)})")
        
        # Get max_ray_workers from global variables, default to 48 if not set
        max_ray_workers = get_global('global_max_ray_workers')
        num_workers = min(max_ray_workers, len(codes))
        print(f"num_workers: {num_workers})")
        
        # Initialize worker pool
        self._initialize_worker_pool(num_workers)
        
        # Prepare tasks
        tasks = list(zip(codes, task_infos))
        results = [None] * len(tasks)
        
        # Submit all tasks to workers in round-robin fashion
        remote_futures = []
        for i, (code, task_info) in enumerate(tasks):
            worker = self.workers[i % len(self.workers)]
            future = worker.execute_single_task_async.remote(code, task_info, timeout)
            remote_futures.append((i, future))
        
        # Process results as they complete using ray.wait
        completed_tasks = 0
        with tqdm(total=len(tasks), desc="Executing tasks") as pbar:
            while completed_tasks < len(tasks):
                # Wait for at least one task to complete
                ready_futures, remaining_futures = ray.wait(
                    [future for _, future in remote_futures], 
                    num_returns=1, 
                    timeout=1.0
                )
                
                if ready_futures:
                    # Process completed tasks
                    for ready_future in ready_futures:
                        # Find the task index for this future
                        task_index = None
                        for i, (idx, future) in enumerate(remote_futures):
                            if future == ready_future:
                                task_index = idx
                                # Remove from remaining futures
                                remote_futures.pop(i)
                                break
                        
                        if task_index is not None:
                            try:
                                result = ray.get(ready_future)
                                results[task_index] = result
                                if result[1]:  # Success
                                    print(f"[DEBUG] Task {task_index+1} completed successfully")
                                else:
                                    print(f"[DEBUG] Task {task_index+1} failed: {result[2]}")
                            except Exception as e:
                                print(f"[DEBUG] Task {task_index+1} execution failed: {e}")
                                results[task_index] = ("", False, f"Task execution failed: {e}")
                            
                            completed_tasks += 1
                            pbar.update(1)
        
        # Ensure all results are filled (shouldn't be None)
        for i, result in enumerate(results):
            if result is None:
                results[i] = ("", False, f"Task {i+1} failed to complete")
        
        print(f"All {len(tasks)} executions completed!")
        print(f"{'='*50}\n")
        
        return results
    
    def execute_mas_batch_sync(self, codes: List[str], task_infos: List[Dict[str, Any]], timeout: int = None) -> List[tuple[str, bool, str]]:
        """
        Synchronous wrapper for execute_mas_batch_async.
        This method is kept for backward compatibility.
        
        Args:
            codes: List of Python code strings containing the forward functions
            task_infos: List of dictionaries containing task information
            timeout: Execution timeout in seconds per execution
            
        Returns:
            List of tuples (result, success, error_message)
        """
        return asyncio.run(self.execute_mas_batch_async(codes, task_infos, timeout))
    
    def cleanup(self):
        """Clean up Ray workers"""
        if self.workers:
            for worker in self.workers:
                ray.kill(worker)
            self.workers = []
            self.worker_pool_size = 0
            print("✓ Ray workers cleaned up")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    @classmethod
    def create_with_cleanup(cls, agent_config: Dict[str, Any] = None):
        """Create AsyncAgentSystem with automatic cleanup"""
        return cls(agent_config)
    
    @classmethod
    def create_with_globals(cls, agent_config: Dict[str, Any] = None, auto_init_ray: bool = False):
        """
        Create AsyncAgentSystem with proper global variable setup.
        This is the recommended way to create AsyncAgentSystem when global variables
        have been set up in the main process.
        """
        return cls(agent_config, auto_init_ray)


# Convenience functions for easy usage
async def execute_mas_batch_async(codes: List[str], task_infos: List[Dict[str, Any]], 
                                 timeout: int = None, agent_config: Dict[str, Any] = None, 
                                 auto_init_ray: bool = False) -> List[tuple[str, bool, str]]:
    """
    Convenience function to execute MAS batch asynchronously.
    
    Args:
        codes: List of Python code strings containing the forward functions
        task_infos: List of dictionaries containing task information
        timeout: Execution timeout in seconds per execution
        agent_config: Optional agent configuration
        auto_init_ray: Whether to auto-initialize Ray (default: False, assumes Ray is already initialized)
        
    Returns:
        List of tuples (result, success, error_message)
    """
    with AsyncAgentSystem.create_with_globals(agent_config, auto_init_ray=auto_init_ray) as agent_system:
        return await agent_system.execute_mas_batch_async(codes, task_infos, timeout)


def execute_mas_batch_sync(codes: List[str], task_infos: List[Dict[str, Any]], 
                          timeout: int = None, agent_config: Dict[str, Any] = None,
                          auto_init_ray: bool = False) -> List[tuple[str, bool, str]]:
    """
    Convenience function to execute MAS batch synchronously.
    
    Args:
        codes: List of Python code strings containing the forward functions
        task_infos: List of dictionaries containing task information
        timeout: Execution timeout in seconds per execution
        agent_config: Optional agent configuration
        auto_init_ray: Whether to auto-initialize Ray (default: False, assumes Ray is already initialized)
        
    Returns:
        List of tuples (result, success, error_message)
    """
    return asyncio.run(execute_mas_batch_async(codes, task_infos, timeout, agent_config, auto_init_ray))