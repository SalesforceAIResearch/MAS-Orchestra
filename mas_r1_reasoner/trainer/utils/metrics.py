# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions and helper methods for MAS-R1 Trainer.
This file contains helper methods that can be extracted from the main trainer
to keep the main trainer file focused on core PPO logic.
"""

import os
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from verl import DataProto
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.agents.agent_system import AgentSystem as SequentialAgentSystem
from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
import re
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code


def compute_pass_at_k_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], val_n: int, seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute pass@k metrics for powers of 2 up to the total number of responses per question.
    
    This function groups responses by original questions and computes pass@k metrics
    by taking the first k responses and checking if at least one is correct.
    
    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        val_n: Number of responses per question (validation n parameter).
        seed: Random seed (not used in this implementation). Defaults to 42.
        
    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }
        
        Where metric_name includes:
        - "pass@1/mean": Pass@1 (first response correctness)
        - "pass@2/mean": Pass@2 (at least one correct in first 2 responses)
        - "pass@4/mean": Pass@4 (at least one correct in first 4 responses)
        - etc. (powers of 2 up to val_n)
    """
    import numpy as np
    from collections import defaultdict
    
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate pass@k metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                
                # Compute pass@k for powers of 2 up to n_resps
                ns = []
                n = 1  # Start from 1 (unlike VERL which starts from 2)
                while n <= n_resps:
                    ns.append(n)
                    n *= 2
                
                # Remove duplicates and ensure we don't exceed n_resps
                ns = sorted(list(set(ns)))
                ns = [n for n in ns if n <= n_resps]

                for n in ns:
                    # Take the first n responses and check if at least one is correct (> 0)
                    first_n_responses = var_vals[:n]
                    pass_at_k = 1.0 if any(val > 0 for val in first_n_responses) else 0.0
                    metric[f"pass@{n}/mean"] = pass_at_k

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val 