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
MAS-R1 Reward Manager that inherits from NaiveRewardManager.
This provides a clean interface for MAS-R1 reward computation with minimal modifications.
"""
import re
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager
from mas_r1_reasoner.agents.common import main_rank_print
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from mas_r1_reasoner.rewards.utils.execution import (
    execute_codes_and_store_results,
)
from mas_r1_reasoner.rewards.utils.extraction import (
    extract_questions_and_ground_truth,
    generate_and_extract_codes,
    generate_and_extract_codes_with_tree_validation
)
from mas_r1_reasoner.agents.shared_vars import get_global
from mas_r1_reasoner.rewards.utils.reformat import (
    from_question_results_to_execution_results,
)

        
def sub_task_sub_agent_to_hierarchical_reward_separate(sub_task_reward: float, sub_agent_reward: float, response_idx: int, level: int) -> float:
    """
    Compute hierarchical reward based on sub-task and sub-agent rewards.
        
    Final reward rules:
    - if level 1, use sub-task reward
    - if level 2, use sub-agent reward
    - sub-task in one group, sub-agent corresponding to the same parent sub-task in the same group
    Args:
        sub_task_reward: Reward for the sub-task (0.0 or 1.0)
        sub_agent_reward: Reward for the sub-agent (0.0 or 1.0)
        response_idx: Index of the response for logging
        level: Level of the response (1 for Level 1, 2 for Level 2)
        
    Returns:
        Hierarchical reward: 1.0 or -1.0
        
    Raises:
        ValueError: If level is not 1 or 2
    """
    if level == 1:
        # Level 1 responses: use sub_task_reward directly
        return sub_task_reward
    elif level == 2:
        # Level 2 responses: use sub_agent_reward directly
        return sub_agent_reward
    else:
        # Invalid level - raise error
        error_msg = f"Invalid level {level} for response {response_idx}. Level must be 1 or 2."
        main_rank_print(f"❌ {error_msg}")
        raise ValueError(error_msg)


def sub_task_sub_agent_to_hierarchical_reward_unified(sub_task_reward: float, sub_agent_reward: float, response_idx: int, level: int) -> float:
    """
    Compute hierarchical reward based on sub-task and sub-agent rewards.

    Final reward rules:
    - sub-task=1, sub-agent=1: reward = 2
    - sub-task=1, sub-agent=0: reward = 1  
    - sub-task=0, sub-agent=0: reward = -1

    Args:
        sub_task_reward: Reward for the sub-task (0.0 or 1.0)
        sub_agent_reward: Reward for the sub-agent (0.0 or 1.0)
        
    Returns:
        Hierarchical reward: 2.0, 1.0, or -1.0
        
    Raises:
        ValueError: If reward combination is invalid
    """
    if sub_task_reward > 0.0 and sub_agent_reward > 0.0:
        return 2.0
    elif sub_task_reward > 0.0 and sub_agent_reward <= 0.0:
        return 1.0
    elif sub_task_reward <= 0.0 and sub_agent_reward <= 0.0:
        return -1.0
    else:
        # this should never happen
        # main_rank_print(f"ERROR: Invalid reward components: sub_task_reward={sub_task_reward}, sub_agent_reward={sub_agent_reward}, response_idx={response_idx}")
        # import time
        # time.sleep(100)
        raise ValueError(f"Invalid reward components: sub_task_reward={sub_task_reward}, sub_agent_reward={sub_agent_reward}, response_idx={response_idx}")

