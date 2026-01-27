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
Mean@k Reward Utility for MAS-R1 Reward Manager.

This module implements the mean@k reward strategy as described in the TODO comment:
- Groups responses into groups of size k
- Computes mean reward for each group
- All responses in the same group share the same group reward
- Prepares data for VERL's existing GRPO advantage computation infrastructure

The key insight is to leverage VERL's existing advantage computation rather than duplicating it.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from mas_r1_reasoner.agents.common import main_rank_print


def compute_mean_at_k_rewards_simple(
    rewards: List[float], 
    k: int
) -> Tuple[List[float], List[int], Dict[str, Any]]:
    """
    Compute mean@k rewards by grouping responses into groups of size k.
    Simple sequential grouping as mentioned in the TODO comment.
    
    Args:
        rewards: List of individual response rewards
        k: Group size for mean@k computation
    
    Returns:
        Tuple of:
        - mean_rewards: List of mean rewards for each group
        - group_assignments: List mapping each response to its group index
        - group_info: Dictionary containing group statistics and metadata
    
    Raises:
        ValueError: If k <= 0
    """
    if k <= 0:
        raise ValueError(f"Group size k must be positive, got {k}")
    
    if not rewards:
        return [], [], {}
    
    n_responses = len(rewards)
    
    # Sequential grouping: [0,1,2,...,k-1], [k,k+1,...,2k-1], ...
    n_groups = (n_responses + k - 1) // k  # Ceiling division
    group_assignments = []
    
    for i in range(n_responses):
        group_idx = i // k
        group_assignments.append(group_idx)
    
    # Compute mean rewards for each group
    mean_rewards = []
    group_sizes = []
    group_members = []
    
    for group_idx in range(n_groups):
        start_idx = group_idx * k
        end_idx = min(start_idx + k, n_responses)
        group_rewards = rewards[start_idx:end_idx]
        
        mean_reward = np.mean(group_rewards) if group_rewards else 0.0
        mean_rewards.append(mean_reward)
        group_sizes.append(len(group_rewards))
        group_members.append(list(range(start_idx, end_idx)))
    
    # Prepare group information
    group_info = {
        'n_groups': len(mean_rewards),
        'group_sizes': group_sizes,
        'group_members': group_members,
        'k': k,
        'total_responses': n_responses,
        'group_details': [(mean_rewards[i], group_members[i]) for i in range(len(mean_rewards))]
    }
    
    return mean_rewards, group_assignments, group_info





def validate_mean_at_k_config_simple(
    k: int,
    total_responses: int
) -> bool:
    """
    Validate mean@k configuration parameters.
    
    Args:
        k: Group size for mean@k computation
        total_responses: Total number of responses
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if k <= 0:
        main_rank_print(f"❌ Invalid group size k: {k} (must be positive)")
        return False
    
    if total_responses <= 0:
        main_rank_print(f"❌ Invalid total responses: {total_responses} (must be positive)")
        return False
    
    if k > total_responses:
        main_rank_print(f"⚠️ Group size k ({k}) is larger than total responses ({total_responses})")
        main_rank_print(f"   This will result in a single group with all responses")
    
    return True


# Note: Advantage injection is handled by mas_r1_reasoner.trainer.utils.advantages
# This keeps reward computation and advantage handling separate


def apply_mean_at_k_rewards(
    reward_extra_info: Dict[str, Any],
    reward_tensor: torch.Tensor,
    data: Any,
    k: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply Mean@k reward strategy with proper advantage computation flow:
    
    1. Compute group rewards (mean@k of each group)
    2. Compute GRPO advantages on the group level  
    3. Assign group advantages to individual responses
    4. Prepare data for VERL to use our advantages instead of computing its own
    
    This matches the diagram flow: Group Rewards → Group Advantages → Response Advantages
    """
    main_rank_print(f"🎯 Applying Mean@k reward strategy with proper advantage flow...")
    
    # Extract hierarchical rewards
    hierarchical_rewards = reward_extra_info.get('hierarchical_final_reward', [])
    total_responses = len(hierarchical_rewards)
    
    if total_responses == 0:
        main_rank_print(f"❌ No hierarchical rewards found")
        return reward_tensor, reward_extra_info
    
    # Validate configuration
    if not validate_mean_at_k_config_simple(k, total_responses):
        main_rank_print(f"❌ Invalid mean@k configuration: k={k}, total_responses={total_responses}")
        return reward_tensor, reward_extra_info
    
    main_rank_print(f"   Total responses: {total_responses}")
    main_rank_print(f"   Group size k: {k}")
    main_rank_print(f"   Number of groups: {total_responses // k}")
    
    # STEP 1: Compute group rewards (mean@k of each group)
    main_rank_print(f"\n📊 STEP 1: Computing group rewards (mean@k)...")
    mean_rewards, group_assignments, group_info = compute_mean_at_k_rewards_simple(
        hierarchical_rewards, k
    )
    
    # Log group rewards
    for group_idx, (mean_reward, response_indices) in enumerate(group_info['group_details']):
        main_rank_print(f"   Group {group_idx}: mean_reward={mean_reward:.4f}, responses={response_indices}")
    
    # STEP 2: Mean@k rewards computed successfully
    main_rank_print(f"\n🎯 STEP 2: Mean@k rewards computed successfully!")
    
    # Log group rewards
    for group_idx, mean_reward in enumerate(mean_rewards):
        main_rank_print(f"   Group {group_idx}: mean_reward={mean_reward:.4f}")
    
    # Update reward_extra_info with mean@k rewards
    reward_extra_info['hierarchical_final_reward'] = [mean_rewards[group_assignments[i]] if i < len(group_assignments) else 0.0 for i in range(total_responses)]
    reward_extra_info['score'] = reward_extra_info['hierarchical_final_reward'].copy()
    
    # Store mean@k information (rewards only)
    reward_extra_info['mean_at_k_info'] = {
        'group_rewards': mean_rewards,
        'group_assignments': group_assignments,
        'group_details': group_info['group_details']
    }
    
    # Update reward tensor with mean rewards
    for response_idx in range(total_responses):
        if response_idx < len(data):
            data_item = data[response_idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            
            if valid_response_length > 0 and valid_response_length <= reward_tensor.size(1):
                if response_idx < len(group_assignments):
                    group_idx = group_assignments[response_idx]
                    if group_idx < len(mean_rewards):
                        reward_tensor[response_idx, valid_response_length - 1] = mean_rewards[group_idx]
    
    # STEP 4: Mean@k rewards applied to reward tensor
    main_rank_print(f"\n🔧 STEP 4: Mean@k rewards applied to reward tensor successfully!")
    
    # STEP 5: Mean@k rewards completed
    main_rank_print(f"\n✅ Mean@k rewards applied successfully!")
    main_rank_print(f"   Group rewards computed and applied to responses")
    main_rank_print(f"   All responses in same group now have same reward")
    main_rank_print(f"   Reward tensor updated with mean@k rewards")
    
    return reward_tensor, reward_extra_info



