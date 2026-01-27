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
Custom Advantage Estimator for Mean@k Rewards.

This module provides a custom advantage estimator that uses our pre-computed mean@k advantages
instead of computing new ones during training. This ensures that VERL uses our carefully
computed group-level advantages rather than overwriting them with individual response advantages.
"""

import torch
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

# Import VERL's GRPO advantage computation for fallback
try:
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    print("⚠️  VERL not available, using fallback advantage computation")


class MeanAtKAdvantageEstimator:
    """
    Custom advantage estimator that uses our pre-computed advantages
    instead of computing new ones.
    
    This is Solution 1 from our integration guide - the most robust approach
    that ensures VERL uses our advantages during training.
    """
    
    def __init__(self):
        """
        Initialize the Mean@k advantage estimator.
        
        Args:
        """
        self.name = "MeanAtKAdvantage"
        
    def compute_advantage(
        self, 
        scores: torch.Tensor, 
        data: Any, 
        norm_adv_by_std_in_grpo: bool = True
    ) -> torch.Tensor:
        """
        Compute mean@k advantages on-the-fly when called by VERL.
        
        This function is called by VERL's advantage computation, and it:
        1. Checks if mean@k is enabled
        2. Computes mean@k rewards and group assignments
        3. Computes group-level advantages using GRPO logic
        4. Assigns group advantages to individual responses
        5. Returns the computed advantages
        
        Args:
            scores: Tensor of scores/rewards [batch_size, sequence_length] or [batch_size]
            data: Data object containing hierarchical rewards and configuration
            norm_adv_by_std_in_grpo: Whether to normalize advantages by standard deviation
            
        Returns:
            Tensor of advantages with same shape as scores
        """
        # Check if mean@k is enabled and we have hierarchical rewards
        if not hasattr(data, 'non_tensor_batch'):
            raise ValueError("No non_tensor_batch found in data for mean@k advantage computation")
        
        non_tensor_batch = data.non_tensor_batch
        
        # Check if we have hierarchical rewards to work with
        if 'hierarchical_final_reward' not in non_tensor_batch:
            raise ValueError("No hierarchical rewards found in data for mean@k advantage computation")
        
        # Check if mean@k is enabled (this should be set by the reward manager)
        if not non_tensor_batch.get('mean_at_k_enabled', False):
            raise ValueError("Mean@k not enabled in data configuration")
        
        # Get mean@k rewards that were already computed
        mean_at_k_rewards = non_tensor_batch.get('hierarchical_final_reward', [])
        group_assignments = non_tensor_batch.get('mean_at_k_group_assignments', [])
        
        if not group_assignments:
            raise ValueError("No mean@k group assignments found. Mean@k rewards must be computed first.")
        
        print(f"🎯 Computing mean@k advantages using pre-computed rewards...")
        print(f"   Total responses: {len(mean_at_k_rewards)}")
        print(f"   Group assignments: {len(group_assignments)}")
        
        try:
            # Step 1: Extract group rewards from the assignments
            # Each response has a group assignment, and we need to get the mean reward for each group
            unique_groups = sorted(set(group_assignments))
            group_rewards = []
            
            for group_idx in unique_groups:
                # Find all responses in this group and get their mean reward
                group_responses = [i for i, g in enumerate(group_assignments) if g == group_idx]
                group_mean_reward = sum(mean_at_k_rewards[i] for i in group_responses) / len(group_responses)
                group_rewards.append(group_mean_reward)
            
            print(f"   Computed {len(group_rewards)} group rewards from assignments")
            
            # Step 2: Compute group-level advantages
            response_advantages, group_advantages = compute_mean_at_k_advantages(
                group_rewards, group_assignments, len(mean_at_k_rewards)
            )
            
            # Step 3: Convert to tensor and ensure correct shape
            advantages_tensor = torch.tensor(response_advantages, dtype=scores.dtype, device=scores.device)
            
            # If scores is 2D [batch_size, sequence_length], expand advantages
            if len(scores.shape) == 2 and len(advantages_tensor.shape) == 1:
                expanded_advantages = advantages_tensor.unsqueeze(1).expand(-1, scores.shape[1])
                print(f"✅ Mean@k advantages computed and expanded to match scores shape: {expanded_advantages.shape}")
                return expanded_advantages
            else:
                print(f"✅ Mean@k advantages computed: {advantages_tensor.shape}")
                return advantages_tensor
                
        except Exception as e:
            print(f"❌ Error computing mean@k advantages: {e}")
            raise RuntimeError(f"Failed to compute mean@k advantages: {e}")
    

    
    def __str__(self) -> str:
        return f"MeanAtKAdvantageEstimator()"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_mean_at_k_advantage_estimator() -> MeanAtKAdvantageEstimator:
    """
    Factory function to create a Mean@k advantage estimator.
    
        
    Returns:
        Configured MeanAtKAdvantageEstimator instance
    """
    return MeanAtKAdvantageEstimator()


def compute_mean_at_k_advantages(
    mean_rewards: List[float], 
    group_assignments: List[int], 
    total_responses: int
) -> Tuple[List[float], torch.Tensor]:
    """
    Complete mean@k advantage computation pipeline.
    
    This function handles the entire advantage computation flow:
    1. Convert rewards to tensor
    2. Compute group-level advantages using GRPO logic
    3. Assign group advantages to individual responses
    
    Args:
        mean_rewards: List of mean rewards for each group
        group_assignments: List mapping each response to its group index
        total_responses: Total number of responses
        
    Returns:
        Tuple of:
        - response_advantages: List of advantages for each response
        - group_advantages: Tensor of group-level advantages
    """
    # Convert group rewards to tensor for advantage computation
    group_rewards_tensor = torch.tensor(mean_rewards, dtype=torch.float32)
    
    # Compute group-level advantages using GRPO logic
    # This simulates what VERL would do: (score - group_mean) / (group_std + epsilon)
    group_advantages = compute_group_level_advantages(group_rewards_tensor)
    
    # Assign group advantages to individual responses
    response_advantages = assign_group_advantages_to_responses(
        group_advantages, group_assignments, total_responses
    )
    
    return response_advantages, group_advantages


def compute_group_level_advantages(group_rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute GRPO advantages on the group level.
    
    This simulates VERL's GRPO advantage computation:
    advantage = (score - group_mean) / (group_std + epsilon)
    
    Args:
        group_rewards: Tensor of group rewards [num_groups]
        
    Returns:
        Tensor of group advantages [num_groups]
    """
    if len(group_rewards) == 0:
        return group_rewards
    
    # Compute group statistics
    group_mean = torch.mean(group_rewards)
    group_std = torch.std(group_rewards)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Compute advantages: (score - group_mean) / (group_std + epsilon)
    advantages = (group_rewards - group_mean) / (group_std + epsilon)
    
    return advantages


def assign_group_advantages_to_responses(
    group_advantages: torch.Tensor, 
    group_assignments: List[int], 
    total_responses: int
) -> List[float]:
    """
    Assign group advantages to individual responses.
    
    Args:
        group_advantages: Tensor of group advantages [num_groups]
        group_assignments: List mapping each response to its group index
        total_responses: Total number of responses
        
    Returns:
        List of response-level advantages [total_responses]
    """
    response_advantages = []
    for response_idx in range(total_responses):
        if response_idx < len(group_assignments):
            group_idx = group_assignments[response_idx]
            if group_idx < len(group_advantages):
                response_advantages.append(group_advantages[group_idx].item())
            else:
                response_advantages.append(0.0)  # fallback
        else:
            response_advantages.append(0.0)  # fallback
    
    return response_advantages










