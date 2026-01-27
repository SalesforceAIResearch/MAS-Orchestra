"""
Utility functions for merging rewards of identical sub-tasks in hierarchical reward computation.
"""

from typing import Dict, List, Tuple
from mas_r1_reasoner.agents.common import main_rank_print


def merge_identical_sub_tasks(
    level_1_responses: List[Tuple[int, dict]], 
    reward_extra_info: Dict[str, List], 
    mock_sub_task_sub_agent: bool = False
) -> int:
    """
    Merge logic for identical sub-tasks.
    
    If multiple Level 1 nodes have the same sub-task content, 
    use the maximum reward among them for all nodes in that group.
    
    Args:
        level_1_responses: List of (response_idx, result) tuples for Level 1 responses
        reward_extra_info: Dictionary containing reward information for all responses
        mock_sub_task_sub_agent: Whether to use mock mode for sub-tasks
        
    Returns:
        Number of unified groups processed
    """
    main_rank_print(f"\n🔗 Merge logic for identical sub-tasks...")
    
    # Group Level 1 responses by their sub-task content
    sub_task_groups = {}
    for response_idx, result in level_1_responses:
        # Get the sub-task content directly from the Level 1 response
        # Look for sub-task information in the result
        sub_task_content = None
        
        # Try to get sub-task from the result directly
        if isinstance(result, dict):
            # Extract sub-task from response text using the same method as extraction_tree.py
            response_text = result.get('response_text', '')
            if response_text:
                from mas_r1_reasoner.agents.common import extract_xml
                extracted_sub_tasks = extract_xml(
                    response_text.replace('[sub-tasks]', '<sub-tasks>').replace('[/sub-tasks]', '</sub-tasks>'), 
                    'sub-tasks'
                )
                if extracted_sub_tasks:
                    sub_task_content = extracted_sub_tasks.strip()
                else:
                    sub_task_content = f'Invalid sub-tasks'
            else:
                main_rank_print(f"No response_text found for Level 1 response")

        # Check if mock mode is enabled for sub-tasks
        if mock_sub_task_sub_agent:
            main_rank_print(f"🔧 MOCK MODE: Using DEBUG sub-task content for Level 1 response {response_idx}")
            sub_task_content = 'DEBUG'
        
        if sub_task_content:
            if sub_task_content not in sub_task_groups:
                sub_task_groups[sub_task_content] = []
            sub_task_groups[sub_task_content].append(response_idx)
            main_rank_print(f"   🔍 Level 1 response {response_idx} -> sub_task: '{sub_task_content}'")
        else:
            main_rank_print(f"   ⚠️  Level 1 response {response_idx} has no sub-task content found")

    main_rank_print(f"   🔗 sub_task_groups '{sub_task_groups}")
    
    # For each group with multiple Level 1 responses, update rewards to use the maximum
    groups_processed = 0
    for sub_task_content, response_indices in sub_task_groups.items():
        if len(response_indices) > 1:
            groups_processed += 1
            main_rank_print(f"   🔗 Sub-task '{sub_task_content[:50]}...' has {len(response_indices)} Level 1 responses: {response_indices}")
            
            # Find the maximum reward among this group
            group_rewards = [reward_extra_info['sub_task_reward'][idx] for idx in response_indices]
            max_reward = max(group_rewards)
            
            main_rank_print(f"      Group rewards: {[f'{r:.3f}' for r in group_rewards]}, using max: {max_reward:.3f}")
            
            # Update all responses in this group to use the maximum reward
            for response_idx in response_indices:
                old_reward = reward_extra_info['sub_task_reward'][response_idx]
                reward_extra_info['sub_task_reward'][response_idx] = max_reward
                main_rank_print(f"      Response {response_idx}: {old_reward:.3f} -> {max_reward:.3f}")
        else:
            main_rank_print(f"   🔗 Sub-task '{sub_task_content[:50]}...' has 1 Level 1 response: {response_indices[0]}")
    
    main_rank_print(f"🔗 Unified group processing completed: {groups_processed} groups with multiple Level 1 responses were unified")
    
    return groups_processed
