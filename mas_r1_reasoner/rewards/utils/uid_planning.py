#!/usr/bin/env python3
"""
UID Planning Module for MAS-R1

This module provides different strategies for assigning UIDs to Level 2 responses
based on the grouping strategy (expansive_group vs unified_group).
"""

import uuid
import numpy as np
from typing import List, Tuple
from mas_r1_reasoner.agents.common import main_rank_print


def plan_uids_expansive_group(
    original_uids: List,
    level_1_keys: List[str],
    level_2_keys: List[str],
    trainer_instance=None
) -> Tuple[List, dict]:
    """
    Plan UIDs for expansive_group strategy.
    
    In this strategy:
    - Level 1 responses keep their original UIDs
    - Level 2 responses get NEW group UIDs (same UID for children of same parent)
    - Each Level 1 parent gets a unique group UID for all its Level 2 children
    
    Args:
        original_uids: List of existing UIDs for Level 1 responses
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        trainer_instance: Trainer instance (optional, for future use)
        
    Returns:
        Tuple of (expanded_uids, uid_mapping_info)
    """
    main_rank_print(f"📋 Planning UIDs for EXPANSIVE_GROUP strategy")
    main_rank_print(f"   - Level 1 responses: {len(level_1_keys)} (keep original UIDs)")
    main_rank_print(f"   - Level 2 responses: {len(level_2_keys)} (get new group UIDs)")
    
    # Start with original Level 1 UIDs
    expanded_uids = list(original_uids)
    
    # Calculate how many Level 2 responses we need to add per Level 1 parent
    level_1_count = len(level_1_keys)
    level_2_count = len(level_2_keys)
    responses_per_level_1 = level_2_count // level_1_count
    
    main_rank_print(f"   - Responses per Level 1 parent: {responses_per_level_1}")
    
    # Track UID mapping for validation
    uid_mapping_info = {
        'strategy': 'expansive_group',
        'level_1_uids': original_uids.copy(),
        'level_2_group_uids': {},
        'parent_child_mapping': {}
    }
    
    # For each Level 1 response, add Level 2 responses with NEW GROUP UIDs
    for level_1_idx in range(level_1_count):
        # Generate ONE new group UID for all Level 2 children of this Level 1 parent
        group_uid = str(uuid.uuid4())
        
        # Store the mapping info
        uid_mapping_info['level_2_group_uids'][level_1_idx] = group_uid
        uid_mapping_info['parent_child_mapping'][level_1_idx] = []
        
        # Add Level 2 responses with the SAME new group UID
        for _ in range(responses_per_level_1):
            expanded_uids.append(group_uid)
            uid_mapping_info['parent_child_mapping'][level_1_idx].append(len(expanded_uids) - 1)
        
        main_rank_print(f"   - Level 1 Response {level_1_idx}: Generated NEW group UID {group_uid[:8]}... for {responses_per_level_1} Level 2 children")
    
    main_rank_print(f"✅ EXPANSIVE_GROUP UID planning completed")
    main_rank_print(f"   - Total UIDs: {len(expanded_uids)} ({len(original_uids)} original + {len(expanded_uids) - len(original_uids)} new)")
    
    return expanded_uids, uid_mapping_info


def plan_uids_unified_group(
    original_uids: List,
    level_1_keys: List[str],
    level_2_keys: List[str],
    trainer_instance=None
) -> Tuple[List, dict]:
    """
    Plan UIDs for unified_group strategy.
    
    In this strategy:
    - Level 1 responses keep their original UIDs
    - Level 2 responses get the SAME UID as their corresponding Level 1 parent
    - No new UIDs are generated - Level 2 children inherit parent UIDs
    
    Args:
        original_uids: List of existing UIDs for Level 1 responses
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        trainer_instance: Trainer instance (optional, for future use)
        
    Returns:
        Tuple of (expanded_uids, uid_mapping_info)
    """
    main_rank_print(f"📋 Planning UIDs for UNIFIED_GROUP strategy")
    main_rank_print(f"   - Level 1 responses: {len(level_1_keys)} (keep original UIDs)")
    main_rank_print(f"   - Level 2 responses: {len(level_2_keys)} (inherit parent UIDs)")
    
    # Start with original Level 1 UIDs
    expanded_uids = list(original_uids)
    
    # Calculate how many Level 2 responses we need to add per Level 1 parent
    level_1_count = len(level_1_keys)
    level_2_count = len(level_2_keys)
    responses_per_level_1 = level_2_count // level_1_count
    
    main_rank_print(f"   - Responses per Level 1 parent: {responses_per_level_1}")
    
    # Track UID mapping for validation
    uid_mapping_info = {
        'strategy': 'unified_group',
        'level_1_uids': original_uids.copy(),
        'level_2_inherited_uids': {},
        'parent_child_mapping': {}
    }
    
    # For each Level 1 response, add Level 2 responses with INHERITED UIDs
    for level_1_idx in range(level_1_count):
        # Get the UID from the corresponding Level 1 parent
        parent_uid = original_uids[level_1_idx]
        
        # Store the mapping info
        uid_mapping_info['level_2_inherited_uids'][level_1_idx] = parent_uid
        uid_mapping_info['parent_child_mapping'][level_1_idx] = []
        
        # Add Level 2 responses with the SAME UID as their Level 1 parent
        for _ in range(responses_per_level_1):
            expanded_uids.append(parent_uid)
            uid_mapping_info['parent_child_mapping'][level_1_idx].append(len(expanded_uids) - 1)
        
        main_rank_print(f"   - Level 1 Response {level_1_idx}: Level 2 children inherit UID {parent_uid[:8]}... ({responses_per_level_1} children)")
    
    main_rank_print(f"✅ UNIFIED_GROUP UID planning completed")
    main_rank_print(f"   - Total UIDs: {len(expanded_uids)} ({len(original_uids)} original + {len(expanded_uids) - len(original_uids)} inherited)")
    main_rank_print(f"   - No new UIDs generated - Level 2 children use parent UIDs")
    
    return expanded_uids, uid_mapping_info


def plan_uids_diff_based_reward(
    original_uids: List,
    level_1_keys: List[str],
    level_2_keys: List[str],
    trainer_instance=None
) -> Tuple[List, dict]:
    """
    Plan UIDs for diff_based_reward strategy.
    
    In this strategy:
    - Level 1 responses keep their original UIDs
    - Level 2 responses get a NEW SHARED UID for all responses from the same question
    - All Level 2 responses from the same question share the same new UID
    - This creates a new group specifically for Level 2 responses
    
    Tree Structure:
    - Level 0: Questions (e.g., 4 questions)
    - Level 1: First-level responses (e.g., 4 questions × rollout.n = 16 responses)
    - Level 2: Second-level responses (e.g., 16 Level 1 responses × sub_agents_per_sub_task = 32 responses)
    
    For diff_based_reward:
    - Level 1 responses from the same question keep their original UIDs (forming one group)
    - Level 2 responses from the same question get a new shared UID (forming another group)
    - So each question contributes 2 groups: one for Level 1, one for Level 2
    
    Args:
        original_uids: List of existing UIDs for Level 1 responses
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        trainer_instance: Trainer instance to access rollout.n configuration
        
    Returns:
        Tuple of (expanded_uids, uid_mapping_info)
    """
    main_rank_print(f"📋 Planning UIDs for DIFF_BASED_REWARD strategy")
    main_rank_print(f"   - Level 1 responses: {len(level_1_keys)} (keep original UIDs)")
    main_rank_print(f"   - Level 2 responses: {len(level_2_keys)} (get new shared UID per question)")
    
    # Start with original Level 1 UIDs
    expanded_uids = list(original_uids)
    
    # Get rollout.n from trainer instance to understand the tree structure
    if trainer_instance is None:
        raise RuntimeError("trainer_instance is required for diff_based_reward UID planning to calculate question groups")
    
    try:
        rollout_n = trainer_instance.config.actor_rollout_ref.rollout.n
        main_rank_print(f"   - Rollout.n from trainer config: {rollout_n}")
    except Exception as e:
        raise RuntimeError(f"Could not get rollout.n from trainer config: {e}")
    
    # Calculate unique question groups: level_1_count // rollout.n
    level_1_count = len(level_1_keys)
    level_2_count = len(level_2_keys)
    unique_question_groups = level_1_count // rollout_n
    level_2_per_question = level_2_count // unique_question_groups
    
    main_rank_print(f"   - Unique question groups: {level_1_count} // {rollout_n} = {unique_question_groups}")
    main_rank_print(f"   - Level 2 responses per question: {level_2_count} // {unique_question_groups} = {level_2_per_question}")
    
    # Track UID mapping for validation
    uid_mapping_info = {
        'strategy': 'diff_based_reward',
        'level_1_uids': original_uids.copy(),
        'level_2_shared_uids': {},
        'parent_child_mapping': {},
        'question_group_mapping': {}
    }
    
    # For each question group, assign Level 2 responses with a NEW SHARED UID
    for question_idx in range(unique_question_groups):
        # Generate ONE new shared UID for all Level 2 responses from this question
        shared_uid = str(uuid.uuid4())
        
        # Calculate the range of Level 1 responses for this question
        level_1_start = question_idx * rollout_n
        level_1_end = level_1_start + rollout_n
        
        # Calculate the range of Level 2 responses for this question
        level_2_start = question_idx * level_2_per_question
        level_2_end = level_2_start + level_2_per_question
            
        # Store the mapping info for this question
        uid_mapping_info['level_2_shared_uids'][question_idx] = shared_uid
        uid_mapping_info['question_group_mapping'][question_idx] = {
            'level_1_range': (level_1_start, level_1_end),
            'level_2_range': (level_2_start, level_2_end),
            'level_1_uids': original_uids[level_1_start:level_1_end],
            'level_2_shared_uid': shared_uid
        }
        
        # Add Level 2 responses with the SAME new shared UID for this question
        for _ in range(level_2_per_question):
            expanded_uids.append(shared_uid)
        
        main_rank_print(f"   - Question {question_idx}: Generated NEW shared UID {shared_uid[:8]}... for {level_2_per_question} Level 2 responses")
        main_rank_print(f"     Level 1 responses {level_1_start}-{level_1_end-1}: keep original UIDs")
        main_rank_print(f"     Level 2 responses {level_2_start}-{level_2_end-1}: share UID {shared_uid[:8]}...")
    
    main_rank_print(f"✅ DIFF_BASED_REWARD UID planning completed")
    main_rank_print(f"   - Total UIDs: {len(expanded_uids)} ({len(original_uids)} original + {len(expanded_uids) - len(original_uids)} new shared)")
    main_rank_print(f"   - Each question creates 2 groups: Level 1 (original UIDs) + Level 2 (new shared UID)")
    main_rank_print(f"   - Total groups: {unique_question_groups} × 2 = {unique_question_groups * 2}")
    
    return expanded_uids, uid_mapping_info


def validate_uid_planning(
    expanded_uids: List,
    level_1_keys: List[str],
    level_2_keys: List[str],
    uid_mapping_info: dict,
    trainer_instance=None
) -> bool:
    """
    Validate the UID planning results.
    
    Args:
        expanded_uids: List of all UIDs after expansion
        level_1_keys: List of Level 1 response keys
        level_2_keys: List of Level 2 response keys
        uid_mapping_info: Information about the UID mapping strategy
        
    Returns:
        bool: True if validation passes, raises RuntimeError otherwise
    """
    main_rank_print(f"\n🔍 Validating UID planning consistency...")
    main_rank_print(f"   - Strategy: {uid_mapping_info['strategy']}")
    
    strategy = uid_mapping_info['strategy']
    level_1_count = len(level_1_keys)
    level_2_count = len(level_2_keys)
    
    # Calculate responses per Level 1, handling potential uneven distribution
    if level_1_count == 0:
        error_msg = "No Level 1 keys found - this indicates a fundamental problem in the data structure"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Calculate responses per Level 1, handling potential uneven distribution
    if level_1_count == 0:
        error_msg = "No Level 1 keys found - this indicates a fundamental problem in the data structure"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    responses_per_level_1 = level_2_count // level_1_count
    remainder = level_2_count % level_1_count
    
    main_rank_print(f"   - Level 1 responses: {level_1_count}")
    main_rank_print(f"   - Level 2 responses: {level_2_count}")
    main_rank_print(f"   - Base responses per Level 1: {responses_per_level_1}")
    main_rank_print(f"   - Remainder responses: {remainder}")
    remainder = level_2_count % level_1_count
    
    main_rank_print(f"   - Level 1 responses: {level_1_count}")
    main_rank_print(f"   - Level 2 responses: {level_2_count}")
    main_rank_print(f"   - Base responses per Level 1: {responses_per_level_1}")
    main_rank_print(f"   - Remainder responses: {remainder}")
    
    # Validate total length
    expected_total = level_1_count + level_2_count
    if len(expanded_uids) != expected_total:
        error_msg = f"UID expansion failed: expected {expected_total} UIDs, got {len(expanded_uids)}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # Validate Level 2 UIDs based on strategy
    level_2_uids = expanded_uids[level_1_count:]
    
    for level_1_idx in range(level_1_count):
        # Calculate start and end indices, distributing remainder evenly
        if level_1_idx < remainder:
            # First 'remainder' Level 1 responses get one extra Level 2 response
            start_idx = level_1_idx * (responses_per_level_1 + 1)
            end_idx = start_idx + (responses_per_level_1 + 1)
        else:
            # Remaining Level 1 responses get the base number of Level 2 responses
            start_idx = remainder * (responses_per_level_1 + 1) + (level_1_idx - remainder) * responses_per_level_1
            end_idx = start_idx + responses_per_level_1
        
        if end_idx > len(level_2_uids):
            error_msg = f"Level 2 UID index out of bounds for Level 1 parent {level_1_idx}: end_idx={end_idx}, available={len(level_2_uids)}"
            main_rank_print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        parent_children_uids = level_2_uids[start_idx:end_idx]
        
        if strategy == 'expansive_group':
            # All Level 2 children of the same Level 1 parent should have the same NEW group UID
            if len(set(parent_children_uids)) != 1:
                error_msg = f"EXPANSIVE_GROUP UID violation: Level 1 parent {level_1_idx} children have different UIDs: {parent_children_uids}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # The group UID should be different from the parent's original UID
            parent_uid = expanded_uids[level_1_idx]
            if parent_children_uids[0] == parent_uid:
                error_msg = f"EXPANSIVE_GROUP UID violation: Level 1 parent {level_1_idx} and its children have the same UID: {parent_uid}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            main_rank_print(f"   ✓ Level 1 Parent {level_1_idx}: All {responses_per_level_1} children have NEW group UID {parent_children_uids[0][:8]}...")
            
        elif strategy == 'unified_group':
            # All Level 2 children of the same Level 1 parent should have the same UID as their parent
            parent_uid = expanded_uids[level_1_idx]
            if not all(child_uid == parent_uid for child_uid in parent_children_uids):
                error_msg = f"UNIFIED_GROUP UID violation: Level 1 parent {level_1_idx} children don't match parent UID {parent_uid[:8]}..."
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            main_rank_print(f"   ✓ Level 1 Parent {level_1_idx}: All {responses_per_level_1} children inherit parent UID {parent_uid[:8]}...")
    
    # NEW: Validate total unique groups for GRPO
    unique_groups = set(expanded_uids)
    total_responses = len(expanded_uids)
    
    # Get rollout.n from trainer instance to calculate unique question groups
    try:
        rollout_n = trainer_instance.config.actor_rollout_ref.rollout.n
        main_rank_print(f"   - Rollout.n from trainer config: {rollout_n}")
    except Exception as e:
        raise RuntimeError(f"Could not get rollout.n from trainer config: {e}")

    # Calculate unique question groups: level_1_count // rollout.n
    unique_question_groups = level_1_count // rollout_n
    main_rank_print(f"   - Level 1 responses: {level_1_count}")
    main_rank_print(f"   - Unique question groups: {level_1_count} // {rollout_n} = {unique_question_groups}")

    
    # Validate Level 2 count: should be rollout_n * sub_agents_per_sub_task per question
    expected_level_2_per_question = rollout_n * responses_per_level_1  # responses_per_level_1 is sub_agents_per_sub_task
    expected_total_level_2 = unique_question_groups * expected_level_2_per_question
    
    if level_2_count != expected_total_level_2:
        error_msg = f"Level 2 count mismatch: expected {expected_total_level_2} (={unique_question_groups} questions × {rollout_n} rollout.n × {responses_per_level_1} sub_agents_per_sub_task), got {level_2_count}"
        main_rank_print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    main_rank_print(f"   - Level 2 responses: {level_2_count}")
    main_rank_print(f"   - Expected Level 2 per question: {rollout_n} × {responses_per_level_1} = {expected_level_2_per_question}")
    main_rank_print(f"   - Total expected Level 2: {expected_total_level_2}")
    
    main_rank_print(f"\n📊 GRPO Group Validation:")
    main_rank_print(f"   - Total responses: {total_responses}")
    main_rank_print(f"   - Unique groups: {len(unique_groups)}")
    main_rank_print(f"   - Group distribution:")
    
    # Count responses per group
    group_counts = {}
    for uid in expanded_uids:
        group_counts[uid] = group_counts.get(uid, 0) + 1
    
    # Show group distribution
    for uid, count in sorted(group_counts.items()):
        main_rank_print(f"     Group {uid[:8]}...: {count} responses")
    
    # Validate group distribution based on strategy
    if strategy == 'expansive_group':
        # In expansive_group: Level 1 responses keep their UIDs, Level 2 responses get new group UIDs
        expected_level_1_groups = unique_question_groups  # Each unique question gets one group for its Level 1 responses
        expected_level_2_groups = unique_question_groups  # Each unique question gets one new group UID for its Level 2 children
        expected_total_groups = expected_level_1_groups + expected_level_2_groups
        
        if len(unique_groups) != expected_total_groups:
            error_msg = f"EXPANSIVE_GROUP group count mismatch: expected {expected_total_groups} unique groups, got {len(unique_groups)}"
            main_rank_print(f"❌ {error_msg}")
            main_rank_print(f"   - Expected: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups}")
            main_rank_print(f"   - Actual: {len(unique_groups)} unique groups")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✅ EXPANSIVE_GROUP: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups} total groups")
        
    elif strategy == 'unified_group':
        # In unified_group: Level 1 responses keep their UIDs, Level 2 responses inherit parent UIDs
        expected_level_1_groups = unique_question_groups  # Each unique question gets one group for its Level 1 responses
        expected_level_2_groups = 0  # No new groups - Level 2 children inherit parent UIDs
        expected_total_groups = expected_level_1_groups + expected_level_2_groups
        
        if len(unique_groups) != expected_total_groups:
            error_msg = f"UNIFIED_GROUP group count mismatch: expected {expected_total_groups} unique groups, got {len(unique_groups)}"
            main_rank_print(f"❌ {error_msg}")
            main_rank_print(f"   - Expected: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups}")
            main_rank_print(f"   - Actual: {len(unique_groups)} unique groups")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✅ UNIFIED_GROUP: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups} total groups")
        main_rank_print(f"   ✅ All Level 2 responses inherit their Level 1 parent UIDs (no new groups created)")
    
    elif strategy == 'diff_based_reward':
        # In diff_based_reward: Level 1 responses keep their UIDs, Level 2 responses get new shared UIDs
        # Total groups = Level 1 groups + Level 2 groups = unique_question_groups + unique_question_groups = 2 * unique_question_groups
        expected_level_1_groups = unique_question_groups  # Each unique question keeps its original UID
        expected_level_2_groups = unique_question_groups  # Each unique question gets one new shared UID for its Level 2 responses
        expected_total_groups = expected_level_1_groups + expected_level_2_groups
        
        if len(unique_groups) != expected_total_groups:
            error_msg = f"DIFF_BASED_REWARD group count mismatch: expected {expected_total_groups} unique groups, got {len(unique_groups)}"
            main_rank_print(f"❌ {error_msg}")
            main_rank_print(f"   - Expected: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups}")
            main_rank_print(f"   - Actual: {len(unique_groups)} unique groups")
            raise RuntimeError(error_msg)
        
        main_rank_print(f"   ✅ DIFF_BASED_REWARD: {expected_level_1_groups} Level 1 groups + {expected_level_2_groups} Level 2 groups = {expected_total_groups} total groups")
        main_rank_print(f"   ✅ Level 2 responses from each question form new groups for diff-based reward computation")

    # Validate that each group has the expected number of responses
    main_rank_print(f"\n🔍 Validating group sizes...")
    for uid, count in group_counts.items():
        if strategy == 'expansive_group':
            # In expansive_group, each group should have either 1 response (Level 1) or responses_per_level_1 responses (Level 2)
            if count != 1 and count != responses_per_level_1:
                error_msg = f"EXPANSIVE_GROUP group size violation: Group {uid[:8]}... has {count} responses, expected 1 or {responses_per_level_1}"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            main_rank_print(f"   ✓ Group {uid[:8]}...: {count} responses (valid)")
            
        elif strategy == 'unified_group':
            # In unified_group, all responses (Level 1 + Level 2) from the same question belong to the same group
            # Each group contains: rollout.n Level 1 responses + (rollout.n * sub_agents_per_sub_task) Level 2 responses
            expected_group_size = rollout_n + (rollout_n * responses_per_level_1)
            if count != expected_group_size:
                error_msg = f"UNIFIED_GROUP group size violation: Group {uid[:8]}... has {count} responses, expected {expected_group_size} (rollout.n={rollout_n} Level 1 + rollout.n×sub_agents_per_sub_task={rollout_n}×{responses_per_level_1}={rollout_n * responses_per_level_1} Level 2)"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            main_rank_print(f"   ✓ Group {uid[:8]}...: {count} responses ({rollout_n} Level 1 + {rollout_n * responses_per_level_1} Level 2 from same question)")
            
        elif strategy == 'diff_based_reward':
            # In diff_based_reward: Level 1 responses keep their UIDs, Level 2 responses get new shared UIDs
            # Each group should have either rollout.n responses (Level 1) or (rollout.n * sub_agents_per_sub_task) responses (Level 2)
            if count != rollout_n and count != (rollout_n * responses_per_level_1):
                error_msg = f"DIFF_BASED_REWARD group size violation: Group {uid[:8]}... has {count} responses, expected {rollout_n} (Level 1) or {rollout_n * responses_per_level_1} (Level 2)"
                main_rank_print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            if count == rollout_n:
                main_rank_print(f"   ✓ Group {uid[:8]}...: {count} responses (Level 1 responses from same question)")
            else:
                main_rank_print(f"   ✓ Group {uid[:8]}...: {count} responses (Level 2 responses from same question)")
    
    main_rank_print(f"✅ UID planning validation passed for {strategy} strategy")
    main_rank_print(f"✅ GRPO will use {len(unique_groups)} groups for advantage computation")
    return True


def get_uid_planning_function(strategy: str):
    """
    Get the appropriate UID planning function based on strategy.
    
    Args:
        strategy: Either 'expansive_group', 'unified_group', or 'diff_based_reward'
        
    Returns:
        Function: The appropriate UID planning function
        
    Raises:
        ValueError: If strategy is not supported
    """
    if strategy == 'expansive_group':
        return plan_uids_expansive_group
    elif strategy == 'unified_group':
        return plan_uids_unified_group
    elif strategy == 'diff_based_reward':
        return plan_uids_diff_based_reward
    else:
        raise ValueError(f"Unsupported UID planning strategy: {strategy}. Must be 'expansive_group', 'unified_group', or 'diff_based_reward'") 