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
Extraction and generation utilities for MAS-R1 Trainer.
"""

from typing import Dict, Tuple, Any, List
from verl.protocol import DataProto
from mas_r1_reasoner.agents.common import main_rank_print
from mas_r1_reasoner.agents.code_sanity import extract_code_from_response, validate_python_code
from mas_r1_reasoner.trainer.utils.helper import get_safe_length
import torch
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def rollout_generation(trainer_instance, gen_batch: DataProto, is_validation: bool) -> DataProto:
        # Generate sequences
    if is_validation:

        gen_batch.meta_info = {
        'eos_token_id': trainer_instance.tokenizer.eos_token_id,
        'pad_token_id': trainer_instance.tokenizer.pad_token_id,
        'recompute_log_prob': False,
        "do_sample": trainer_instance.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
        'validate': True,
    }       
        # For validation, pad to be divisible by dp_size
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, trainer_instance.actor_rollout_wg.world_size)
        mas_code_generation_output_padded = trainer_instance.actor_rollout_wg.generate_sequences(gen_batch_padded)
        mas_code_generation_output = unpad_dataproto(mas_code_generation_output_padded, pad_size=pad_size)
    else:
        mas_code_generation_output = trainer_instance.actor_rollout_wg.generate_sequences(gen_batch)

    return mas_code_generation_output