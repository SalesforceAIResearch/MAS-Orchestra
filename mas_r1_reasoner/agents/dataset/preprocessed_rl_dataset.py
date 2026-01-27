"""
Preprocessed RL Dataset for MAS-R1 Stage 1 training.
This dataset loads preprocessed data that has already been tokenized and formatted.
"""

import os
import copy
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Union
from omegaconf import ListConfig


class PreprocessedRLDataset(Dataset):
    """
    Dataset for loading preprocessed Stage 1 data that has already been tokenized.
    This bypasses the need for chat template application and tokenization during training.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer=None,  # Not used for preprocessed data
                 prompt_key='prompt_text',  # Key for the raw prompt text
                 input_ids_key='input_ids',  # Key for preprocessed input_ids
                 attention_mask_key='attention_mask',  # Key for preprocessed attention_mask
                 max_prompt_length=None,  # Not used for preprocessed data
                 filter_prompts=False,  # Not used for preprocessed data
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,  # Not used for preprocessed data
                 return_raw_chat=True,  # Always return raw chat for preprocessed data
                 truncation='error',  # Not used for preprocessed data
                 extra_source_key=None,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer  # Keep for compatibility but not used
        self.extra_source_key = extra_source_key

        self.prompt_key = prompt_key
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.max_prompt_length = max_prompt_length  # Keep for compatibility
        self.filter_prompts = filter_prompts  # Keep for compatibility

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func  # Keep for compatibility
        self.truncation = truncation  # Keep for compatibility

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self):
        """Read preprocessed parquet files without tokenization"""
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'Preprocessed dataset len: {len(self.dataframe)}{". Source: " + self.extra_source_key if self.extra_source_key else ""}')
        print(f'Using preprocessed data - no tokenization or filtering needed')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Return preprocessed data directly without additional tokenization.
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        # Get preprocessed input_ids and attention_mask
        input_ids = row_dict[self.input_ids_key]
        attention_mask = row_dict[self.attention_mask_key]
        
        # Convert numpy arrays to torch tensors
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long()
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).long()

        # Compute position_ids from attention_mask
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        # Store the processed tensors
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids

        # Store raw prompt for compatibility
        if self.return_raw_chat:
            raw_prompt = row_dict.get(self.prompt_key, [])
            if isinstance(raw_prompt, str):
                # If it's a string, convert to list format for compatibility
                row_dict['raw_prompt'] = [{"role": "user", "content": raw_prompt}]
            else:
                row_dict['raw_prompt'] = raw_prompt

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", item)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy() 