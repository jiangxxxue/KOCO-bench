from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd
import copy 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import pad_sequence_to_length
from ..rl_dataset_with_target import RLHFDatasetWithTarget

import logging
import os
import random
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


class RLHFDatasetWithTargetExperience(RLHFDatasetWithTarget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience = dict()
        self.uid_key = kwargs.get('uid_key', 'dataframe_uid')
        self.skip_uid_set = set()
        self.valid_indices = list(range(len(self.dataframe)))
        # Ensure dataframe_uid exists
        if 'dataframe_uid' not in self.dataframe.columns:
            self.dataframe['dataframe_uid'] = range(len(self.dataframe))

    def set_experience(self, experience):
        self.experience = experience

    def set_skip_uid_set(self):
        # Filter out indices of samples that are fully solved.
        self.valid_indices = [
            i for i in range(len(self.dataframe))
            if self.dataframe.iloc[i][self.uid_key] not in self.skip_uid_set
        ]

    def __len__(self): 
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row_dict = self.dataframe.iloc[real_idx].to_dict()
        uid = self.dataframe.iloc[real_idx]['dataframe_uid']
        # row_dict['dataframe_uid'] = self.dataframe.iloc[real_idx]['dataframe_uid']
        # uid = row_dict.get(self.uid_key, None)

        # Check for experience and replace the target if available
        if self.experience is not None and uid is not None and uid in self.experience and len(self.experience[uid]) > 0:
            chosen_experience = random.choice(self.experience[uid])["text"]
            tgt = {"content": chosen_experience, "role": "assistant"}
            row_dict[self.target_key] = [tgt]
        else:
            # remove external policy targets 
            row_dict[self.target_key] = None

        # --- Start of inlined logic from parent __getitem__ ---
        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['is_off_policy'] = torch.tensor(False, dtype=torch.bool) # default, will assign in mix_trainer_experience.py
        
        tgt = row_dict.pop(self.target_key)
        sample = np.random.rand() < self.sample_target_ratio
        
        if tgt is not None and sample is True:
            tgt = tgt[0]
        
            if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
                tgt['content'] = tgt['content'][len('<think>\n'):]
            tgt_input_ids = self.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1) # [1, l]

            # NOTE: we don't need to do this because we add eos token id at mix_vllm_rollout.py
            
            # if tgt_input_ids[-1].item() != self.tokenizer.eos_token_id:
            #     eos_tensor = torch.tensor([self.tokenizer.eos_token_id], device=tgt_input_ids.device, dtype=tgt_input_ids.dtype).reshape(-1)
            #     tgt_input_ids = torch.cat([tgt_input_ids, eos_tensor], dim=-1)
            
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
        else:
            tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0) # empty target, will be pad to max_target_length

        # padding or truncate
        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.max_target_length:
            # right pad for tgt_input_ids
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                                            max_seq_len=self.max_target_length,
                                            pad_token_id=self.tokenizer.pad_token_id,
                                            left_pad=False)
        else:
            assert self.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.max_target_length]
        
        tgt_input_ids = tgt_input_ids.squeeze(0)

        row_dict['tgt_input_ids'] = tgt_input_ids

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict