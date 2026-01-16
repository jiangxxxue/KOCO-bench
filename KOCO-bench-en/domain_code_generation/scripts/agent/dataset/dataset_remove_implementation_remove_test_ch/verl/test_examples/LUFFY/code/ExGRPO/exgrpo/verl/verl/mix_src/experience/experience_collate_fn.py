from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
import verl.utils.torch_functional as verl_F
from verl import DataProto

import random
import torch

from .experience_bucket_manager import ExperienceBucketManager, CONSTANT_DATA_UID
from .experience_helpers import (
    sample_experience_uids,
    prepare_metric_inputs,
    compute_postcal_metrics,
    select_best_experiences,
    build_off_policy_sample,
    build_on_policy_sample,
)

class ExperienceMixCollateFn:
    def __init__(self, base_collate_fn, dataset, experience_pool, skip_uid_set, n_rollout, exp_ratio=0.5, exp_metric='ent', actor_rollout_wg=None, constant_data=None, exp_select_mode='argmin', exp_sample_range='all', exp_bucket_manager=None):
        self.base_collate_fn = base_collate_fn
        self.dataset = dataset
        self.experience_pool = experience_pool  # dict[uid] = [exp1, exp2, ...]
        self.skip_uid_set = skip_uid_set
        self.n_rollout = n_rollout
        self.exp_ratio = exp_ratio
        self.exp_metric = exp_metric
        self.is_mixed_policy = False  # flag only
        self.actor_rollout_wg = actor_rollout_wg
        self.constant_data = None
        self.exp_select_mode = exp_select_mode
        self.exp_sample_range = exp_sample_range  # 'all', 'first_half', 'last_half'
        self.exp_bucket_manager = exp_bucket_manager
        
        # ========== Initialize bucket manager based on switch ==========
        if self.exp_bucket_manager:
            self.bucket_manager = ExperienceBucketManager()
        else:
            self.bucket_manager = None  
        
        if self.constant_data:
            print(self.constant_data)
            print(self.dataset.tokenizer.decode(self.constant_data["tgt_input_ids"], skip_special_tokens=True))
        else:
            print("no constant batch data.")

    def set_mixed_policy(self, flag: bool):
        self.is_mixed_policy = flag

    def __getstate__(self):
        """Called during serialization, excludes non-serializable objects"""
        state = self.__dict__.copy()
        # Exclude actor_rollout_wg as it's a Ray object that cannot be serialized
        state['actor_rollout_wg'] = None
        print("[ExperienceMixCollateFn] Serializing: actor_rollout_wg excluded")
        return state

    def __setstate__(self, state):
        """Called during deserialization, restores object state"""
        self.__dict__.update(state)
        # actor_rollout_wg will be set later through set_actor_rollout_wg method        
        print("[ExperienceMixCollateFn] Deserialized: actor_rollout_wg needs to be set later")

    def set_actor_rollout_wg(self, actor_rollout_wg):
        """Set actor_rollout_wg reference"""
        self.actor_rollout_wg = actor_rollout_wg
        print("[ExperienceMixCollateFn] actor_rollout_wg reference restored")
    
    def update_exp_ratio(self, new_ratio):
        """Update experience ratio"""
        self.exp_ratio = new_ratio
    
    def update_bucket(self, df_uid, num_success):
        """
        Update bucket grouping information
        """
        if self.bucket_manager:
            self.bucket_manager.update_bucket(df_uid, num_success)
    
    def remove_from_bucket(self, df_uid):
        """
        Remove specified df_uid from bucket
        """
        if self.bucket_manager:
            self.bucket_manager.remove_from_bucket(df_uid)
    
    def cleanup_bucket_consistency(self):
        """
        Maintain bucket consistency
        """
        if self.bucket_manager:
            self.bucket_manager.cleanup_consistency(self.experience_pool)
    
    def get_bucket_stats(self):
        """
        Get bucket statistics
        """
        if self.bucket_manager:
            return self.bucket_manager.get_bucket_stats()
        return {}
    
    def print_bucket_stats(self):
        """
        Print bucket statistics
        """
        if self.bucket_manager:
            self.bucket_manager.print_bucket_stats()

    def __call__(self, data_list):
        print(f"[ExperienceMixCollateFn] exp_select_mode: {self.exp_select_mode}; exp_metric: {self.exp_metric}; range: {self.exp_sample_range}; exp_bucket_manager: {self.exp_bucket_manager}")
        if not self.is_mixed_policy:
            for i, row_dict in enumerate(data_list):
                data_list[i]['is_off_policy'] = torch.tensor(False, dtype=torch.bool)
                data_list[i]['recorded_old_log_prob'] = None
            return self.base_collate_fn(data_list)
        # Filter out experiences in skip_uid_set
        valid_exp_uids = [uid for uid in self.experience_pool if uid not in self.skip_uid_set and len(self.experience_pool[uid]) > 0]
        
        # Limit sampling range based on sample_range parameter
        if self.exp_sample_range == 'first_half':
            half_size = len(valid_exp_uids) // 2
            valid_exp_uids = valid_exp_uids[:half_size]
        elif self.exp_sample_range == 'last_half':
            half_size = len(valid_exp_uids) // 2
            valid_exp_uids = valid_exp_uids[half_size:]
        elif self.exp_sample_range == 'all':
            pass
        else:
            raise ValueError(f"Invalid sample_range: {self.exp_sample_range}. Must be 'all', 'first_half', or 'last_half'")
        
        target_batch_size = len(data_list) # 128
        
        # Calculate target number of experience samples
        target_exp_count = int(target_batch_size * self.exp_ratio)
        n_exp = min(len(valid_exp_uids), target_exp_count)
        
        # print("raw sample keys:", data_list[0].keys())
        # print(valid_exp_uids)
        exp_samples = []
        if n_exp > 0:
            uids = sample_experience_uids(self, valid_exp_uids, n_exp)
            if self.exp_metric == 'ppl' or self.exp_metric == 'ent':
                all_inputs, all_exps, uid_exp_indices = prepare_metric_inputs(self, uids)
                postcal_metrics = compute_postcal_metrics(self, all_inputs, all_exps)
                best_exp_per_uid = select_best_experiences(self, uids, uid_exp_indices, postcal_metrics)
            for idx_uid, uid in enumerate(uids):
                if self.exp_metric == 'random':
                    exp_entry = random.choice(self.experience_pool[uid])
                elif self.exp_metric == 'ppl' or self.exp_metric == 'ent':
                    exp_entry = best_exp_per_uid[uid]
                else:
                    raise ValueError(f"Invalid exp_metric: {self.exp_metric}")
                row_dict = build_off_policy_sample(self, uid, exp_entry, idx_uid)
                exp_samples.append(row_dict)

        # print("row_dict keys:", row_dict.keys())
        # Ensure final batch size equals input size
        n_exp_actual = len(exp_samples) # Actual number of experience samples
        n_raw = target_batch_size - n_exp_actual # Number of original samples needed to supplement
        
        # If experience samples are insufficient, fill with original On-Policy samples
        if n_raw > 0:
            raw_samples = []
            for row_dict in data_list[:n_raw]:
                raw_samples.append(build_on_policy_sample(self, row_dict))           
        else:
            raw_samples = []
        
        # If there are too many experience samples, take only the needed amount
        if n_exp_actual > target_batch_size:
            exp_samples = exp_samples[:target_batch_size]
            raw_samples = []
        
        if self.constant_data is not None and n_exp_actual > 0:
            exp_samples = exp_samples[:-1]
            exp_samples.append(self.constant_data)

        mixed_data = exp_samples + raw_samples
        # Ensure final size is correct
        assert len(mixed_data) == target_batch_size, f"Expected batch size {target_batch_size}, got {len(mixed_data)}"
        random.shuffle(mixed_data)
        # print("mixed data keys:", mixed_data[0].keys())
        return self.base_collate_fn(mixed_data)

    def _process_constant_data(self, constant_data):
        """Process constant data, return formatted sample"""
        if constant_data and isinstance(constant_data, dict):
            prompt = constant_data.get("prompt", "")
            target = constant_data.get("target", "")
        else:
            return None
    
        # Create row_dict
        row_dict = constant_data
        # Reuse dataset processing logic

        chat = row_dict.pop(self.dataset.prompt_key)
        prompt_with_chat_template = self.dataset.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.dataset.tokenizer,
            max_length=self.dataset.max_prompt_length,
            pad_token_id=self.dataset.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.dataset.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        tgt = row_dict.pop(self.dataset.target_key) 
        assert tgt is not None
        tgt = tgt[0]
        tgt['content'] = tgt['content'] + self.dataset.tokenizer.eos_token
        if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
            tgt['content'] = tgt['content'][len('<think>\n'):]
        tgt_input_ids = self.dataset.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1)
        tgt_input_ids = tgt_input_ids.reshape(1, -1)

        sequence_length = tgt_input_ids.shape[-1]
        if sequence_length < self.dataset.max_target_length:
            tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                max_seq_len=self.dataset.max_target_length,
                pad_token_id=self.dataset.tokenizer.pad_token_id,
                left_pad=False)
        else:
            assert self.dataset.truncation in ('right', 'error')
            tgt_input_ids = tgt_input_ids[:, :self.dataset.max_target_length]
        tgt_input_ids = tgt_input_ids.squeeze(0)
        row_dict['tgt_input_ids'] = tgt_input_ids        
        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict['is_off_policy'] = torch.tensor(True, dtype=torch.bool)

        # row_dict['is_constant_data'] = torch.tensor(True, dtype=torch.bool)  # constant label, dropped
        row_dict["dataframe_uid"] = CONSTANT_DATA_UID # dummy uid
        # Create a zero tensor for constant data as well for consistency
        row_dict['recorded_old_log_prob'] = torch.zeros(self.dataset.max_target_length, dtype=torch.float32)

        return row_dict
