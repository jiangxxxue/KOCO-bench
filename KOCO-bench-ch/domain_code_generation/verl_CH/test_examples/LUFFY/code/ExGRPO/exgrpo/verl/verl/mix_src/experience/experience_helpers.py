from .weighted_bucket_sampler import WeightedBucketSampler

import numpy as np
import random

import torch

from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
import verl.utils.torch_functional as verl_F
from verl import DataProto



def sample_experience_uids(self, valid_exp_uids, n_exp):
    """
    Sample uids and return (uids) according to bucket settings.
    """
    if n_exp <= 0:
        return [], []
    if self.exp_bucket_manager and self.bucket_manager:
        buckets_for_sampler = {}
        total_valid_uids = 0
        for bucket_key in self.bucket_manager.experience_bucket.keys():
            bucket_uids = self.bucket_manager.get_bucket_uids(bucket_key)
            valid_bucket_uids = [uid for uid in bucket_uids if uid in valid_exp_uids]
            if len(valid_bucket_uids) > 0:
                buckets_for_sampler[bucket_key] = valid_bucket_uids
                total_valid_uids += len(valid_bucket_uids)
        if len(buckets_for_sampler) > 0 and total_valid_uids > 0:
            weighted_sampler = WeightedBucketSampler(buckets_for_sampler, n_rollout=self.n_rollout, weighting_method=self.exp_bucket_manager)
            uids, _ = weighted_sampler.sample(n_exp)
        else:
            print("[WeightedBucketSampler] No valid buckets found, falling back to random sampling")
            uids = random.sample(valid_exp_uids, n_exp)
    else:
        uids = random.sample(valid_exp_uids, n_exp)
    return uids


def prepare_metric_inputs(self, uids):
    """
    Build all_inputs, all_exps, uid_exp_indices for metric computation.
    """
    all_inputs = []
    all_exps = []
    uid_exp_indices = {}
    idx = 0
    uid_list = []
    for i, uid in enumerate(uids):
        df_idx = self.dataset.dataframe[self.dataset.dataframe[self.dataset.uid_key] == uid][self.dataset.uid_key]
        if len(df_idx) == 0:
            raise IndexError
        real_idx = df_idx.values[0]
        if real_idx not in self.dataset.valid_indices:
            raise IndexError
        row_dict = self.dataset.dataframe.iloc[real_idx].to_dict()
        chat = row_dict[self.dataset.prompt_key]
        exps = self.experience_pool[uid]
        uid_exp_indices[uid] = (idx, idx + len(exps))
        for exp in exps:
            all_inputs.append(self.dataset.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False))
            all_exps.append(exp['text'] + self.dataset.tokenizer.eos_token)
            uid_list.append(uid)
        idx += len(exps)
    return all_inputs, all_exps, uid_exp_indices


def compute_postcal_metrics(self, all_inputs, all_exps):
    """
    Compute postcal_metrics for either 'ppl' or 'ent'. Returns list of floats aligned with all_inputs/all_exps.
    Uses same batching/padding logic as the original implementation.
    """
    BATCH_SIZE = 8
    all_metrics = []
    for start in range(0, len(all_inputs), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_inputs))
        batch_inputs = all_inputs[start:end]
        batch_exps = all_exps[start:end]
        real_batch_size = len(batch_inputs)
        pad_num = 0
        if real_batch_size < BATCH_SIZE:
            pad_num = BATCH_SIZE - real_batch_size
            dummy_input = self.dataset.tokenizer.pad_token * self.dataset.max_prompt_length if hasattr(self.dataset.tokenizer, 'pad_token') else ''
            dummy_exp = self.dataset.tokenizer.pad_token * self.dataset.max_target_length if hasattr(self.dataset.tokenizer, 'pad_token') else ''
            for _ in range(pad_num):
                batch_inputs.append(dummy_input)
                batch_exps.append(dummy_exp)
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_responses = []
        for input_chat_template, exp in zip(batch_inputs, batch_exps):
            full_seq = input_chat_template + exp
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=full_seq,
                tokenizer=self.dataset.tokenizer,
                max_length=self.dataset.max_prompt_length + self.dataset.max_target_length,
                pad_token_id=self.dataset.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.dataset.truncation)
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_input_ids.append(input_ids[0])
            batch_attention_mask.append(attention_mask[0])
            batch_position_ids.append(position_ids[0])

            tgt_input_ids = self.dataset.tokenizer(exp, add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1)
            tgt_input_ids = tgt_input_ids.reshape(1, -1)
            if tgt_input_ids.shape[-1] < self.dataset.max_target_length:
                tgt_input_ids = pad_sequence_to_length(tgt_input_ids,
                    max_seq_len=self.dataset.max_target_length,
                    pad_token_id=self.dataset.tokenizer.pad_token_id,
                    left_pad=False)
            else:
                assert self.dataset.truncation in ('right', 'error')
                tgt_input_ids = tgt_input_ids[:, :self.dataset.max_target_length]
            tgt_input_ids = tgt_input_ids.squeeze(0)
            batch_responses.append(tgt_input_ids)
        batch_dict = {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'responses': torch.stack(batch_responses),
            'position_ids': torch.stack(batch_position_ids),
        }
        ppl_batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = self.actor_rollout_wg.compute_log_prob_ents(ppl_batch)
        if self.exp_metric == 'ppl':
            all_metrics.append(metrics["old_log_probs"].batch[:real_batch_size])
        elif self.exp_metric == 'ent':
            all_metrics.append(metrics["ents"].batch[:real_batch_size])
        else:
            raise ValueError(f"Invalid exp_metric: {self.exp_metric}")
    if len(all_metrics) == 0:
        return []
    all_metrics = torch.cat(all_metrics, dim=0)
    postcal_metrics = []
    if self.exp_metric == 'ppl':
        for i in range(all_metrics.shape[0]):
            mask = ~(ppl_batch.batch['responses'][i % BATCH_SIZE] != self.dataset.tokenizer.pad_token_id)
            nll = -all_metrics[i][mask].mean().item()
            ppl = np.exp(nll)
            postcal_metrics.append(ppl)
    elif self.exp_metric == 'ent':
        for i in range(all_metrics.shape[0]):
            mask = ~(ppl_batch.batch['responses'][i % BATCH_SIZE] != self.dataset.tokenizer.pad_token_id)
            ent = all_metrics[i][mask].mean().item()
            postcal_metrics.append(ent)
    else:
        raise ValueError(f"Invalid exp_metric: {self.exp_metric}")
    return postcal_metrics


def select_best_experiences(self, uids, uid_exp_indices, postcal_metrics):
    """
    For each uid, pick best experience index according to exp_select_mode and build maps.
    Returns best_exp_per_uid.
    """
    best_exp_per_uid = {}
    for i, uid in enumerate(uids):
        start, end = uid_exp_indices[uid]
        uid_postcal_metrics = postcal_metrics[start:end]
        if self.exp_select_mode == 'argmin':
            best_idx = int(np.argmin(uid_postcal_metrics))
        elif self.exp_select_mode == 'argmax':
            best_idx = int(np.argmax(uid_postcal_metrics))
        else:
            raise ValueError(f"Invalid exp_select_mode: {self.exp_select_mode}")
        best_exp_per_uid[uid] = self.experience_pool[uid][best_idx]
    return best_exp_per_uid


def build_off_policy_sample(self, uid, exp_entry, idx_uid):
    """
    Build and return one off-policy row_dict for given uid and chosen exp_entry.
    """
    df_idx = self.dataset.dataframe[self.dataset.dataframe[self.dataset.uid_key] == uid][self.dataset.uid_key]
    if len(df_idx) == 0:
        raise IndexError
    real_idx = df_idx.values[0]
    if real_idx not in self.dataset.valid_indices:
        raise IndexError

    row_dict = self.dataset.dataframe.iloc[real_idx].to_dict()

    if isinstance(exp_entry, dict):
        exp_str = exp_entry['text']
        exp_old_log_prob = exp_entry['old_log_prob']
    else:
        raise ValueError(f"Invalid exp_entry: {exp_entry}")

    tgt = {"content": exp_str, "role": "assistant"}
    row_dict[self.dataset.target_key] = [tgt]
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
    if tgt is not None:
        tgt = tgt[0]
        if prompt_with_chat_template.endswith('<think>\n') and tgt['content'].startswith('<think>\n'):
            tgt['content'] = tgt['content'][len('<think>\n'):]
        tgt_input_ids = self.dataset.tokenizer(tgt['content'], add_special_tokens=False, return_tensors='pt')['input_ids'].reshape(-1)
        tgt_input_ids = tgt_input_ids.reshape(1, -1)
    else:
        tgt_input_ids = torch.tensor([], dtype=torch.long).reshape(1, 0)
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
    index = row_dict.get("extra_info", {}).get("index", 0)
    row_dict["index"] = index
    row_dict['is_off_policy'] = torch.tensor(True, dtype=torch.bool)

    if exp_old_log_prob is not None:
        exp_old_log_prob = torch.nn.functional.pad(exp_old_log_prob, (0, self.dataset.max_target_length - exp_old_log_prob.shape[-1]), value=0)
        row_dict['recorded_old_log_prob'] = exp_old_log_prob
    else:
        row_dict['recorded_old_log_prob'] = torch.zeros(self.dataset.max_target_length, dtype=torch.float32)
    return row_dict


def build_on_policy_sample(self, row_dict):
    """
    Return a normalized on-policy sample row based on given row_dict.
    """
    rd = row_dict.copy()
    rd['is_off_policy'] = torch.tensor(False, dtype=torch.bool)
    rd['recorded_old_log_prob'] = torch.zeros(self.dataset.max_target_length, dtype=torch.float32)
    return rd


