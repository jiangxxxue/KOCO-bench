import numpy as np
import torch

from .experience_bucket_manager import CONSTANT_DATA_UID


def collect_experience_entries(self, batch, reward_tensor, success_value, metrics):
    """
    Collect successful generations into the experience pool during mixed-policy training.
    """
    if not self.train_collate_fn.is_mixed_policy:
        return

    responses = batch.batch['responses']
    responses_mask = batch.batch['attention_mask'][:, -responses.shape[-1]:].bool()
    uids = batch.non_tensor_batch['uid']
    dataframe_uids = batch.non_tensor_batch['dataframe_uid']
    uid2dfuid = {uid: df_uid for uid, df_uid in zip(uids, dataframe_uids)}
    unique_uids = np.unique(uids)
    picked_success = []

    for uid in unique_uids:
        uid_mask = uids == uid
        uid_rewards = reward_tensor[uid_mask].sum(-1)
        df_uid = uid2dfuid[uid]

        if df_uid == CONSTANT_DATA_UID:
            print("constant data.")
            continue

        if (uid_rewards == success_value).all():
            continue

        num_success = (uid_rewards == success_value).sum().item()
        total = len(uid_rewards)

        assert total == self.config.actor_rollout_ref.rollout.n

        if (num_success > self.config.trainer.experience_rbound) or (num_success <= self.config.trainer.experience_lbound):
            continue
        picked_success.append(num_success)

        if self.use_bucket_manager:
            self.train_collate_fn.update_bucket(df_uid, num_success)

        for idx, is_true in enumerate((uid_rewards == success_value).tolist()):
            if is_true:
                gen_mask = responses_mask[uid_mask][idx]
                gen_ids = responses[uid_mask][idx][gen_mask]
                gen_str = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False).rstrip(self.tokenizer.eos_token) # We will add back in vLLM rollout. This is to ensure intermediate special tokens are not skipped, so that it matches old_log_probs shape
                # get the corresponding old_log_prob - using the current calculated one, not the recorded one
                gen_old_log_prob = batch.batch['old_log_probs'][uid_mask][idx][gen_mask] # save valid part, save for memory
                exp_entry = {
                    'text': gen_str,
                    'old_log_prob': gen_old_log_prob.detach().cpu()
                }

                if df_uid in self.train_dataset.experience:
                    existing_texts = [exp['text'] for exp in self.train_dataset.experience[df_uid]]
                    if gen_str not in existing_texts:
                        self.train_dataset.experience[df_uid].append(exp_entry)
                else:
                    self.train_dataset.experience[df_uid] = [exp_entry]

    if 'gen_str' in locals():
        print(gen_str)
    if self.use_bucket_manager:
        bucket_stats = self.train_collate_fn.get_bucket_stats()
        metrics.update(bucket_stats)

    metrics['experience/picked_success_max'] = max(picked_success) if picked_success else 0
    metrics['experience/picked_success_min'] = min(picked_success) if picked_success else 0
    metrics['experience/picked_success_mean'] = sum(picked_success) / len(picked_success) if len(picked_success) > 0 else 0


def replace_recorded_old_log_probs(self, batch, off_policy_mask, prefix_mask):
    """
    Optionally replace computed old_log_probs with recorded ones for experience samples.
    """
    if not (self.train_collate_fn.is_mixed_policy and 'recorded_old_log_prob' in batch.batch):
        return

    recorded_old_log_probs = batch.batch['recorded_old_log_prob']
    if self.config.trainer.exp_is_correct and (recorded_old_log_probs.sum().item() != 0.0):
        for i in range(len(off_policy_mask)):
            if off_policy_mask[i] and recorded_old_log_probs[i].sum().item() != 0.0:
                recorded_old_log_prob = recorded_old_log_probs[i].to(batch.batch["old_log_probs"].device)
                batch.batch["old_log_probs"][i] = recorded_old_log_prob

                valid_response = batch.batch["responses"][i][prefix_mask[i]]
                pad_len = batch.batch["responses"][i].shape[-1] - prefix_mask[i].sum().item()
                if pad_len > 0:
                    pad_tensor = torch.ones(pad_len, dtype=torch.int64) * self.tokenizer.pad_token_id
                    batch.batch["responses"][i] = torch.cat([valid_response, pad_tensor], dim=-1)
                else:
                    batch.batch["responses"][i] = valid_response
                batch.batch["input_ids"][i][~batch.batch["attention_mask"][i].bool()] = self.tokenizer.pad_token_id
    else:
        print("no exp_is_correct, not replacing recorded_old_log_probs")


def cleanup_experience_buckets(self):
    """
    Remove entries in experience that are now in skip_uid_set and maintain buckets.
    """
    for df_uid in list(self.train_dataset.experience.keys()):
        if df_uid in self.train_dataset.skip_uid_set:
            del self.train_dataset.experience[df_uid]
            if self.use_bucket_manager:
                self.train_collate_fn.remove_from_bucket(df_uid)

    if self.use_bucket_manager:
        self.train_collate_fn.cleanup_bucket_consistency()


