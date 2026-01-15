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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from dataclasses import dataclass, field
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo import core_algos

import torch
import json

from torch.utils.data import DataLoader, SequentialSampler
from verl.utils.dataset.rl_dataset import collate_fn, RLHFDataset

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean

from .experience.rl_dataset_with_experience import RLHFDatasetWithTargetExperience
from .experience.experience_collate_fn import ExperienceMixCollateFn
from .experience.experience_bucket_manager import CONSTANT_DATA_UID

from .mix_trainer_helper import compute_data_metrics_ours, apply_kl_penalty, compute_advantage
from .experience.experience_trainer_ops import collect_experience_entries, replace_recorded_old_log_probs, cleanup_experience_buckets

class ExperienceMixRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self.train_dataset = RLHFDatasetWithTargetExperience(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)


        self.use_bucket_manager = getattr(config.trainer, 'exp_bucket_manager', None)

        exp_ratio = self.config.trainer.experience_ratio
        # load json from self.config.trainer.constant_data
        if self.config.trainer.constant_data:
            constant_data = json.load(open(self.config.trainer.constant_data, 'r'))
        else:
            constant_data = None
        print(f'[ExperienceMix] Using experience_ratio={exp_ratio}, rollout_n={self.config.actor_rollout_ref.rollout.n}')
        self.train_collate_fn = ExperienceMixCollateFn(
            base_collate_fn=collate_fn,
            dataset=self.train_dataset,
            n_rollout=self.config.actor_rollout_ref.rollout.n,
            experience_pool=self.train_dataset.experience,
            skip_uid_set=self.train_dataset.skip_uid_set,
            exp_metric=self.config.trainer.exp_metric,
            exp_ratio=exp_ratio,
            actor_rollout_wg=None,  
            constant_data=constant_data, 
            exp_select_mode=self.config.trainer.exp_select_mode,
            exp_sample_range=self.config.trainer.exp_sample_range,
            exp_bucket_manager=self.use_bucket_manager  
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=self.train_collate_fn,
            sampler=sampler
        )
        self.train_collate_fn.set_mixed_policy(False)  # 初始on-policy
        
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        print("[Policy Switch] Start On-Policy training!")

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.


        if self.global_steps > 2:
            # resume from ckpt
            print("[ExperienceMixCollateFn] resume from ckpt")
            self.train_dataset.experience = self.train_dataloader.dataset.experience
            self.train_dataset.skip_uid_set = self.train_dataloader.dataset.skip_uid_set
            self.train_dataset.valid_indices = self.train_dataloader.dataset.valid_indices
            
            if len(self.train_dataloader.dataset.experience) > 0:
                self.train_collate_fn.dataset = self.train_dataset
                self.train_collate_fn.bucket_manager = self.train_dataloader.collate_fn.bucket_manager
                self.train_collate_fn.experience_pool = self.train_dataset.experience
                self.train_collate_fn.skip_uid_set =  self.train_dataset.skip_uid_set
                if hasattr(self.train_collate_fn, 'set_actor_rollout_wg'):
                    self.train_collate_fn.set_actor_rollout_wg(self.actor_rollout_wg)
                    print("[ExperienceMixCollateFn] restore actor_rollout_wg")
                self.train_collate_fn.set_mixed_policy(True)

        for _ in range(self.config.trainer.total_epochs):
            # if the length of availble data < batch size, we need to repeat the data
            BATCH_TOLERANCE = 20
            if len(self.train_dataloader.dataset) > self.config.data.train_batch_size + BATCH_TOLERANCE:
                # self.train_dataset.set_experience(self.experience)
                self.train_dataset.set_skip_uid_set()

            if self.config.data.shuffle:
                from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
                sampler = ResumableRandomSampler(data_source=self.train_dataset)
            else:
                sampler = SequentialSampler(data_source=self.train_dataset)

            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.config.data.train_batch_size,
                                            drop_last=True,
                                            collate_fn=self.train_collate_fn,
                                            sampler=sampler)

            print(f'@step: {self.global_steps} Size of train dataloader: {len(self.train_dataloader.dataset)}, Size of experience: {len(self.train_dataloader.dataset.experience)}, skiped samples: {len(self.train_dataloader.dataset.skip_uid_set)}, is_mixed_policy: {self.train_collate_fn.is_mixed_policy}, exp_is_weighting: {self.config.trainer.exp_is_weighting}')

            for batch_dict in self.train_dataloader:
                # check pool
                assert len(self.train_collate_fn.experience_pool) == len(self.train_dataset.experience)
                assert len(self.train_collate_fn.skip_uid_set) == len(self.train_dataset.skip_uid_set)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                metrics = {}
                timing_raw = {}

                metrics['experience/batch_size'] = len(batch_dict['input_ids'])
                metrics['experience/data_length'] = len(self.train_dataloader.dataset)
                metrics["experience/skip_set"] = len(self.train_dataloader.dataset.skip_uid_set)
                metrics["experience/experience"] = len(self.train_dataloader.dataset.experience)

                # pop those keys for generation
                try:
                    assert "is_off_policy" in batch.batch.keys()
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids', 'is_off_policy'])
                except Exception as e:
                    print(f"Error popping batch keys: {e}")
                    breakpoint()

                gen_batch.meta_info['global_steps'] = self.global_steps

                if self.train_collate_fn.is_mixed_policy:
                    gen_batch.meta_info['min_prefix_ratio'] = 1.0
                    gen_batch.meta_info['max_prefix_ratio'] = 1.0
                else:
                    gen_batch.meta_info['min_prefix_ratio'] = 0.0
                    gen_batch.meta_info['max_prefix_ratio'] = 0.0

                if self.global_steps == 1:
                    print(self.tokenizer.decode(gen_batch.batch["input_ids"][0], skip_special_tokens=True))

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    # dataframe_uids = batch.non_tensor_batch['dataframe_uid']
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # log avg prefix ratio
                    if 'prefix_ratios' in gen_batch_output.meta_info.keys():
                        metrics['batch/avg_prefix_ratio'] = float(np.mean(gen_batch_output.meta_info['prefix_ratios']))
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    
                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward
                        batch.batch['token_level_scores'] = reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        dataframe_uids = batch.non_tensor_batch['dataframe_uid']
                        # uid-> dataframe_uid
                        uid2dfuid = {uid: df_uid for uid, df_uid in zip(uids, dataframe_uids)}
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        # only used in skip set and experience pool
                        for i, uid in enumerate(unique_uids):
                            # 这里的uid是batch内部的uuid
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                            # get the corresponding global dataframe_uid
                            df_uid = uid2dfuid[uid]
                            
                            if df_uid == CONSTANT_DATA_UID:
                                print("constant data.")
                                continue
                            
                            if (uid_rewards == fail_value).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                                # self.skip_uid_set.add(df_uid)
                            elif (uid_rewards == success_value).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1
                                self.train_dataset.skip_uid_set.add(df_uid)
                            elif (uid_rewards == format_value).all():
                                valid_mask[uid_mask] = False
                                solve_none_format += 1

                        if self.config.trainer.skip_valid_mask:
                            valid_mask[:] = True

                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)
                        # add on-policy metrics
                        prefix_mask = batch.batch['prefix_mask']
                        off_policy_mask = prefix_mask.any(-1)
                        on_policy_mask = ~off_policy_mask
                        metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                        metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)
                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            # compute the old_log_prob of the current policy (not replace)
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        # experience pool collection using extracted op
                        collect_experience_entries(self, batch, reward_tensor, success_value, metrics)

                        # now replace recorded_old_log_prob (if needed) using extracted op
                        replace_recorded_old_log_probs(self, batch, off_policy_mask, prefix_mask)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.train_collate_fn.is_mixed_policy:
                            batch.meta_info["loss_remove_clip"] = self.config.actor_rollout_ref.actor.loss_remove_clip
                            batch.meta_info["off_policy_reshape"] = self.config.actor_rollout_ref.actor.off_policy_reshape
                            print(f"loss_remove_clip: {batch.meta_info['loss_remove_clip']}, off_policy_reshape: {batch.meta_info['off_policy_reshape']}")
                            # control importance weighting usage from trainer config
                            batch.meta_info['exp_is_weighting'] = getattr(self.config.trainer, 'exp_is_weighting', False)
                        else:
                            batch.meta_info["loss_remove_clip"] = False
                            batch.meta_info["off_policy_reshape"] = "no_reshape"
                            batch.meta_info['exp_is_weighting'] = False

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                            if 'avg_score_stable' not in val_metrics:
                                val_metrics['avg_score_stable'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/') and "minerva" not in key])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # check if it's time to switch to mixed-policy
                if (not self.train_collate_fn.is_mixed_policy) and (metrics.get('batch/solved', 0) > self.config.trainer.experience_start_sratio):
                    print("[Policy Switch] Switching to mixed-policy training!")
                    self.train_collate_fn.set_mixed_policy(True)
                    self.train_collate_fn.actor_rollout_wg = self.actor_rollout_wg

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        print(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

                # cleanup experience pool and buckets using extracted op
                cleanup_experience_buckets(self)

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score_stable']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score_stable']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score_stable': best_score, 'global_step': self.global_steps})+'\n')

