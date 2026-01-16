# Copyright 2024 PRIME team and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import itertools

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

from .prime_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm

__all__ = ["DataParallelPRIMERewardModel"]


class DataParallelPRIMERewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)


    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip
            )
        self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.to(get_device_name())
                attention_mask = data["attention_mask"]
                acc = data["acc"]

                prompt_ids = data["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "dpo":
                    # the implementation of dpo is actually detached, which means we have to know the average
                    # value of w/l reward before the update.
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q, acc, Q_bc=data["Q_bc"], acc_bc=data["acc_bc"], response_mask=response_mask, beta=beta
                    )
                elif self.config.model.loss_type == "bon_acc":
                    # change the original distribution of each sample to BoN distribution, then update reward model
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )
                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
