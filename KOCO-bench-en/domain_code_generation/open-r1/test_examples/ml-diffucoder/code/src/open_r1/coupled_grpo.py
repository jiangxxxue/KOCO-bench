#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
)
from trl.trainer.grpo_trainer import nanmin, nanmax
import wandb
import random
if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.
    Does NOT split fields with list type like 'mask_seeds' (copies them as-is).
    """
    first = next(v for v in tensor_dict.values() if isinstance(v, torch.Tensor))
    chunk_size = first.shape[0] // num_chunks

    return [
        {
            key: (
                val[i * chunk_size : (i + 1) * chunk_size]
                if isinstance(val, torch.Tensor)
                else val
            )
            for key, val in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]

def selective_log_softmax(logits, index, weights=None, mask=None):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation with weighted probabilities.
    
    This function handles three versions of probabilities for each sequence:
    1. p0: Original sequence probability
    2. p1: Masked sequence probability (when mask is True)
    3. p2: Reverse masked sequence probability (when mask is False)
    
    The final probability is computed as: (p0 + weighted_sum(p1, p2)) / 2
    where weighted_sum uses weights[1] for p1 and weights[2] for p2.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `[num_iterations * 3 * batch_size, seq_len, vocab_size]`.
        index (`torch.Tensor`):
            Index tensor of shape `[num_iterations * batch_size, seq_len]`.
        weights (`torch.Tensor`, optional):
            Weights tensor of shape `[num_iterations * 3]` for weighting different versions.
        mask (`torch.Tensor`, optional):
            Mask tensor of shape `[num_iterations * batch_size, seq_len]` indicating which tokens are masked.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with shape `[num_iterations * batch_size, seq_len]`.
    """
    # Process sequences in chunks to reduce memory usage
    full_batch_size = logits.size(0) // 3  # Each sequence has 3 versions, visit num_iterations * batch_size blocks
    num_iterations = weights.size(0) // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []
    
    # Process each sequence's three versions together
    for i in range(full_batch_size):
        # Get the three versions for this sequence
        seq_labels = index[i]    # [seq_len]

        chunk, offset = divmod(i, batch_size)
        base = chunk * 3 * batch_size
        logits_index = torch.tensor([base + 0*batch_size + offset, base + 1*batch_size + offset, base + 2*batch_size + offset], device=logits.device)
        
        seq_logits = logits[logits_index]  # [3, seq_len, vocab_size]

        # Compute log probabilities for all three versions
        seq_logps = F.log_softmax(seq_logits, dim=-1)
        seq_per_token_logps = seq_logps.gather(dim=-1, index=seq_labels.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1)).squeeze(-1)  # [3, seq_len]
        # import pdb; pdb.set_trace();
        if weights is not None and mask is not None:
            # Get weights and mask for this sequence
            # Use modulo to get the correct weights for each sequence
            # import pdb; pdb.set_trace();
            weight_idx = i // batch_size
            seq_weights = weights[weight_idx*3:(weight_idx+1)*3]  # [3]
            seq_mask = mask[i]  # [seq_len]
            
            # Weight the masked and unmasked versions
            weighted_logps = torch.where(
                seq_mask,
                seq_per_token_logps[1] * seq_weights[1],  # p1 * t1
                seq_per_token_logps[2] * seq_weights[2]   # p2 * t2
            )
            
            # Combine with original probability and average
            final_logps = (seq_per_token_logps[0] + weighted_logps) / 2
        else:
            final_logps = seq_per_token_logps[0]  # Just use original probability if no weights/mask
        # if i == 9:
        #     import pdb; pdb.set_trace();
        per_token_logps.append(final_logps)

    # import pdb; pdb.set_trace();
    
    return torch.stack(per_token_logps)

class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        # import pdb; pdb.set_trace();

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][:, this_itr_idx, :].unsqueeze(1)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][:, this_itr_idx, :].unsqueeze(1)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # import pdb; pdb.set_trace();
        per_token_loss1 = coef_1 * advantages.view(-1, 1, 1)
        per_token_loss2 = coef_2 * advantages.view(-1, 1, 1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        # import pdb; pdb.set_trace();

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

    def forward_process(self, batch, prompt_index, mask_id, seed=None, accumulate=False):
        set_seed(seed)
        b, l = batch.shape
        noisy_batch = []
        # generate a random number between 0.1 and 0.9
        mask_ratio = random.uniform(0.2, 0.8)
        t_p = torch.ones(b, device=batch.device) * mask_ratio
        # Create a random matrix to decide whether each prompt token is masked
        if accumulate:
            random_matrix = torch.rand((b//self.args.gradient_accumulation_steps, l), device=batch.device)
            random_matrix = torch.cat([random_matrix] * self.args.gradient_accumulation_steps, dim=0)
        else:
            random_matrix = torch.rand((b, l), device=batch.device)

        # 1. always mask completion tokens
        is_mask = ~prompt_index
        noisy_batch.append(torch.where(is_mask, mask_id, batch))
        # 2. mask completion tokens with probability t_p
        is_mask = ~prompt_index & (random_matrix < t_p.unsqueeze(1))
        completion_mask = is_mask
        noisy_batch.append(torch.where(is_mask, mask_id, batch))
        # 3. mask completion tokens reversely
        is_mask = ~prompt_index & (random_matrix > t_p.unsqueeze(1))
        noisy_batch.append(torch.where(is_mask, mask_id, batch))

        return noisy_batch, [1, 1/mask_ratio, 1/(1-mask_ratio)], completion_mask
    
    def get_logits(self, model, batch):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch).logits
            # since bos always unmask, the first logits will not be used
            # logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
        return logits[:, :-1]


    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        # Validate input dimensions
        if input_ids.dim() != 3:
            raise ValueError(f"Expected input_ids to have 3 dimensions, got {input_ids.dim()}")
        
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Ensure logits_to_keep is valid
        logits_to_keep = min(logits_to_keep, seq_len)
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length
        if len(mask_seeds) != num_iterations:
            raise ValueError(f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}")

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_weighted = []
        all_expanded_inputs = []
        all_completion_masks = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, t_weights, completion_mask = self.forward_process(
                expanded_input, prompt_index, self.processing_class.mask_token_id, seed=mask_seed, accumulate=num_iterations > 1
            )
            all_perturbed_seqs.extend(perturbed_seq)
            all_weighted.extend(t_weights) # [num_iterations * 3] list
            all_expanded_inputs.append(expanded_input)
            all_completion_masks.append(completion_mask)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * 3 * batch_size, seq_len]
        completion_mask_seq = torch.cat(all_completion_masks, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]
        all_weights_t = torch.tensor(all_weighted, device=device) # [num_iterations * 3]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq
        )  # [num_iterations * 3 * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * 3 * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        completion_loss_mask = completion_mask_seq[
            :, -logits_to_keep:
        ]  # [num_iterations * 3 * batch_size, logits_to_keep]
        # import pdb; pdb.set_trace();
        # Compute log probabilities using selective_log_softmax
        per_token_logps = selective_log_softmax(
            completion_logits, 
            completion_targets, 
            all_weights_t, 
            completion_loss_mask
        ).view(num_iterations, batch_size, logits_to_keep).permute(1, 0, 2)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = split_tensor_dict(
                    accumulated_local_batch, self.args.gradient_accumulation_steps
                )
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        steps = self.args.diffusion_steps
        temperature = self.args.generation_temperature

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                batch_prompt_completion_ids = self.model.diffusion_generate(
                    batch_prompt_ids,
                    attention_mask=batch_prompt_mask,
                    max_new_tokens=gen_length,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=temperature,
                    top_p=0.95 if temperature > 0 else 1.0,
                    alg="entropy",
                    alg_temp=0.0,
                    mask_token_id=self.processing_class.mask_token_id
                )
                # import pdb; pdb.set_trace();
                prompt_completion_ids_all.append(batch_prompt_completion_ids.sequences)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        eos_token = '<|im_end|>'
        eos_token_id = self.processing_class.encode(eos_token)[0]
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device).tolist()
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        with torch.no_grad():
            if self.num_iterations > 1:
                # repeat prompt completion ids self.num_iterations times
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                )
                all_old_per_token_logps = old_per_token_logps
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                if self.ref_model is not None:
                    print("Using reference model")
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, mask_seeds
                        )
                all_ref_per_token_logps = ref_per_token_logps

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )

                # Clip rewards to valid range
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        leave_one_out = True
        if not leave_one_out:
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)
        else:
            rewards_grouped = rewards.view(-1, self.num_generations)           # (batch, k)
            sum_group = rewards_grouped.sum(dim=1, keepdim=True)               # (batch, 1)
            baseline  = (sum_group - rewards_grouped) / (self.num_generations - 1)

            advantages = (rewards_grouped - baseline).view(-1)                 # (batch*k,)
            std_grouped_rewards = rewards_grouped.std(dim=1, keepdim=True)         # (batch, 1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=1
            ).view(-1)   
            if self.scale_rewards:
                advantages = advantages / (std_grouped_rewards + 1e-4)


        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }
