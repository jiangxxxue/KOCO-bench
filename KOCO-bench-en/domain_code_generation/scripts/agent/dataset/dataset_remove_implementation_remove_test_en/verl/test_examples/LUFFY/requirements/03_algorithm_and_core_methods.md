# LUFFY Algorithm Core Function Descriptions

# FUNCTION: compute_grpo_outcome_advantage_split

## Function Overview
Computes group GRPO (Group Relative Policy Optimization) advantage estimation, with the key innovation of using only on-policy samples to compute within-group baseline.

## Function Signature
def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):

## Input Parameters
- `token_level_rewards`: Token-level reward tensor with shape `[batch_size, response_length]`, type `torch.Tensor`
- `eos_mask`: Valid token mask with shape `[batch_size, response_length]`, type `torch.Tensor`
- `index`: Sample grouping indices, n responses from the same prompt share the same index. **Note: This parameter should be a Python list or indexable sequence (e.g., `[0, 0, 1, 1]`), not `torch.Tensor`, as index values will be used as dictionary keys**
- `on_policy_mask`: Boolean mask with shape `[batch_size]`, type `torch.Tensor`, marking which samples are on-policy generated
- `epsilon`: Numerical stability constant, default 1e-6
- `use_std`: Whether to use standard deviation for normalization, default True

## Detailed Description
This function needs to sum token-level reward tensors along the time dimension for each response sequence to obtain sequence-level total scores.

The main implementation performs grouped normalization (including both on-policy and off-policy) for multiple responses from the same prompt, computing relative advantage values. During normalization, only samples marked as on-policy are selected to compute the group's mean and standard deviation. If a group has only one on-policy sample, set mean to 0 and standard deviation to 1; if there are multiple samples, compute statistics normally. For cases where standard deviation is 0, replace it with 1 to avoid division by zero errors.

If standard deviation normalization is enabled, compute (score - mean) / (std + epsilon); otherwise only subtract the mean. Finally, expand sequence-level advantage values to token-level advantage tensors.

## Output
- `advantages`: Token-level advantage tensor with shape `[batch_size, response_length]`
- `returns`: Token-level return tensor with shape `[batch_size, response_length]`, same as advantages

## Function Implementation
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 12-67

## Test Code
code/test_code/test_compute_grpo_outcome_advantage_split.py

---

# FUNCTION: compute_token_on_off_policy_loss

## Function Overview
Computes mixed policy loss, combining on-policy PPO clip loss and off-policy importance sampling loss, supporting multiple policy shaping methods to emphasize low-probability but critical actions.

## Function Signature
def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False, 
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):

## Input Parameters
- `old_log_prob`: Old policy log probabilities with shape `[batch_size, response_length]`, type `torch.Tensor`
- `log_prob`: Current policy log probabilities with shape `[batch_size, response_length]`, type `torch.Tensor`
- `advantages`: Advantage estimates with shape `[batch_size, response_length]`, type `torch.Tensor`
- `eos_mask`: Valid token mask with shape `[batch_size, response_length]`, type `torch.Tensor`
- `cliprange`: PPO clip range, default 0.2
- `clip_upper_bound`: PPO clip upper bound, default 100.0
- `prefix_mask`: Mask marking off-policy tokens with shape `[batch_size, response_length]`, **type must be `torch.Tensor` with dtype `torch.bool`, as the code uses bitwise NOT operator `~prefix_mask` to obtain on-policy token mask**
- `off_cliprange`: off-policy clip range (not used in current implementation)
- `off_normalize`: Whether to normalize off-policy ratio, default False
- `off_abs_cliprange`: off-policy absolute value clip range (not used in current implementation)
- `off_max_clip`: off-policy ratio upper limit, None means no limit
- `off_min_clip`: off-policy ratio lower limit, None means no limit
- `all_max_clip`: Global probability upper limit, None means no limit
- `off_policy_reshape`: off-policy policy shaping method name, default 'no_reshape'
- `off_policy_reshape_weight`: Policy shaping weight parameter, default 1.0
- `off_policy_reshape_pow_exp`: Power transformation exponent, default 0.5
- `on_policy_reshape`: on-policy policy shaping method name, default 'no_reshape'
- `on_policy_reshape_weight`: on-policy policy shaping weight, default 1.0
- `on_policy_reshape_pow_exp`: on-policy power transformation exponent, default 0.5
- `target_probs`: Optional target probabilities (for explicit behavior policy), default None
- `loss_remove_token_mean`: Whether to remove token-level averaging, default False
- `loss_remove_clip`: Whether to remove clip operation, default False

## Detailed Description
This function computes mixed policy gradient loss, with the core idea of simultaneously handling both on-policy and off-policy data. For on-policy data, it uses standard PPO algorithm to compute importance sampling ratio and perform clipping; for off-policy data, it directly uses current policy probabilities (or ratio relative to target probabilities) to compute loss, without relying on old policy.

The function distinguishes which tokens are off-policy and which are on-policy through prefix_mask, then combines the two losses with weighting. Additionally, the function should support multiple probability reshaping strategies and clipping mechanisms.

## Output
Returns dictionary containing the following key-value pairs:
- `pg_loss`: Total policy gradient loss (scalar)
- `off_pg_loss`: off-policy loss component (scalar)
- `on_pg_loss`: on-policy loss component (scalar)
- `off_pg_clipfrac`: off-policy clip fraction (scalar, currently 0 in implementation)
- `on_pg_clipfrac`: on-policy clip fraction (scalar)
- `ppo_kl`: PPO KL divergence (scalar)
- `off_policy_prob`: Average probability of off-policy samples (scalar)
- `on_policy_prob`: Average probability of on-policy samples (scalar)
- `off_ratio_mean`: off-policy average ratio (scalar)
- `off_ratio_max_clip_frac`: Fraction reaching upper clip limit (scalar)
- `off_ratio_min_clip_frac`: Fraction reaching lower clip limit (scalar)

## Function Implementation
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 69-251

## Test Code
code/test_code/test_compute_token_on_off_policy_loss.py

---

# FUNCTION: compute_sft_pure_loss

## Function Signature
def compute_sft_pure_loss(log_prob, eos_mask):

## Function Overview
Computes pure SFT (Supervised Fine-Tuning) loss, i.e., negative log likelihood loss, used for supervised learning on off-policy samples in multi-task learning.

## Input Parameters
- `log_prob`: Model-predicted log probabilities with shape `[batch_size, response_length]`
- `eos_mask`: Valid token mask with shape `[batch_size, response_length]`

## Detailed Description
Computes negative log probability for each token as loss value. Uses eos_mask to mask the loss, computing loss only at valid token positions. Averages the masked loss to obtain batch-level SFT loss.

## Output
- `sft_loss`: Scalar representing average SFT loss

## Function Implementation
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 7-10

## Test Code
code/test_code/test_compute_sft_pure_loss.py
