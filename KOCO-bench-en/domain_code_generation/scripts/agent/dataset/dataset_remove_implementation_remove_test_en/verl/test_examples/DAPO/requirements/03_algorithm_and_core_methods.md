# DAPO Algorithm Core Function Descriptions


# FUNCTION: agg_loss

## Function Overview

Aggregates loss matrix to scalar, supporting multiple aggregation modes. DAPO recommends using `token-mean` mode, making each token's contribution to loss equal, avoiding long sequences dominating gradients.

## Function Signature
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):

## Input Parameters

- `loss_mat`: Loss matrix with shape `(batch_size, response_length)`
- `loss_mask`: Mask tensor with shape `(batch_size, response_length)` (response part is 1, padding is 0)
- `loss_agg_mode`: String specifying aggregation mode
  - `"token-mean"`: Average over all tokens (**DAPO recommended**)
  - `"seq-mean-token-sum"`: Sum tokens within sequences first, then average across sequences
  - `"seq-mean-token-mean"`: Average tokens within sequences first, then average across sequences
  - `"seq-mean-token-sum-norm"`: Sum sequences then normalize by sequence length

## Detailed Description

The core function is to aggregate a 2D loss matrix into a single scalar value, supporting the following four aggregation strategies: token-mean, seq-mean-token-sum, seq-mean-token-mean, seq-mean-token-sum-norm.

## Output

- Scalar loss value for backpropagation

## Function Implementation
code/verl/trainer/ppo/core_algos.py:line 704-737


## Test Code
code/test_code/test_agg_loss.py

---

# FUNCTION: compute_policy_loss_vanilla

## Function Overview

Computes PPO policy gradient loss, supporting decoupled clipping (DAPO's core innovation). Through `clip_ratio_low` and `clip_ratio_high` parameters, allows asymmetric clipping for positive and negative advantages, implementing the "Clip-Higher" mechanism.

## Function Signature
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

## Input Parameters

- `old_log_prob`: Old policy log probabilities with shape `(batch_size, response_length)`
- `log_prob`: Current policy log probabilities with shape `(batch_size, response_length)`
- `advantages`: Advantage estimates with shape `(batch_size, response_length)`
- `response_mask`: Response mask with shape `(batch_size, response_length)`
- `loss_agg_mode`: Loss aggregation mode (default is token-mean)
- `config`: Actor configuration object, containing:
  - `clip_ratio`: Default clipping parameter (backward compatibility)
  - `clip_ratio_low`: Lower bound clipping parameter (**DAPO new**)
  - `clip_ratio_high`: Upper bound clipping parameter (**DAPO new**)
  - `clip_ratio_c`: Dual clipping lower bound (handles extreme negative advantages, default 3.0)
- `rollout_log_probs`: Optional, Rollout policy log probabilities (for off-policy training)

## Detailed Description

The function implements enhanced PPO policy loss computation, including standard PPO clipped loss calculation, dual-clip PPO, and truncated importance sampling (optional), ultimately calling related functions to implement loss aggregation.

Standard PPO clipping mechanism supports asymmetric clipping, requiring computation of unclipped original policy gradient loss and asymmetrically clipped loss, taking the maximum of both as pessimistic update strategy. When advantage value is negative, apply dual clipping to more conservatively reduce the probability of that action. If truncated importance sampling is enabled and rollout policy log probabilities are provided, the function computes the importance ratio of old policy relative to rollout policy, clips it to the configured upper limit, then multiplies it into the policy gradient loss.

## Output

Returns four scalar tensors:
- `pg_loss`: Policy gradient loss
- `pg_clipfrac`: Clipping fraction (proportion of clipped tokens, for monitoring training dynamics)
- `ppo_kl`: Approximate KL divergence
- `pg_clipfrac_lower`: Lower bound clipping fraction (proportion of tokens where dual clipping takes effect)

## Function Implementation

code/verl/trainer/ppo/core_algos.py:line 816-902

## Test Code
code/test_code/test_compute_policy_loss_vanilla.py



---

# FUNCTION: DAPORewardManager.__call__

## Function Overview

DAPO's reward manager, responsible for computing task rewards and applying overlong penalty. This is one of the key components distinguishing DAPO from other methods, preventing the model from generating meaningless lengthy outputs through a linear penalty mechanism.

## Function Signature
def __call__(self, data: DataProto, return_dict: bool = False):

## Input Parameters

- `data`: DataProto object containing:
  - `batch["prompts"]`: Prompt token IDs with shape `(bs, prompt_len)`
  - `batch["responses"]`: Response token IDs with shape `(bs, response_len)`
  - `batch["attention_mask"]`: Attention mask with shape `(bs, total_len)`
  - `non_tensor_batch["reward_model"]["ground_truth"]`: Ground truth answers
  - `non_tensor_batch["data_source"]`: Data source (e.g., "gsm8k", "math")
- `return_dict`: Boolean, whether to return dictionary format (including additional information like accuracy, overlong penalty, etc.)

## Detailed Description

The function is responsible for computing reward scores for generated responses. First checks if there are pre-computed rewards; if so, returns directly; otherwise processes each response individually, decoding token sequences into text and computing task rewards based on ground truth answers (needs to call related functional functions). Computed scores are placed at the last valid token position in the response sequence. If length control is enabled, the function applies additional negative reward penalty for overlong responses (linear penalty mechanism). Throughout the process, detailed information for some samples is output for debugging verification, ultimately returning complete reward tensor or dictionary containing additional evaluation metrics.


## Output

- If `return_dict=False`: Returns sparse reward tensor with shape `(batch_size, response_length)` (only last valid token position is non-zero)
- If `return_dict=True`: Returns dictionary containing:
  - `"reward_tensor"`: Reward tensor as above
  - `"reward_extra_info"`: Dictionary containing `"acc"` (accuracy list), `"overlong_reward"` (overlong penalty list), `"overlong"` (boolean list indicating if overlong), etc.

## Function Implementation
code/verl/workers/reward_manager/dapo.py:line 53-149


## Test Code
code/test_code/test_dapo_reward_manager.py



