# PRIME Algorithm Core Function Descriptions

# FUNCTION: DataParallelPRIMERewardModel._forward_micro_batch

## Function Overview
Computes token-level reward scores for implicit process reward model

## Function Signature
def _forward_micro_batch(self, micro_batch, prompt_length):

## Input Parameters
- `micro_batch`: Micro-batch data {input_ids, attention_mask, position_ids, acc}
- `prompt_length`: Prompt sequence length

## Detailed Description
First selects computation path based on whether padding removal optimization is enabled. For reward model and reference model, performs forward pass to compute output logits, then extracts log probabilities for corresponding tokens.
Implicit rewards are obtained by computing log probability differences between the two models on response sequences.
Process reward generation supports two modes: direct mode and GAE mode.
In token granularity, each position receives corresponding reward value; in whole granularity, all rewards accumulate to the last valid token position.

## Output
- `token_level_scores`: Token-level reward scores (torch.Tensor, shape: [batch_size, seq_len])
- `q_values`: Raw log probability differences (torch.Tensor, shape: [batch_size, seq_len])

## Function Implementation
code/recipe/prime/prime_dp_rm.py:line 51-228

## Test Code
code/test_code/test_forward_micro_batch.py

---

# FUNCTION: compute_ce_dpo_loss_rm

## Function Overview
Computes cross-entropy style DPO loss

## Function Signature
def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):

## Input Parameters
- `token_level_scores`: Token-level rewards
- `acc`: Binary accuracy labels
- `response_mask`: Response mask  
- `beta`: Temperature parameter

## Detailed Description
Computes sequence-level scores, converts them to probabilities, and finally computes cross-entropy loss.

## Output
Scalar loss value

## Function Implementation
code/recipe/prime/prime_core_algos.py:line 82-85

## Test Code
code/test_code/test_ce_dpo_loss.py

---

# FUNCTION: compute_detach_dpo_loss_rm

## Function Overview
Computes detached DPO loss, supporting BoN (Best-of-N) mode

## Function Signature
def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):

## Input Parameters
- Base: `token_level_scores`, `acc`, `response_mask`, `beta`
- Contrast: `Q_bc` (batch Q-value matrix), `acc_bc` (batch accuracy matrix)  
- `bon_mode`: Weight mode {"none", "bon_rm", "bon_acc"}

## Detailed Description

Computes detached DPO loss based on contrastive samples, selecting samples with opposite accuracy labels within batch as contrastive baseline, and supports assigning higher training weights to high-quality samples through BoN weight mechanism.

BoN weights include three modes: none mode, bon_rm mode, and bon_acc mode.

## Output
Weighted DPO loss value

## Function Implementation
code/recipe/prime/prime_core_algos.py:line 88-116

## Test Code
code/test_code/test_detach_dpo_loss.py

---

# FUNCTION: compute_dpo_accuracy

## Function Overview
Computes pairwise comparison DPO accuracy

## Function Signature
def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):

## Input Parameters
- `token_level_scores`: Token-level reward scores
- `acc`: Accuracy labels
- `response_mask`: Response mask
- `n_samples`: Sample count per group

## Detailed Description
Uses pairwise comparison to evaluate reward model accuracy. Aggregates token-level scores to sequence-level scores, constructs sample pairs for comparison. Judges directional consistency by computing score differences and accuracy differences.
Uses weighted average to compute accuracy, with higher weights for sample pairs with larger accuracy differences. Returns 0.5 when all samples have the same accuracy.

## Output
Weighted pairwise comparison accuracy

## Function Implementation
code/recipe/prime/prime_core_algos.py:line 119-143

## Test Code
code/test_code/test_dpo_accuracy.py

---

# FUNCTION: compute_dpo_abs_accuracy

## Function Overview
Computes absolute accuracy metric

## Function Signature
def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):

## Input Parameters
- `token_level_scores`: Token-level reward scores
- `acc`: Accuracy labels
- `response_mask`: Response mask

## Detailed Description
Converts aggregated token-level scores to predicted labels, compares with true labels to compute absolute accuracy.

## Output
Absolute classification accuracy

## Function Implementation
code/recipe/prime/prime_core_algos.py:line 146-147

## Test Code
code/test_code/test_dpo_abs_accuracy.py

---

# FUNCTION: compute_rloo_advantage_return

## Function Overview
Implements RLOO advantage estimation, supporting multiple reward source fusion

## Function Signature
def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):

## Input Parameters
- `data`: Data protocol object containing DataProto.batch fields
  - `data.batch["rm_scores"]`: Implicit reward scores
  - `data.batch["acc"]`: Binary accuracy labels
  - `data.batch["prompts"]`: Input prompt sequences
  - `data.batch["attention_mask"]`: Attention mask
- `response_mask`: Response mask indicating valid token positions
- `n_samples`: Sample count per group, used for RLOO grouping computation
- `config`: Reward weight configuration, hierarchical configuration object
  - `config.algorithm.reward_dpo_coef`: DPO reward weight coefficient, default 5.0
  - `config.algorithm.reward_gt_coef`: Accuracy reward weight coefficient, default 5.0

## Detailed Description
Function supports advantage computation with multi-source reward fusion. Reward sources include reward model scores and accuracy rewards.
Groups multiple responses generated from the same prompt by n_samples intervals.
Uses RLOO to compute advantage function for each reward source and fuses multiple reward sources, computes cumulative return values after reward fusion and applies whitening.

## Output
- `advantages`: Whitened advantage function (torch.Tensor, shape: [batch_size, seq_len])
- `returns`: Cumulative return values (torch.Tensor, shape: [batch_size, seq_len])

## Function Implementation
code/recipe/prime/prime_core_algos.py:line 21-79

## Test Code
code/test_code/test_rloo_advantage_return.py

---

# FUNCTION: RayPRIMETrainer.filter_and_downsample

## Function Overview
Implements intelligent sample filtering and priority downsampling for PRIME algorithm

## Function Signature
def filter_and_downsample(self, scores, batch: DataProto):

## Input Parameters
- `scores`: Verification score list, binary accuracy labels
- `batch`: Data batch object
- `config`: Filtering parameter configuration

## Detailed Description
Filters high-quality samples from oversampled candidate responses. Reconstructs verification scores into 2D matrix, performs dual filtering: accuracy filtering and length filtering.
Accuracy filtering retains sample groups with average accuracy within configured range; length filtering removes sample groups exceeding maximum length threshold.
Sample groups meeting filtering conditions are retained with priority, ultimately retaining 1/oversample_factor proportion of samples. If all groups fail to meet filtering conditions, algorithm still retains corresponding proportion of samples to ensure training continues.

## Output
Filtered high-quality data batch

## Function Implementation
code/recipe/prime/prime_ray_trainer.py:line 540-572

## Test Code
code/test_code/test_filter_and_downsample.py

