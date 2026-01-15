# PACS Algorithm Core Function Descriptions

# FUNCTION: compute_reward

## Function Overview
Computes "reward" for policy improvement, measuring log probability change of current policy relative to reference policy.

## Function Signature
def compute_reward(
    old_log_prob,
    log_prob,
    response_mask,
    beta=1.0,
    reward_method="1",
):

## Input Parameters
- `old_log_prob`: Reference policy token-level log probabilities, shape `[batch_size, response_length]`
- `log_prob`: Current policy token-level log probabilities, shape `[batch_size, response_length]`
- `response_mask`: Response portion mask, shape `[batch_size, response_length]`
- `beta`: Reward scaling factor, controls aggressiveness of policy updates (default 1.0)
- `reward_method`: Reward computation method, '1' or '2'

## Detailed Description
Function computes reward values in reinforcement learning based on specified method.
Method 1 computes sum of new-old log probability differences, measuring policy improvement magnitude.
Method 2 computes normalized sum of current log probabilities.
If an invalid method number is specified, function raises an exception.

## Output
- `reward`: Policy improvement reward, shape `[batch_size, 1]`

## Function Implementation
code/src/pacs/pacs_core_algos.py:line 7-21

## Test Code
code/test_code/test_compute_reward.py

---

# FUNCTION: compute_weight

## Function Overview
Computes sample weights to handle class imbalance problem (uneven positive/negative sample ratio).

## Function Signature
def compute_weight(score, rollout_n, mode="question"):

## Input Parameters
- `score`: Verifiable rewards (labels), shape `[batch_size, 1]`
- `rollout_n`: Number of responses generated per prompt
- `mode`: Weight computation mode, 'question'/'only_positive'/'only_negative'

## Detailed Description
Function assigns training weights to samples based on different modes. In "question" mode, function groups samples by rollout_n size, using class-balanced weights for groups with label imbalance to make positive and negative samples contribute equally to training. In "only_positive" mode, only positive samples (label 1) are assigned weight 1.0, negative samples get weight 0. In "only_negative" mode, the opposite applies: only negative samples (label 0) are assigned weight 1.0. If an invalid mode is specified, function raises an exception.

## Output
- `weight`: Sample weights, shape `[batch_size, 1]`

## Function Implementation
code/src/pacs/pacs_core_algos.py:line 59-86

## Test Code
code/test_code/test_compute_weight.py

---

# FUNCTION: compute_pacs_loss

## Function Overview
Computes PACS's core loss function, using advantage function as logit and verifiable rewards as labels with binary cross-entropy loss.

## Function Signature
def compute_pacs_loss(
    prompts,
    old_log_prob,
    log_prob,
    token_level_scores,
    response_mask,
    rollout_n,
    algorithm_config=None,
):

## Input Parameters
- `prompts`: Input prompts, shape `[batch_size, prompt_length]`
- `old_log_prob`: Reference policy log probabilities, shape `[batch_size, response_length]`
- `log_prob`: Current policy log probabilities, shape `[batch_size, response_length]`
- `token_level_scores`: Verifiable rewards (labels), shape `[batch_size, response_length]`
- `response_mask`: Response mask, shape `[batch_size, response_length]`
- `rollout_n`: Number of responses generated per prompt
- `algorithm_config`: Algorithm configuration object containing `beta`, `reward_method`, `adv_estimator`, `use_weight`, `weight_mode`, etc.

## Detailed Description

Function needs to verify that every rollout_n samples correspond to the same prompt, then compute reward values and advantage values (based on specified advantage estimator). Finally, function sums token-level scores and uses binary cross-entropy loss with advantage values as supervision signal and scores as predicted values to compute loss. If sample weighting is enabled, balanced weights are computed based on score labels to handle sample imbalance.

## Output
- `pacs_loss`: PACS loss value, scalar

## Function Implementation
code/src/pacs/pacs_core_algos.py:line 89-138

## Test Code
code/test_code/test_compute_pacs_loss.py

