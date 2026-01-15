# PRIME Algorithm Pipeline

## 1. Algorithm Overview

PRIME (Process Reinforcement through Implicit rEwards) algorithm trains through the following steps: First, oversample each prompt to generate multiple candidate responses, then use rule-based verifiers to check response correctness and filter samples of medium difficulty. Next, compute the log probability difference between the implicit reward model and reference model as token-level reward signals. During model update phase, update reward model parameters according to configured strategy, use RLOO algorithm to fuse process rewards and outcome rewards to compute advantage function, and finally update policy model through PPO loss. The entire process achieves joint online training of policy model and reward model.

## 2. Training Pipeline

```
Algorithm: PRIME Training Pipeline
Input: Dataset D, Policy model π_θ, Reference model π_ref, Reward model π_φ
Parameters: oversample_factor=4.0, n=4, beta_train=0.05, reward_dpo_coef=5, reward_gt_coef=5
Output: Optimized policy model π_θ

1: Initialize π_φ ← π_θ, π_ref ← π_θ
2: for epoch = 1 to total_epochs do
3:    for batch B ∼ D do
4:        // Data Preparation and Response Generation
5:        B_expanded ← expand_batch(B, oversample_factor)
6:        responses ← generate_responses(π_θ, B_expanded, n)
7:        
8:        // Response Verification and Filtering
9:        scores ← verify_responses(responses)
10:       B_filtered ← filter_samples(B_expanded, responses, scores)
11:       
12:       // Log Probability Computation
13:       log_probs_old ← compute_log_prob(π_θ, B_filtered)
14:       log_probs_ref ← compute_log_prob(π_ref, B_filtered)
15:       
16:       // Reward Model Update and Score Computation
17:       if update == "before" then
18:           π_φ ← update_reward_model(π_φ, B_filtered)
19:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
20:       elif update == "after" then
21:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
22:           π_φ ← update_reward_model(π_φ, B_filtered)
23:       elif update == "reverse" then
24:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
25:           statistics ← compute_batch_statistics(rm_scores, scores)
26:           π_φ ← update_reward_model(π_φ, B_filtered, statistics)
27:       else
28:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
29:       end if
30:       
31:       // Advantage Computation
32:       advantages ← compute_rloo_advantage(rm_scores, scores, n, config)
33:       
34:       // Policy Update
35:       π_θ ← update_policy(π_θ, B_filtered, advantages)
36:    end for
37: end for
38: return π_θ
```

Key function descriptions in pipeline:
- `generate_responses(π, batch, n)`: Use policy model to generate n candidate responses for each prompt
- `verify_responses(responses)`: Use rule-based verifier to check response correctness, return binary labels
- `filter_samples(batch, responses, scores)`: Filter samples based on accuracy and length, then downsample
- `compute_rloo_advantage(rm_scores, scores, n, config)`: Fuse process rewards and outcome rewards to compute RLOO advantage
- `compute_reward_scores(π_φ, batch)`: Compute implicit reward model scores
- `update_reward_model(π_φ, batch)`: Update reward model parameters using cross-entropy loss
- `update_policy(π_θ, batch, advantages)`: Update policy model parameters using PPO loss


## 3. Application Scenarios

### Mathematical Reasoning Tasks
- Input: Mathematical problem text
- Output: Mathematical solution with reasoning steps
- Verification method: Exact match of LaTeX format answers
- Data sources: GSM8K, MATH, AMC, AIME and other math datasets

### Code Generation Tasks
- Input: Programming problem description
- Output: Executable program code
- Verification method: Test case pass rate calculation
- Data sources: APPS, CodeContests, TACO, LeetCode and other programming datasets

