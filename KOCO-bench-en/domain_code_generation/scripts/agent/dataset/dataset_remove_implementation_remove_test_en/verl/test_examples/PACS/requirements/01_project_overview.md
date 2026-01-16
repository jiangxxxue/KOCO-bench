# PACS Project Overview (Implicit Actor Critic Coupling via a Supervised Learning Framework for RLVR)

## 1. Algorithm Overview

PACS (im**P**licit **A**ctor **C**ritic coupling via a **S**upervised learning framework) is a novel RLVR (Reinforcement Learning with Verifiable Rewards) framework that achieves implicit Actor-Critic coupling through a supervised learning paradigm. This method treats outcome rewards as predictable labels, reformulating the RLVR problem as a supervised learning task for a scoring function parameterized by the policy model and optimized using cross-entropy loss.

Core idea: Unlike traditional methods such as PPO or GRPO, PACS transforms the policy optimization problem into a binary classification problem. For each prompt, the model generates n responses, with each response receiving a binary label (0 or 1) based on verifiable rewards (such as correctness of mathematical problems). By computing the log probability change of the policy relative to a reference policy as "reward" and using methods like RLOO or GRPO to estimate the advantage function, PACS uses the advantage function as logit input and verifiable rewards as labels, training with binary cross-entropy loss (BCEWithLogitsLoss).

## 2. Training Pipeline (Pseudocode)

```
Algorithm: PACS Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n, beta, adv_estimator ∈ {'rloo','grpo','naive'},
        use_weight, weight_mode ∈ {'question','only_positive','only_negative'}

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Generate n responses (vLLM/HF inference)
4:       responses ← rollout(π_θ, B, n)
5:       
6:       # Compute verifiable rewards (VR)
7:       VR ← R.compute(B, responses)  # Binary labels: 0 or 1
8:       
9:       # Compute old policy log probabilities
10:      old_log_prob ← π_θ.compute_log_prob(B, responses)
11:      
12:      # Compute current policy log probabilities
13:      log_prob ← π_θ.forward(B, responses)
14:      
15:      # Compute "reward" (measure of policy improvement)
16:      if reward_method == '1' then
17:          reward ← beta × sum(log_prob - old_log_prob)
18:      elif reward_method == '2' then
19:          reward ← beta × sum(log_prob) / response_length
20:      
21:      # Compute advantage function
22:      if adv_estimator == 'rloo' then
23:          A ← RLOO(reward, grouping_by_prompt)
24:      elif adv_estimator == 'grpo' then
25:          A ← GRPO(reward, grouping_by_prompt)
26:      elif adv_estimator == 'naive' then
27:          A ← reward
28:      
29:      # Compute sample weights (optional)
30:      if use_weight then
31:          w ← compute_weight(VR, rollout_n, weight_mode)
32:      else
33:          w ← 1
34:      
35:      # Compute PACS loss (binary cross-entropy)
36:      L_PACS ← BCEWithLogitsLoss(A, VR, weight=w)
37:      
38:      # Update policy
39:      θ ← θ - Adam(∇_θ L_PACS)
40:   end for
41: end for
```

Key points:
- Supervised learning paradigm: Transform RLVR into supervised learning problem, with advantage function as logit and verifiable rewards as labels;
- Grouping and advantage estimation: Group n responses from the same prompt together, use RLOO/GRPO to compute advantages;
- Beta parameter: Controls aggressiveness of policy updates; larger beta leads to faster policy changes;
- Sample weighting: Optionally assign different weights to positive and negative samples to handle class imbalance;
- No Critic needed: PACS doesn't require a separate value network, simplifying the training process.

