# LUFFY Project Overview (Learning to Reason Under Off-Policy Guidance)

## 1. Algorithm Overview

LUFFY (Learning to Reason Under Off-Policy Guidance) is an innovative reinforcement learning framework designed to bridge the gap between zero-RL and imitation learning by incorporating off-policy reasoning trajectories. Built on GRPO (Group Relative Policy Optimization), this method simultaneously leverages on-policy self-generated samples and off-policy external demonstration data during training.


LUFFY's training paradigm allows the model to learn from stronger teacher models while maintaining autonomous exploration capabilities, demonstrating excellent performance on tasks requiring complex reasoning chains such as mathematical reasoning.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: LUFFY Training Pipeline
Input: Dataset D (containing prompts and optional off-policy targets), 
       Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n (on-policy samples), 
        n_prefix (off-policy samples),
        prefix_ratio (off-policy guidance ratio),
        off_policy_reshape (policy shaping method),
        adv_estimator=grpo_split

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Mixed sampling: generate n responses (some using off-policy prefix)
4:       for each prompt in B do
5:           # Generate n_prefix responses with off-policy prefix
6:           prefix_ratio ← sample_prefix_ratio(strategy, global_step)
7:           off_policy_responses ← rollout_with_prefix(π_θ, prompt, 
8:                                    target_prefix, prefix_ratio, n_prefix)
9:           # Generate (n - n_prefix) pure on-policy responses
10:          on_policy_responses ← rollout(π_θ, prompt, n - n_prefix)
11:          responses ← concat(off_policy_responses, on_policy_responses)
12:      end for
13:
14:      # Compute verifiable rewards (VR)
15:      VR ← R.compute(B, responses)
16:
17:      # Rejection sampling: remove all-success or all-failure prompt groups
18:      valid_mask ← filter_extreme_groups(VR, responses)
19:      responses ← responses[valid_mask]
20:
21:      # Compute old policy log probabilities
22:      old_log_prob ← π_θ.compute_log_prob(responses)
23:
24:      # Compute reference policy log probabilities (for KL penalty)
25:      if use_reference_policy then
26:          ref_log_prob ← π_ref.compute_log_prob(responses)
27:          token_level_rewards ← VR - β * KL(old_log_prob, ref_log_prob)
28:      else
29:          token_level_rewards ← VR
30:      end if
31:
32:      # Group advantage estimation (GRPO with split)
33:      # Use only on-policy samples to compute baseline
34:      A ← compute_grpo_split_advantage(token_level_rewards, 
35:                                       on_policy_mask, group_by_prompt)
36:
37:      # Apply prefix reward weights
38:      A ← apply_prefix_weight(A, prefix_mask, α, β)
39:
40:      # Mixed policy loss computation (on-policy + off-policy)
41:      for micro_batch in split(responses, micro_batch_size) do
42:          # Compute current policy log probabilities
43:          log_prob ← π_θ.forward(micro_batch)
44:          
45:          # On-policy loss (standard PPO clip)
46:          ratio_on ← exp(log_prob - old_log_prob)
47:          if on_policy_reshape != 'no_reshape' then
48:              ratio_on ← reshape_ratio(ratio_on, method=on_policy_reshape)
49:          end if
50:          L_on ← -A * clip(ratio_on, 1-ε, 1+ε)
51:          
52:          # Off-policy loss (importance sampling + policy shaping)
53:          ratio_off ← exp(log_prob)  # relative to implicit behavior policy
54:          if off_policy_reshape != 'no_reshape' then
55:              ratio_off ← reshape_ratio(ratio_off, method=off_policy_reshape)
56:          end if
57:          ratio_off ← clip(ratio_off, min_clip, max_clip)
58:          L_off ← -A * ratio_off
59:          
60:          # Mixed loss
61:          L_policy ← prefix_mask * L_off + (1 - prefix_mask) * L_on
62:          
63:          # Entropy regularization
64:          L_entropy ← -entropy_coeff * H(π_θ)
65:          
66:          # Total loss
67:          L ← mean(L_policy) + L_entropy
68:      end for
69:
70:      # Update policy
71:      θ ← θ - Adam(∇_θ L)
72:   end for
73: end for
```

## 3. Key Features

### 3.1 Off-Policy Prefix Strategy
- **Dynamic prefix ratio**: Supports multiple strategies (random, linear, fixed) to control how much off-policy prefix to use
- **Curriculum learning**: Through linear strategy, use more off-policy guidance in early training, gradually increasing autonomous exploration later
- **Flexible configuration**: Control how many prefix samples to generate per prompt (n_prefix)

### 3.2 Policy Shaping Methods
Supports multiple ratio transformation methods to emphasize important actions:
- `no_reshape`: Standard importance sampling
- `logp`: Use log probability instead of probability ratio
- `p_logp`: Weighted combination of probability ratio and log probability
- `square_root`: Square root of probability ratio
- `pow`: Power transformation of probability ratio
- `p_div_p_α`: Probability normalization transformation p/(p+α)

### 3.3 Group Advantage Estimation
- **GRPO Split**: Improved GRPO that uses only on-policy samples to compute within-group baseline
- **Variance reduction**: Reduce variance of advantage estimation through within-group normalization
- **Fair comparison**: Ensure off-policy samples don't affect baseline computation for on-policy samples

### 3.4 Rejection Sampling and Curriculum Learning
- Automatically filter prompt groups with all successes or all failures
- Improve training signal-to-noise ratio, avoid uninformative samples
- Support reward-based sample filtering strategies

