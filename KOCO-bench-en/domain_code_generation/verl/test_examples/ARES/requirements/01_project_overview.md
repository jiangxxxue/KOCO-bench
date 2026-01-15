# ARES Project Overview (Adaptive Reasoning via Entropy Shaping)

## 1. Algorithm Overview

ARES (Adaptive Reasoning via Entropy Shaping) is a difficulty-aware adaptive multimodal reasoning framework designed to dynamically adjust model reasoning resource allocation based on input question difficulty. The method is based on a core observation: existing multimodal reasoning models tend to overthink on **easy** tasks (generating lengthy reasoning trajectories), while under-exploring on **hard** tasks (missing solutions due to insufficient search).

ARES introduces a mechanism based on **High Window Entropy (HWE) tokens** to detect moments of reasoning uncertainty and flexibly adjust exploration intensity. HWE tokens identify sustained high-uncertainty regions by averaging token-level entropy within a sliding window, marking critical reasoning nodes where the model needs more exploration.

ARES adopts a **two-stage training pipeline**:

1. **Adaptive Cold-Start Phase**: Constructs difficulty-aware multimodal and text reasoning samples where reasoning trajectory length matches task difficulty, enabling the model to learn difficulty awareness. This phase uses an adaptive KL-guided fine-tuning strategy to ensure robust initialization across different difficulty tasks.

2. **Adaptive Entropy Policy Optimization (AEPO)**: Uses HWE tokens as triggers to decide **when** to explore further, combined with **hierarchical entropy rewards** and **dynamic KL control** to determine **how much** to explore. Through online difficulty bucketing and entropy-aware rollout allocation, AEPO achieves uncertainty-aware, difficulty-adaptive reasoning depth adjustment.

Experimental results show that ARES achieves a better trade-off between reasoning **efficiency** and **accuracy**, outperforming baseline methods on multimodal, mathematical, and logical benchmarks while reducing inference costs and narrowing the gap with commercial systems.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: ARES Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref
Config: difficulty_thresholds = {acc_high: 2/3, acc_low: 1/3},
        alpha_entropy_dict = {"easy": α_e, "medium": α_m, "hard": α_h},
        target_entropy_token_dict = {"easy": T_e, "medium": T_m, "hard": T_h}

# Stage 1: Adaptive Cold-Start
1: for epoch in 1..E_coldstart do
2:   for batch B ~ D_coldstart do
3:       # Use difficulty-aware data where reasoning length matches difficulty
4:       responses ← rollout(π_θ, B)
5:       # Compute base rewards
6:       R_base ← RewardManager.compute(B, responses)
7:       # Adaptive KL-guided fine-tuning
8:       L_coldstart ← L_SFT + β_adaptive · KL(π_θ || π_ref)
9:       θ ← θ - ∇_θ L_coldstart
10:  end for
11: end for

# Stage 2: Adaptive Entropy Policy Optimization (AEPO)
12: Initialize difficulty_dict ← bootstrap_difficulty(D, π_θ)
13: for epoch in 1..E_RL do
14:   for batch B ~ D do
15:       # Online difficulty bucketing
16:       for prompt p in B do
17:           responses_p ← rollout(π_θ, p, n=n_rollout)
18:           acc_rate_p ← mean([verify(r) for r in responses_p])
19:           if acc_rate_p >= 2/3 then difficulty[p] ← "easy"
20:           elif acc_rate_p >= 1/3 then difficulty[p] ← "medium"
21:           else difficulty[p] ← "hard"
22:       end for
23:       
24:       # Entropy-aware rollout: trigger exploration in HWE regions
25:       for response r in responses do
26:           window_entropies ← compute_window_entropy(r, window_size=4)
27:           HWE_tokens ← identify_high_entropy_tokens(window_entropies, threshold=τ)
28:           # HWE tokens mark high-uncertainty regions requiring deeper exploration
29:       end for
30:       
31:       # Hierarchical entropy reward shaping
32:       for response r, difficulty d in zip(responses, difficulties) do
33:           α_d ← alpha_entropy_dict[d]
34:           T_d ← target_entropy_token_dict[d]
35|           N_HWE ← count(HWE_tokens in r)
36|           
37|           # Adaptively adjust entropy reward based on difficulty
38|           if d == "easy" then
39|               # Easy tasks: penalize over-exploration
40|               if verified(r) and N_HWE > T_d then
41|                   R_entropy ← -α_d · penalty(N_HWE - T_d)
42|               else
43|                   R_entropy ← 0
44|           elif d == "medium" then
45|               # Medium tasks: balance exploration
46|               if verified(r) then
47|                   R_entropy ← -α_d · deviation(N_HWE, T_d)
48|               else
49|                   R_entropy ← α_d · bonus(N_HWE)
50|           else  # hard
51|               # Hard tasks: encourage exploration
52|               if verified(r) then
53|                   R_entropy ← α_d · bonus(N_HWE)
54|               else
55|                   R_entropy ← α_d · bonus(N_HWE)
56|       end for
57|       
58|       # Dynamic KL control: relax KL constraints within HWE windows
59|       for response r in responses do
60:           KL_coef_token ← []
61:           for token t in r do
62:               if t is HWE_token then
63:                   # Relax KL constraints in high-entropy regions, allowing more exploration
64:                   KL_coef_token.append(β_low)
65:               else
66:                   KL_coef_token.append(β_normal)
67:           end for
68:       end for
69:       
70:       # Compute total reward: accuracy + entropy shaping
71:       R_total ← R_accuracy + R_entropy - KL_coef_token · KL(π_θ || π_ref)
72|       
73|       # Advantage estimation (GRPO/RLOO)
74|       A ← compute_advantage(R_total, grouping_by_prompt)
75|       A ← whiten(A)
76|       
77|       # PPO update
78|       θ ← θ + Adam(∇_θ L_PPO(π_θ; A))
79|       
80|       # Adaptively update difficulty targets and entropy coefficients
81|       update_alpha_entropy(difficulty_dict, α_entropy_dict, target_entropy_dict)
82|   end for
83| end for
```

Key points:
- **Difficulty bucketing**: Dynamically categorize questions into easy/medium/hard buckets based on rollout accuracy;
- **HWE detection**: Use sliding window to average token-level entropy, identifying sustained high-uncertainty regions as exploration branch points;
- **Hierarchical rewards**: Easy tasks penalize over-exploration, hard tasks encourage deep exploration, medium tasks balance both;
- **Dynamic KL**: Reduce KL penalty coefficient within HWE windows, allowing the model to deviate from reference policy in high-uncertainty regions;
- **Adaptive adjustment**: Dynamically adjust entropy coefficients and target token counts for each difficulty bucket through dual ascent mechanism, achieving difficulty-aware reasoning resource budget control.


