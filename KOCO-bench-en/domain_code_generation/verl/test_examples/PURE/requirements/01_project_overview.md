# PURE Project Overview (Process-supervised Reinforcement Learning)


## 1. Algorithm Overview

PURE (Process-supervised RL) adopts a training paradigm that fuses "process rewards + verifiable rewards", using min-form credit assignment to address the "weak link masking" problem in chain-of-thought reasoning. During training, the policy model generates n responses for each prompt. The RewardManager computes outcome-level verifiable rewards (VR), while the PRM predicts token/step-level process rewards (PR). Subsequently, RLOO-based advantage estimation fuses VR and PR, and PPO is used to update the policy model.

Core idea: Apply temperature-weighted weighting w_t = softmax(-r_t/T) to the process reward sequence r_t and perform weighted summation. When T→0, this approximates strict min(r_t), highlighting the step with the largest error and greatest criticality, achieving "Stop Summation".

## 2. Training Pipeline (Pseudocode)

```
Algorithm: PURE Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, Process RM f_φ, RewardManager R
Config: rollout.n, credit_assignment ∈ {'gamma-decay','strict min-form', T>0},
        coef_vr, coef_pr, adv_estimator=rloo

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Generate n responses (vLLM/HF inference)
4:       responses ← rollout(π_θ, B, n)
5:       # Compute outcome-level VR and (optional) additional statistics
6:       VR ← R.compute(B, responses)
7:       # Compute PRM process rewards (token/step level)
8:       PR_token ← f_φ.forward(B, responses)
9:       # Approximate min-form weights (or strict/γ decay)
10:      if credit_assignment == T > 0 then
11:          w ← softmax(-PR_token/T, dim=-1)
12:          PR_token ← PR_token ⊙ w
13:      elif credit_assignment == 'strict min-form' then
14:          PR_token ← strictly_select_min(PR_token)
15:      else  # 'gamma-decay'
16:          PR_token ← apply_gamma_decay(PR_token, γ)
17:
18:      # Fuse rewards and compute RLOO advantage
19:      A ← RLOO(PR_token, VR, grouping_by_prompt, coef_pr, coef_vr)
20:      A ← whiten(A)  # optional
21:
22:      # PPO update
23:      θ ← θ + Adam(∇_θ L_PPO(π_θ; A))
24:   end for
25: end for
```

Key points:
- Grouping and RLOO: Group n responses from the same prompt together, compute leave-one-out baseline to reduce variance;
- Reward fusion: `coef_vr` and `coef_pr` control the relative weights of VR and PR;
- Min-form: Temperature-based approximation is more stable by default, with strict min-form as the limiting case;
- Curriculum learning: Extreme samples with all 0s or all 1s can be removed at the batch level to improve training signal-to-noise ratio.



