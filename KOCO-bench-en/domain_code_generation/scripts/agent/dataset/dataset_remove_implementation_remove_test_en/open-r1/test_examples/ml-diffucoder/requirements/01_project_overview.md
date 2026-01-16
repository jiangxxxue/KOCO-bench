# DiffuCoder Project Overview (Masked Diffusion Models for Code Generation)

## 1. Algorithm Overview

DiffuCoder is a code generation system based on Masked Diffusion Models, adopting the "Coupled-GRPO" post-training method to improve code generation performance. Unlike traditional autoregressive (AR) models that strictly generate from left to right, diffusion language models (dLLM) generate text through an iterative denoising process, providing faster generation speed while maintaining performance.

Core Idea: DiffuCoder is based on the Masked Denoising Model (MDM) architecture, gradually adding noise (masking) to complete sequences through a forward diffusion process, then iteratively recovering the original sequence through a reverse denoising process. In the post-training phase, Coupled-GRPO solves the inefficiency problem in diffusion models where single-step loss calculation only occurs at masked positions through a paired sampling scheme, ensuring each token is calculated in at least one forward pass, thereby reducing variance and improving training efficiency.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: Coupled-GRPO Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, Reward Functions R
Config: num_generations k, num_iterations m, λ pairs, 
        diffusion_steps T, mask_ratio ∈ [0.2, 0.8]

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Generate k responses (diffusion generation)
4:       responses ← diffusion_generate(π_θ, B, k, T)
5:       
6:       # Calculate rewards (code execution + format checking)
7:       rewards ← R.compute(B, responses)
8:       
9:       # Calculate leave-one-out advantage
10:      for each prompt p in B do
11:          baseline_p ← (sum(rewards_p) - rewards_p[i]) / (k-1)
12:          advantages_p[i] ← rewards_p[i] - baseline_p
13:      end for
14:      
15:      # Coupled sampling: generate paired masks for each iteration
16:      for iteration i in 1..m do
17:          if random_masking then
18:              seed_i ← random_seed()
19:          else
20:              seed_i ← fixed_seed
21:          
22:          # Generate three versions of masked sequences
23:          mask_ratio ← uniform(0.2, 0.8)
24:          
25:          # Version 1: Fully mask all completion tokens
26:          seq_1 ← mask_all_completion_tokens(responses)
27:          
28:          # Version 2: Randomly mask by mask_ratio
29:          seq_2 ← random_mask(responses, mask_ratio, seed_i)
30:          
31:          # Version 3: Reverse mask (mask unmasked parts in Version 2)
32:          seq_3 ← reverse_mask(responses, mask_ratio, seed_i)
33:          
34:          # Calculate weighted log probabilities
35:          logits_1, logits_2, logits_3 ← π_θ.forward([seq_1, seq_2, seq_3])
36:          
37:          # For each token, weight based on its mask status in seq_2 or seq_3
38:          weights ← [1, 1/mask_ratio, 1/(1-mask_ratio)]
39:          per_token_logps ← selective_log_softmax(
40:              [logits_1, logits_2, logits_3], 
41:              responses, 
42:              weights, 
43:              mask_indicator
44:          )
45:          
46:          # Calculate KL divergence (if using reference model)
47:          if β > 0 then
48:              ref_logps ← π_ref.forward([seq_1, seq_2, seq_3])
49:              kl_penalty ← KL(per_token_logps, ref_logps)
50:          
51:          # PPO-style clipped loss
52:          ratio ← exp(per_token_logps - old_logps)
53:          clipped_ratio ← clip(ratio, 1-ε, 1+ε)
54:          loss ← -min(ratio * advantages, clipped_ratio * advantages)
55:          if β > 0 then
56:              loss ← loss + β * kl_penalty
57:          
58:          # Update policy
59:          θ ← θ - Adam(∇_θ loss)
60:      end for
61:   end for
62: end for
```

Key Points:
- **Coupled Sampling**: Ensures each token is calculated in at least one forward pass through paired masks (complementary masks), reducing variance;
- **Weighted Probability**: Weight log probabilities of different versions based on mask ratio, balancing contributions of different mask patterns;
- **Leave-one-out Baseline**: Group k responses by same prompt, calculate leave-one-out baseline to reduce variance;
- **Random Masking**: Use random mask seeds in multiple iterations to enhance training robustness;
- **Diffusion Generation**: Use iterative denoising process to generate code, supporting non-strictly left-to-right generation patterns.
