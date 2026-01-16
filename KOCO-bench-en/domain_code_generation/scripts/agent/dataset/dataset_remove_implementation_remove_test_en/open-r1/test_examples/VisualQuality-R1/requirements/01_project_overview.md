# VisualQuality-R1 Project Overview

## 1. Algorithm Overview

VisualQuality-R1 is the first No-Reference Image Quality Assessment (NR-IQA) model enhanced through Reinforcement Learning to Rank (RL2R), capable of simultaneously performing quality description and scoring reasoning. This method is based on the open-r1 framework, adopting the Group Relative Policy Optimization (GRPO) algorithm, and implements reasoning-based image quality assessment through multimodal vision-language models.

Core Idea: Transform the image quality assessment task into a ranking learning problem, optimizing the model's relative ranking ability for images of different quality through reinforcement learning. The model generates G quality scoring responses for each image, uses pairwise comparison Fidelity Reward to measure the consistency between the model's predicted ranking relationship and ground truth quality labels, and updates the policy model through the GRPO algorithm.

During training, the policy model is based on the Qwen2.5-VL-7B-Instruct multimodal architecture, generating structured responses containing reasoning process (thinking) and final score (answer) for input images. By calculating ranking fidelity between all image pairs within a batch, the model learns more accurate quality perception capabilities.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: VisualQuality-R1 Training Pipeline
Input: Dataset D (images with MOS scores), Policy π_θ, Reference π_ref
Config: num_generations G, beta β, epsilon ε, num_iterations μ

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Generate G quality scoring responses (including reasoning process and score)
4:       responses ← rollout(π_θ, B, G)
5:       
6:       # Extract predicted score from each response
7:       predictions ← extract_scores(responses)
8:       
9:       # Calculate batch statistics (mean and variance)
10:      for each sample i in B do
11:          μ_i ← mean(predictions_i)
12:          σ²_i ← variance(predictions_i)
13:      end for
14:      
15:      # Calculate fidelity reward (based on pairwise ranking)
16:      rewards ← []
17:      for each sample i in B do
18:          for each generation j in 1..G do
19:              r_ij ← 0
20:              for each other sample k ≠ i in B do
21:                  # Calculate ranking fidelity
22:                  Δ ← (pred_ij - μ_k) / √(σ²_i + σ²_k + ε)
23:                  p ← Φ(Δ)  # Standard normal cumulative distribution function
24:                  
25:                  # Determine ground truth ranking relationship based on true MOS
26:                  if MOS_i > MOS_k then
27:                      gt ← 1.0
28:                  elif MOS_i < MOS_k then
29:                      gt ← 0.0
30:                  else
31:                      gt ← 0.5
32:                  end if
33:                  
34:                  # Fidelity reward
35:                  r_ij += √(p·gt + ε) + √((1-p)·(1-gt) + ε)
36:              end for
37:              r_ij ← r_ij / (|B| - 1)
38:              rewards.append(r_ij)
39:          end for
40:      end for
41:      
42:      # Calculate format reward (optional)
43:      format_rewards ← check_format(responses)  # Check <think></think> and <answer></answer> format
44:      
45:      # Fuse rewards
46:      total_rewards ← rewards + format_rewards
47:      
48:      # Calculate group-wise advantage
49:      for each group g (same prompt) do
50:          μ_g ← mean(total_rewards_g)
51:          σ_g ← std(total_rewards_g)
52:      end for
53:      advantages ← (total_rewards - μ_g) / (σ_g + ε)
54:      
55:      # GRPO update (μ iterations)
56:      for iter in 1..μ do
57:          # Calculate policy ratio
58:          log_π_θ ← log_prob(π_θ, responses)
59:          log_π_old ← log_prob(π_old, responses)
60:          ratio ← exp(log_π_θ - log_π_old)
61:          
62:          # Clipped objective
63:          L_clip ← min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
64:          
65:          # KL divergence penalty
66:          if β > 0 then
67:              log_π_ref ← log_prob(π_ref, responses)
68:              KL ← exp(log_π_ref - log_π_θ) - (log_π_ref - log_π_θ) - 1
69:              L ← -L_clip + β·KL
70:          else
71:              L ← -L_clip
72:          end if
73:          
74:          # Update policy
75:          θ ← θ - ∇_θ L
76:      end for
77:   end for
78: end for
```

Key Points:
- Ranking learning: Learn relative relationships of image quality through pairwise comparison, rather than absolute scoring;
- Fidelity reward: Based on normal distribution assumption, measures consistency between predicted ranking and ground truth ranking;
- Intra-batch contrast: Each sample is compared with all other samples in the batch, fully utilizing batch information;
- Multiple generations: Generate G responses for each image, estimate prediction uncertainty through variance;
- GRPO optimization: Adopts group relative policy optimization, stabilizing training through clipping and KL penalty.
