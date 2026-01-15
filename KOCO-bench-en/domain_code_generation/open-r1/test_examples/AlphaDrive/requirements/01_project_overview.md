# AlphaDrive Project Overview


## 1. Algorithm Overview

AlphaDrive is the first method to combine GRPO (Group Relative Policy Optimization) based reinforcement learning with planning reasoning for autonomous driving. This method uses Vision-Language Models (VLM) to understand driving scenarios and generate driving decisions with reasoning processes, significantly improving planning performance and training efficiency.

Core idea: Train vision-language models using GRPO algorithm to enable them to:
1. Understand multimodal driving scenarios (images + navigation instructions + current speed)
2. Generate structured reasoning processes (within `<think>` tags)
3. Output executable driving decisions (within `<answer>` tags, including speed planning and path planning)

The training process adopts a multi-reward function fusion mechanism, including:
- Format reward: Ensures output conforms to `<think>...</think><answer>...</answer>` format
- Speed planning reward: Evaluates accuracy of speed decisions (KEEP/ACCELERATE/DECELERATE/STOP)
- Path planning reward: Evaluates accuracy of path decisions (STRAIGHT/LEFT_TURN/RIGHT_TURN/LEFT_CHANGE/RIGHT_CHANGE)

## 2. Training Pipeline (Pseudocode)

```
Algorithm: AlphaDrive Training Pipeline
Input: Dataset D (images, navigation, speed, ground truth plans), 
       Policy π_θ (VLM), Reference π_ref, 
       Reward functions R = {R_format, R_speed, R_path}
Config: num_generations G, max_prompt_length, max_completion_length,
        beta (KL penalty coefficient)

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Construct multimodal input (images + text prompts)
4:       prompts ← construct_multimodal_prompts(B)
5:       
6:       # Generate G responses (including reasoning process and decisions)
7:       responses ← generate(π_θ, prompts, G)
8:       
9:       # Compute multiple reward functions
10:      R_format ← check_format(responses)  # Check <think></think><answer></answer> format
11:      R_speed ← evaluate_speed_plan(responses, B.ground_truth_speed)
12:      R_path ← evaluate_path_plan(responses, B.ground_truth_path)
13:      
14:      # Fuse rewards
15:      R_total ← R_format + R_speed + R_path
16:      
17:      # Compute group advantages (GRPO)
18:      # Group G responses from same prompt
19:      mean_R ← group_mean(R_total, by_prompt)
20:      std_R ← group_std(R_total, by_prompt)
21:      advantages ← (R_total - mean_R) / (std_R + ε)
22:      
23:      # Compute KL divergence (relative to reference model)
24:      logp_θ ← log_prob(π_θ, responses)
25:      logp_ref ← log_prob(π_ref, responses)
26:      KL ← compute_kl(logp_θ, logp_ref)
27:      
28:      # Compute loss and update
29:      loss ← -mean(exp(logp_θ - logp_θ.detach()) * advantages - beta * KL)
30:      θ ← θ + Adam(∇_θ loss)
31:   end for
32: end for
```

Key points:
- Multimodal input: Combines driving scene images, navigation instructions, and current speed information
- Structured output: Model generates structured responses containing reasoning process (`<think>`) and decisions (`<answer>`)
- Multi-reward fusion: Simultaneously optimizes format correctness, speed planning, and path planning across three dimensions
- GRPO advantage estimation: Reduces variance through group-based relative comparison, improving training stability
- Complexity weighting: Different decision actions have different complexity weights (e.g., STOP is more important than KEEP)
- Diversity factor: Encourages model to generate diverse decisions within batch


