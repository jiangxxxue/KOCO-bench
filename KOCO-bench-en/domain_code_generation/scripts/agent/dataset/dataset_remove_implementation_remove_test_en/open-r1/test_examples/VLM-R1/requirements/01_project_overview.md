# VLM-R1 Project Overview (A Stable and Generalizable R1-style Large Vision-Language Model)

## 1. Algorithm Overview

VLM-R1 is a stable and generalizable R1-style large vision-language model framework designed to enhance the reasoning capabilities and generalization performance of multimodal models on vision-language tasks through reinforcement learning methods. This approach is based on the GRPO (Group Relative Policy Optimization) algorithm, specifically designed and optimized for the characteristics of vision-language models.

Core Ideas:
1. Vision-Language Joint Reasoning: Through structured Chain-of-Thought format, the model is required to first output the reasoning process in `<think>` tags, then provide the final answer in `<answer>` tags, thereby enhancing the model's interpretability and reasoning capabilities.
2. Multi-task Adaptation Capability: Supports various vision-language tasks, including Referring Expression Comprehension (REC), Open-Vocabulary Object Detection (OVD), multimodal math reasoning (Math), GUI defect detection, etc., achieving targeted optimization through task-specific reward functions.
3. Flexible Reward Mechanism: Designed a rich library of reward functions, including accuracy reward, format reward, length reward, repetition penalty, etc., which can be flexibly combined according to task requirements.
4. Generalization Advantage: Compared to Supervised Fine-Tuning (SFT) methods, VLM-R1 demonstrates stronger generalization capabilities on out-of-domain data, able to transfer reasoning abilities to unseen data distributions.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: VLM-R1 Training Pipeline (GRPO for Vision-Language Models)
Input: Dataset D (containing image, question, solution), Policy π_θ, Reference π_ref
Config: num_generations n, reward_funcs=[accuracy, format, length, repetition],
        temperature, max_completion_length, task_type

1: # Data preprocessing
2: for each sample in D do
3:     image_path ← sample.image  # Store image path
4:     problem ← sample.question  # Extract question text
5:     solution ← sample.solution  # Extract standard answer
6:     accu_reward_method ← determine_reward_method(task_type)  # Determine reward method based on task type
7:     prompt ← format_prompt(image_path, problem, task_type)  # Construct multimodal prompt
8: end for

9: # Training loop
10: for epoch in 1..E do
11:   for batch B ~ D do
12:       # Generation phase
13:       prompts ← format_multimodal_prompts(B)  # Multimodal prompts containing images and text
14:       completions ← generate(π_θ, prompts, n)  # Generate n responses
15:       
16:       # Calculate multiple rewards
17:       R_accuracy ← compute_accuracy_reward(completions, B.solution, accu_reward_method)
18:       # Select different accuracy reward calculation methods based on task type:
19:       #   - REC task: Calculate IoU between predicted bounding box and ground truth
20:       #   - OVD task: Calculate mAP or mAP@50
21:       #   - Math task: Use symbolic verification or numerical matching
22:       #   - MCQ task: Extract options and perform exact matching
23:       #   - Other tasks: Use fuzzy matching or LLM evaluation
24:       
25:       R_format ← format_reward(completions)  # Check <think><answer> format
26:       
27:       if "length" in reward_funcs then
28:           R_length ← cosine_reward(completions)  # Cosine-based length reward
29:           # For correct answers: shorter length → higher reward (1.0 → 0.5)
30:           # For incorrect answers: shorter length → greater penalty (0.0 → -0.5)
31:       else
32:           R_length ← 0
33:       
34:       if "repetition" in reward_funcs then
35:           R_repetition ← repetition_reward(completions)  # Detect repetitive content
36:           # Use n-gram statistics to detect repetition, higher repetition rate → greater penalty (0 → -1.0)
37:       else
38:           R_repetition ← 0
39:       
40:       # Fuse all rewards
41:       R_total ← R_accuracy + R_format + R_length + R_repetition
42:       
43:       # Calculate log probabilities and KL divergence
44:       logps ← compute_logps(π_θ, completions)
45:       ref_logps ← compute_logps(π_ref, completions)
46:       KL ← compute_kl(logps, ref_logps)
47:       
48:       # GRPO advantage estimation (group-wise relative advantage)
49:       A ← compute_grpo_advantage(R_total, grouping_by_prompt)
50:       A ← whiten(A)  # Normalize
51:       
52:       # Policy gradient update
53:       L ← -A * logps + β * KL  # β controls KL penalty
54:       θ ← θ - Adam(∇_θ L)
55:   end for
56: end for
```

Key Points:
- Multimodal input processing: Supports single-image and multi-image inputs, images are processed through vision encoder and fused with text features;
- Task-specific rewards: Select corresponding accuracy reward calculation methods based on different task types (REC, OVD, Math, etc.);
- Structured output: Force model to output structured format containing reasoning process (`<think>`) and final answer (`<answer>`);
- Length and repetition control: Encourage concise answers through cosine reward function, avoid redundant content through repetition penalty;
- GRPO advantage estimation: Group n responses by the same prompt, calculate group-wise relative advantage to reduce variance;
- Vision module freezing option: Support freezing vision encoder parameters, only train language model part to improve training efficiency.
