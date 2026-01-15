# CANOE Project Overview (Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning)

## 1. Algorithm Overview

CANOE (Contextual fAithfulness via syNthetic tasks and reinfOrcEment learning) is a systematic framework designed to improve large language model (LLM) faithfulness in given contexts, applicable to both short-text and long-text generation tasks without requiring human annotation. This method constructs high-quality, easily verifiable training data through synthetic short-text question-answering (QA) data, and proposes the Dual-GRPO reinforcement learning method to simultaneously optimize short-text and long-text response generation.

Core ideas:
1. Synthetic task construction: Construct automatically verifiable training data through four diverse tasks synthesizing short-text QA data, including Counterfactual QA and Factual QA.
2. Dual-GRPO training: Based on GRPO (Group Relative Policy Optimization) algorithm, design three rule-based reward functions to simultaneously optimize short-text and long-text generation, avoiding over-optimization of short-text generation.
3. Multi-task reward fusion: Combine accuracy reward, format reward, influence reward, and length reward to comprehensively evaluate model generation quality.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: CANOE Training Pipeline (Dual-GRPO)
Input: Dataset D (containing context and question), Policy π_θ, Reference π_ref
Config: num_generations n, reward_funcs=[accuracy, format, influence, length],
        temperature, max_completion_length, do_long_answer=True

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # Stage 1: Generate short-text answers
4:       prompts ← format_prompts(B, SYSTEM_PROMPT)  # Contains <context>, <think>, <long_answer>, <answer> tags
5:       completions ← generate(π_θ, prompts, n)  # Generate n responses
6:       
7:       # Compute short-text rewards
8:       R_accuracy ← accuracy_reward(completions, B.solution)  # Symbolic verification or string matching
9:       R_format ← format_reward(completions)  # Check <think><long_answer><answer> format
10:      
11:      # Stage 2: Generate long-text answers (if enabled)
12:      if do_long_answer then
13:          # Extract <long_answer> content to replace original context
14:          long_answers ← extract_long_answer(completions)
15:          new_contexts ← replace_context(B, long_answers)
16:          completions_long ← generate(π_θ, new_contexts, 1)  # Generate 1 long answer per sample
17:          
18:          # Compute long-text rewards
19:          R_influence ← influence_reward(completions_long, B.solution)  # Long answer accuracy
20:          R_length ← length_reward(completions, B.context)  # Length constraint: 0.2 < len(long_answer)/len(context) < 0.8
21:      else
22:          R_influence ← 0
23:          R_length ← 0
24:      
25:      # Fuse all rewards
26:      R_total ← R_accuracy + R_format + R_influence + R_length
27:      
28:      # Compute log probabilities and KL divergence
29:      logps ← compute_logps(π_θ, completions)
30:      ref_logps ← compute_logps(π_ref, completions)
31:      KL ← compute_kl(logps, ref_logps)
32:      
33:      # GRPO advantage estimation (within-group relative advantage)
34:      A ← compute_grpo_advantage(R_total, grouping_by_prompt)
35:      A ← whiten(A)  # Normalize
36:      
37:      # Policy gradient update
38:      L ← -A * logps + β * KL  # β controls KL penalty
39:      θ ← θ - Adam(∇_θ L)
40:   end for
41: end for
```

Key points:
- Dual-stage generation: First generate complete response containing short answer, then regenerate based on long answer to evaluate influence;
- Multi-reward fusion: Four reward functions work together to ensure generation quality, format compliance, and contextual faithfulness;
- GRPO advantage estimation: Group n responses from same prompt, compute within-group relative advantage to reduce variance;
- Length constraint: Ensure long answer length is within reasonable range (20%-80% of context length), avoiding too long or too short.

