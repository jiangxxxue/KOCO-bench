# CTRL Project Overview (Critique Through Reinforcement Learning)

## 1. Algorithm Overview

CTRL (Critique Through Reinforcement Learning) is a framework for training language models to perform critique through reinforcement learning. The core objective is to enable models to autonomously learn critique capabilities without human supervision, thereby achieving test-time scaling through iterative critique-revision cycles.

CTRL's core innovation lies in: After generating an initial solution, the model can perform critical analysis of its own answer, identify errors and provide revision suggestions, then generate improved solutions based on the critique. This process can iterate multiple rounds, continuously improving solution quality. The training process adopts a two-stage strategy: Stage 1 performs supervised fine-tuning (SFT) using synthetic critique data generated from execution feedback; Stage 2 uses verifiable rewards for reinforcement learning (RL) based on the verl framework.

Core idea: Utilize verifiable feedback from code sandbox execution as reward signals to train the model to generate effective critiques. Positive rewards are given when critiques identify errors and revised code passes tests; otherwise, zero rewards are given. Through this approach, the model gradually learns to identify real errors in code and propose effective improvement suggestions.

## 2. Training Pipeline

### Stage 1: SFT Stage (Based on Synthetic Data)
```
Algorithm: CTRL SFT Stage
Input: Dataset D, Base model M_base
Output: Fine-tuned model M_sft

1: for each problem P in D do
2:     # Generate initial solution
3:     initial_solution ← generate(M_base, P)
4:     
5:     # Execute tests to get feedback
6:     test_results ← execute_in_sandbox(initial_solution, P.test_cases)
7:     
8:     # Generate synthetic critique based on execution feedback
9:     if test_results.has_failures then
10:        critique ← generate_critique_from_feedback(test_results)
11:        dataset_sft.add(P, initial_solution, critique)
12:    end if
13: end for
14: 
15: # Supervised fine-tuning
16: M_sft ← finetune(M_base, dataset_sft)
17: return M_sft
```

### Stage 2: RL Stage (verl-based)
```
Algorithm: CTRL RL Stage (verl-based)
Input: Dataset D, Policy π_θ (initialized as M_sft), Reference π_ref
Config: batch_size, n_samples, adv_estimator=grpo, kl_coef

1: for epoch in 1..E do
2:     for batch B ~ D do
3:         # Step 1: Generate critiques (Actor generation)
4:         critiques ← generate_critiques(π_θ, B, n_samples)
5:         
6:         # Step 2: Build revision prompts
7:         revision_prompts ← build_revision_prompts(B, critiques)
8:         
9:         # Step 3: Generate revised solutions (Reference/Proxy generation)
10:        revisions ← generate_revisions(π_ref, revision_prompts)
11:        
12:        # Step 4: Compute verifiable rewards (sandbox execution)
13:        for i in 1..batch_size * n_samples do
14:            # Only execute revised code when critique is valid
15:            if critique_is_valid(critiques[i]) then
16:                test_results[i] ← execute_in_sandbox(revisions[i], B.test_cases)
17:                rewards[i] ← compute_pass_rate(test_results[i])
18:            else
19:                rewards[i] ← 0.0  # Zero reward for invalid critiques
20:            end if
21:        end for
22:        
23:        # Step 5: Assign rewards to last token of critiques
24:        token_level_scores ← assign_reward_to_last_token(critiques, rewards)
25:        
26:        # Step 6: Compute reference policy log probabilities (optional KL penalty)
27:        ref_log_probs ← compute_log_prob(π_ref, critiques)
28:        old_log_probs ← compute_log_prob(π_θ, critiques)
29:        
30:        # Step 7: Apply KL penalty
31:        kl_divergence ← compute_kl(old_log_probs, ref_log_probs)
32:        token_level_rewards ← token_level_scores - kl_coef * kl_divergence
33:        
34:        # Step 8: Compute GRPO advantages
35:        # GRPO: Compute group average baseline based on multiple samples from same prompt
36:        advantages ← compute_grpo_advantage(token_level_rewards, group_by_prompt)
37:        advantages ← normalize(advantages)
38:        
39:        # Step 9: PPO update policy model
40:        θ ← θ + optimizer(∇_θ L_PPO(π_θ; advantages))
41:     end for
42: end for
```

Key points:
- **Two-stage training**: SFT stage generates synthetic data and performs supervised learning; RL stage optimizes using verifiable rewards based on verl
- **Critique-revision mechanism**: Model generates critique → Generate revision based on critique → Execute revised code → Compute rewards based on execution results
- **Verifiable rewards**: Execute test cases through code sandbox, compute pass rate as reward signal (float between 0-1)
- **GRPO advantage estimation**: No value network (Critic) needed, compute group average baseline based on multiple samples from same prompt
- **Reward assignment**: Assign sequence-level reward (pass rate) to last valid token of critique sequence
- **Test-time scaling**: Trained model can perform multiple rounds of critique-revision iteration, continuously improving solution quality


