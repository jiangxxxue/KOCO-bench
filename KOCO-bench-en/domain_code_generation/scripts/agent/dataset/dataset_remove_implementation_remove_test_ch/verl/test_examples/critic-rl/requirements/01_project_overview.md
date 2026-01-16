# CTRL 项目总览（Critique Through Reinforcement Learning）

## 1. 算法概述

CTRL（Critique Through Reinforcement Learning）是一个通过强化学习训练语言模型进行批评（critique）的框架，核心目标是使模型能够在无人类监督的情况下自主学习批评能力，进而通过迭代的批评-修订（critique-revision）实现测试时扩展（test-time scaling）。

CTRL的核心创新在于：模型生成初始解答后，能够对自己的解答进行批评性分析，识别错误并给出修订建议，然后基于批评生成改进的解答。这一过程可以迭代多轮，持续提升解答质量。训练过程采用两阶段策略：第一阶段通过执行反馈生成合成批评数据进行监督微调（SFT），第二阶段基于verl框架使用可验证奖励进行强化学习（RL）。

核心思想：利用代码沙盒执行的可验证反馈作为奖励信号，训练模型学习生成有效的批评。当批评指出错误且修订后的代码通过测试时给予正向奖励，否则给予零奖励。通过这种方式，模型逐步学会识别代码中的真实错误并提出有效的改进建议。

## 2. 训练 Pipeline

### 阶段一：SFT 阶段（基于合成数据）
```
Algorithm: CTRL SFT Stage
Input: Dataset D, Base model M_base
Output: Fine-tuned model M_sft

1: for each problem P in D do
2:     # 生成初始解答
3:     initial_solution ← generate(M_base, P)
4:     
5:     # 执行测试获取反馈
6:     test_results ← execute_in_sandbox(initial_solution, P.test_cases)
7:     
8:     # 基于执行反馈生成合成批评
9:     if test_results.has_failures then
10:        critique ← generate_critique_from_feedback(test_results)
11:        dataset_sft.add(P, initial_solution, critique)
12:    end if
13: end for
14: 
15: # 监督微调
16: M_sft ← finetune(M_base, dataset_sft)
17: return M_sft
```

### 阶段二：RL 阶段（基于 verl）
```
Algorithm: CTRL RL Stage (verl-based)
Input: Dataset D, Policy π_θ (初始化为M_sft), Reference π_ref
Config: batch_size, n_samples, adv_estimator=grpo, kl_coef

1: for epoch in 1..E do
2:     for batch B ~ D do
3:         # Step 1: 生成批评（Actor生成）
4:         critiques ← generate_critiques(π_θ, B, n_samples)
5:         
6:         # Step 2: 构建修订提示
7:         revision_prompts ← build_revision_prompts(B, critiques)
8:         
9:         # Step 3: 生成修订解答（Reference/Proxy生成）
10:        revisions ← generate_revisions(π_ref, revision_prompts)
11:        
12:        # Step 4: 计算可验证奖励（沙盒执行）
13:        for i in 1..batch_size * n_samples do
14:            # 仅当批评有效时才执行修订代码
15:            if critique_is_valid(critiques[i]) then
16:                test_results[i] ← execute_in_sandbox(revisions[i], B.test_cases)
17:                rewards[i] ← compute_pass_rate(test_results[i])
18:            else
19:                rewards[i] ← 0.0  # 无效批评给予零奖励
20:            end if
21:        end for
22:        
23:        # Step 5: 将奖励分配到批评的最后一个token
24:        token_level_scores ← assign_reward_to_last_token(critiques, rewards)
25:        
26:        # Step 6: 计算参考策略对数概率（可选KL惩罚）
27:        ref_log_probs ← compute_log_prob(π_ref, critiques)
28:        old_log_probs ← compute_log_prob(π_θ, critiques)
29:        
30:        # Step 7: 应用KL惩罚
31:        kl_divergence ← compute_kl(old_log_probs, ref_log_probs)
32:        token_level_rewards ← token_level_scores - kl_coef * kl_divergence
33:        
34:        # Step 8: 计算GRPO优势
35:        # GRPO：基于同一prompt的多个样本计算组平均基线
36:        advantages ← compute_grpo_advantage(token_level_rewards, group_by_prompt)
37:        advantages ← normalize(advantages)
38:        
39:        # Step 9: PPO更新策略模型
40:        θ ← θ + optimizer(∇_θ L_PPO(π_θ; advantages))
41:     end for
42: end for
```

要点：
- **两阶段训练**：SFT 阶段生成合成数据并进行监督学习；RL 阶段基于 verl 使用可验证奖励优化
- **批评-修订机制**：模型生成批评 → 基于批评生成修订 → 执行修订代码 → 根据执行结果计算奖励
- **可验证奖励**：通过代码沙盒执行测试用例，计算通过率作为奖励信号（0-1之间的浮点数）
- **GRPO 优势估计**：无需价值网络（Critic），基于同一 prompt 的多个样本计算组平均基线
- **奖励分配**：将序列级奖励（通过率）分配到批评序列的最后一个有效 token
- **测试时扩展**：训练后的模型可进行多轮批评-修订迭代，持续提升解答质量

