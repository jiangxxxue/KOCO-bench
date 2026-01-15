# PRIME算法Pipeline

## 1. 算法概述

PRIME (Process Reinforcement through Implicit rEwards) 算法通过以下步骤进行训练：首先对每个prompt进行过采样生成多个候选响应，然后使用规则验证器检查响应正确性并筛选出中等难度的样本。接着计算隐式奖励模型与参考模型的对数概率差异作为token级奖励信号。在模型更新阶段，根据配置策略更新奖励模型参数，使用RLOO算法融合过程奖励和结果奖励计算优势函数，最后通过PPO损失更新策略模型。整个过程实现了策略模型与奖励模型的联合在线训练。

## 2. 训练Pipeline

```
Algorithm: PRIME Training Pipeline
Input: Dataset D, Policy model π_θ, Reference model π_ref, Reward model π_φ
Parameters: oversample_factor=4.0, n=4, beta_train=0.05, reward_dpo_coef=5, reward_gt_coef=5
Output: Optimized policy model π_θ

1: Initialize π_φ ← π_θ, π_ref ← π_θ
2: for epoch = 1 to total_epochs do
3:    for batch B ∼ D do
4:        // Data Preparation and Response Generation
5:        B_expanded ← expand_batch(B, oversample_factor)
6:        responses ← generate_responses(π_θ, B_expanded, n)
7:        
8:        // Response Verification and Filtering
9:        scores ← verify_responses(responses)
10:       B_filtered ← filter_samples(B_expanded, responses, scores)
11:       
12:       // Log Probability Computation
13:       log_probs_old ← compute_log_prob(π_θ, B_filtered)
14:       log_probs_ref ← compute_log_prob(π_ref, B_filtered)
15:       
16:       // Reward Model Update and Score Computation
17:       if update == "before" then
18:           π_φ ← update_reward_model(π_φ, B_filtered)
19:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
20:       elif update == "after" then
21:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
22:           π_φ ← update_reward_model(π_φ, B_filtered)
23:       elif update == "reverse" then
24:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
25:           statistics ← compute_batch_statistics(rm_scores, scores)
26:           π_φ ← update_reward_model(π_φ, B_filtered, statistics)
27:       else
28:           rm_scores ← compute_reward_scores(π_φ, B_filtered)
29:       end if
30:       
31:       // Advantage Computation
32:       advantages ← compute_rloo_advantage(rm_scores, scores, n, config)
33:       
34:       // Policy Update
35:       π_θ ← update_policy(π_θ, B_filtered, advantages)
36:    end for
37: end for
38: return π_θ
```

Pipeline中的关键函数说明：
- `generate_responses(π, batch, n)`: 使用策略模型为每个prompt生成n个候选响应
- `verify_responses(responses)`: 使用规则验证器检查响应正确性，返回二元标签
- `filter_samples(batch, responses, scores)`: 基于准确率和长度过滤样本并下采样
- `compute_rloo_advantage(rm_scores, scores, n, config)`: 融合过程奖励和结果奖励计算RLOO优势
- `compute_reward_scores(π_φ, batch)`: 计算隐式奖励模型分数
- `update_reward_model(π_φ, batch)`: 使用交叉熵损失更新奖励模型参数
- `update_policy(π_θ, batch, advantages)`: 使用PPO损失更新策略模型参数


## 3. 应用场景

### 数学推理任务
- 输入: 数学问题文本
- 输出: 包含推理步骤的数学解答
- 验证方式: LaTeX格式答案的精确匹配
- 数据来源: GSM8K, MATH, AMC, AIME等数学数据集

### 代码生成任务
- 输入: 编程问题描述
- 输出: 可执行的程序代码
- 验证方式: 测试用例通过率计算
- 数据来源: APPS, CodeContests, TACO, LeetCode等编程数据集
