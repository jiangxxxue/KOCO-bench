# PACS 项目总览（Implicit Actor Critic Coupling via a Supervised Learning Framework for RLVR）

## 1. 算法概述

PACS（im**P**licit **A**ctor **C**ritic coupling via a **S**upervised learning framework）是一种新颖的 RLVR（Reinforcement Learning with Verifiable Rewards）框架，通过监督学习范式实现隐式的 Actor-Critic 耦合。该方法将结果奖励视为可预测的标签，将 RLVR 问题重新表述为对评分函数的监督学习任务，该评分函数由策略模型参数化并使用交叉熵损失进行优化。

核心思想：PACS 不同于传统的 PPO 或 GRPO 等方法，它将策略优化问题转化为一个二分类问题。对于每个 prompt，模型生成 n 个响应，每个响应根据可验证奖励（如数学问题的正确性）获得一个二元标签（0 或 1）。通过计算策略相对于参考策略的对数概率变化作为"奖励"，并使用 RLOO 或 GRPO 等方法估计优势函数，PACS 将优势函数作为 logit 输入，将可验证奖励作为标签，使用二元交叉熵损失（BCEWithLogitsLoss）进行训练。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: PACS Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n, beta, adv_estimator ∈ {'rloo','grpo','naive'},
        use_weight, weight_mode ∈ {'question','only_positive','only_negative'}

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 生成 n 个响应（vLLM/HF 推理）
4:       responses ← rollout(π_θ, B, n)
5:       
6:       # 计算可验证奖励（VR）
7:       VR ← R.compute(B, responses)  # 二元标签：0 或 1
8:       
9:       # 计算旧策略的对数概率
10:      old_log_prob ← π_θ.compute_log_prob(B, responses)
11:      
12:      # 计算当前策略的对数概率
13:      log_prob ← π_θ.forward(B, responses)
14:      
15:      # 计算"奖励"（策略改进的度量）
16:      if reward_method == '1' then
17:          reward ← beta × sum(log_prob - old_log_prob)
18:      elif reward_method == '2' then
19:          reward ← beta × sum(log_prob) / response_length
20:      
21:      # 计算优势函数
22:      if adv_estimator == 'rloo' then
23:          A ← RLOO(reward, grouping_by_prompt)
24:      elif adv_estimator == 'grpo' then
25:          A ← GRPO(reward, grouping_by_prompt)
26:      elif adv_estimator == 'naive' then
27:          A ← reward
28:      
29:      # 计算样本权重（可选）
30:      if use_weight then
31:          w ← compute_weight(VR, rollout_n, weight_mode)
32:      else
33:          w ← 1
34:      
35:      # 计算 PACS 损失（二元交叉熵）
36:      L_PACS ← BCEWithLogitsLoss(A, VR, weight=w)
37:      
38:      # 更新策略
39:      θ ← θ - Adam(∇_θ L_PACS)
40:   end for
41: end for
```

要点：
- 监督学习范式：将 RLVR 转化为监督学习问题，优势函数作为 logit，可验证奖励作为标签；
- 分组与优势估计：按同一 prompt 的 n 个响应为一组，使用 RLOO/GRPO 计算优势；
- Beta 参数：控制策略更新的激进程度，beta 越大，策略变化越快；
- 样本权重：可选择性地对正负样本赋予不同权重，处理类别不平衡问题；
- 无需 Critic：PACS 不需要单独的价值网络，简化了训练流程。
