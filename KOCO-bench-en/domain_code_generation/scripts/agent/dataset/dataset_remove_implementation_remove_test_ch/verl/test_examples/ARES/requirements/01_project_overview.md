# ARES 项目总览（Adaptive Reasoning via Entropy Shaping）

## 1. 算法概述

ARES（Adaptive Reasoning via Entropy Shaping）是一个基于难度感知的自适应多模态推理框架，旨在根据输入问题的难度动态调整模型的推理资源分配。该方法基于一个核心观察：现有的多模态推理模型在**简单**任务上倾向于过度思考（生成冗长的推理轨迹），而在**困难**任务上又探索不足（由于搜索不充分导致错失解决方案）。

ARES 引入了基于**高窗口熵（HWE）token**的机制来检测推理不确定性时刻，并灵活调整探索强度。HWE token 通过在滑动窗口内对 token 级熵进行平均来识别持续的高不确定性区域，这些区域标志着模型需要更多探索的关键推理节点。

ARES 采用**两阶段训练流程**：

1. **自适应冷启动阶段（Adaptive Cold-Start）**：构建难度感知的多模态和文本推理样本，其中推理轨迹长度与任务难度相匹配，使模型学习到难度意识。该阶段使用自适应 KL 引导的微调策略，确保模型在不同难度任务上建立稳健的初始化。

2. **自适应熵策略优化（AEPO）**：利用 HWE token 作为触发器决定**何时**进一步探索，结合**分层熵奖励**和**动态 KL 控制**决定**探索多少**。通过在线难度分桶（difficulty bucketing）和熵感知的 rollout 分配，AEPO 实现了不确定性感知的、难度自适应的推理深度调整。

实验结果表明，ARES 在推理**效率**和**准确率**之间实现了更优的权衡，在多模态、数学和逻辑基准测试中优于基线方法，同时降低推理成本，缩小与商业系统的差距。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: ARES Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref
Config: difficulty_thresholds = {acc_high: 2/3, acc_low: 1/3},
        alpha_entropy_dict = {"easy": α_e, "medium": α_m, "hard": α_h},
        target_entropy_token_dict = {"easy": T_e, "medium": T_m, "hard": T_h}

# 阶段 1: 自适应冷启动
1: for epoch in 1..E_coldstart do
2:   for batch B ~ D_coldstart do
3:       # 使用难度感知的数据，其中推理长度与难度匹配
4:       responses ← rollout(π_θ, B)
5:       # 计算基础奖励
6:       R_base ← RewardManager.compute(B, responses)
7:       # 自适应 KL 引导的微调
8:       L_coldstart ← L_SFT + β_adaptive · KL(π_θ || π_ref)
9:       θ ← θ - ∇_θ L_coldstart
10:  end for
11: end for

# 阶段 2: 自适应熵策略优化（AEPO）
12: Initialize difficulty_dict ← bootstrap_difficulty(D, π_θ)
13: for epoch in 1..E_RL do
14:   for batch B ~ D do
15:       # 在线难度分桶
16:       for prompt p in B do
17:           responses_p ← rollout(π_θ, p, n=n_rollout)
18:           acc_rate_p ← mean([verify(r) for r in responses_p])
19:           if acc_rate_p >= 2/3 then difficulty[p] ← "easy"
20:           elif acc_rate_p >= 1/3 then difficulty[p] ← "medium"
21:           else difficulty[p] ← "hard"
22:       end for
23:       
24:       # 熵感知 rollout：在 HWE 区域触发探索
25:       for response r in responses do
26:           window_entropies ← compute_window_entropy(r, window_size=4)
27:           HWE_tokens ← identify_high_entropy_tokens(window_entropies, threshold=τ)
28:           # HWE token 标记高不确定性区域，需要更深入探索
29:       end for
30:       
31:       # 分层熵奖励塑形
32:       for response r, difficulty d in zip(responses, difficulties) do
33:           α_d ← alpha_entropy_dict[d]
34:           T_d ← target_entropy_token_dict[d]
35|           N_HWE ← count(HWE_tokens in r)
36|           
37|           # 根据难度自适应调整熵奖励
38|           if d == "easy" then
39|               # 简单任务：惩罚过度探索
40|               if verified(r) and N_HWE > T_d then
41|                   R_entropy ← -α_d · penalty(N_HWE - T_d)
42|               else
43|                   R_entropy ← 0
44|           elif d == "medium" then
45|               # 中等任务：平衡探索
46|               if verified(r) then
47|                   R_entropy ← -α_d · deviation(N_HWE, T_d)
48|               else
49|                   R_entropy ← α_d · bonus(N_HWE)
50|           else  # hard
51|               # 困难任务：鼓励探索
52|               if verified(r) then
53|                   R_entropy ← α_d · bonus(N_HWE)
54|               else
55|                   R_entropy ← α_d · bonus(N_HWE)
56|       end for
57|       
58|       # 动态 KL 控制：在 HWE 窗口内放松 KL 约束
59|       for response r in responses do
60:           KL_coef_token ← []
61:           for token t in r do
62:               if t is HWE_token then
63:                   # 在高熵区域放松 KL 约束，允许更多探索
64:                   KL_coef_token.append(β_low)
65:               else
66:                   KL_coef_token.append(β_normal)
67:           end for
68:       end for
69:       
70:       # 计算总奖励：准确率 + 熵塑形
71:       R_total ← R_accuracy + R_entropy - KL_coef_token · KL(π_θ || π_ref)
72|       
73|       # 优势估计（GRPO/RLOO）
74|       A ← compute_advantage(R_total, grouping_by_prompt)
75|       A ← whiten(A)
76|       
77|       # PPO 更新
78|       θ ← θ + Adam(∇_θ L_PPO(π_θ; A))
79|       
80|       # 自适应更新难度目标和熵系数
81|       update_alpha_entropy(difficulty_dict, α_entropy_dict, target_entropy_dict)
82|   end for
83| end for
```

要点：
- **难度分桶**：根据 rollout 准确率动态将问题分为 easy/medium/hard 三个桶；
- **HWE 检测**：使用滑动窗口平均 token 级熵，识别持续的高不确定性区域作为探索分支点；
- **分层奖励**：easy 任务惩罚过度探索，hard 任务鼓励深入探索，medium 任务平衡两者；
- **动态 KL**：在 HWE 窗口内降低 KL 惩罚系数，允许模型在高不确定性区域偏离参考策略；
- **自适应调整**：通过对偶上升（dual ascent）机制动态调整每个难度桶的熵系数和目标 token 数，实现难度感知的推理资源预算控制。

