# PURE 项目总览（Process-supervised Reinforcement Learning）


## 1. 算法概述

PURE（Process-supervised RL）采用“过程奖励 + 可验证奖励”的融合训练范式，以最小式（min-form）信用分配解决链式推理中“薄弱环节掩盖”问题。训练中，策略模型为每个 prompt 生成 n 个响应，通过 RewardManager 计算结果级可验证奖励（VR），并通过 PRM 预测 token/step 级过程奖励（PR）。随后，基于 RLOO 的优势估计融合 VR 与 PR，使用 PPO 更新策略模型。

核心思想：对过程奖励序列 r_t 施加温度加权 w_t = softmax(-r_t/T) 并做加权求和，T→0 时近似严格 min(r_t)，以突出错误最大、最关键的一步，实现“Stop Summation”。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: PURE Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, Process RM f_φ, RewardManager R
Config: rollout.n, credit_assignment ∈ {'gamma-decay','strict min-form', T>0},
        coef_vr, coef_pr, adv_estimator=rloo

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 生成 n 个响应（vLLM/HF 推理）
4:       responses ← rollout(π_θ, B, n)
5:       # 计算结果级 VR 与（可选）额外统计
6:       VR ← R.compute(B, responses)
7:       # 计算 PRM 过程奖励（token/step 级）
8:       PR_token ← f_φ.forward(B, responses)
9:       # 近似 min-form 权重（或严格/γ 衰减）
10:      if credit_assignment == T > 0 then
11:          w ← softmax(-PR_token/T, dim=-1)
12:          PR_token ← PR_token ⊙ w
13:      elif credit_assignment == 'strict min-form' then
14:          PR_token ← strictly_select_min(PR_token)
15:      else  # 'gamma-decay'
16:          PR_token ← apply_gamma_decay(PR_token, γ)
17:
18:      # 融合奖励并计算 RLOO 优势
19:      A ← RLOO(PR_token, VR, grouping_by_prompt, coef_pr, coef_vr)
20:      A ← whiten(A)  # 可选
21:
22:      # PPO 更新
23:      θ ← θ + Adam(∇_θ L_PPO(π_θ; A))
24:   end for
25: end for
```

要点：
- 分组与 RLOO：按同一 prompt 的 n 个响应为一组，计算 leave-one-out 基线降低方差；
- 奖励融合：`coef_vr` 与 `coef_pr` 控制 VR 与 PR 的相对权重；
- Min-form：温度型近似默认更稳定，严格 min-form 作为极限情形；
- 课程学习：可在 batch 级移除全 0 或全 1 的极端样本，提升训练信噪比。


