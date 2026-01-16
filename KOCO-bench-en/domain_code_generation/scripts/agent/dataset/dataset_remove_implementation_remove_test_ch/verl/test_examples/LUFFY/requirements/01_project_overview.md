# LUFFY 项目总览（Learning to Reason Under Off-Policy Guidance）

## 1. 算法概述

LUFFY（Learning to Reason Under Off-Policy Guidance）是一个创新的强化学习框架，旨在通过融合离策略（off-policy）推理轨迹来弥合零强化学习（zero-RL）与模仿学习之间的差距。该方法基于 GRPO（Group Relative Policy Optimization）构建，在训练过程中同时利用在策略（on-policy）的自我生成样本和离策略的外部示范数据。


LUFFY 的训练范式允许模型从更强的教师模型中引导学习，同时保持自主探索能力，在数学推理等需要复杂推理链的任务上表现出色。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: LUFFY Training Pipeline
Input: Dataset D (含 prompt 与可选的 off-policy target), 
       Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n (on-policy samples), 
        n_prefix (off-policy samples),
        prefix_ratio (off-policy guidance ratio),
        off_policy_reshape (policy shaping method),
        adv_estimator=grpo_split

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 混合采样：生成 n 个响应（部分使用 off-policy prefix）
4:       for each prompt in B do
5:           # 生成 n_prefix 个带 off-policy 前缀的响应
6:           prefix_ratio ← sample_prefix_ratio(strategy, global_step)
7:           off_policy_responses ← rollout_with_prefix(π_θ, prompt, 
8:                                    target_prefix, prefix_ratio, n_prefix)
9:           # 生成 (n - n_prefix) 个纯 on-policy 响应
10:          on_policy_responses ← rollout(π_θ, prompt, n - n_prefix)
11:          responses ← concat(off_policy_responses, on_policy_responses)
12:      end for
13:
14:      # 计算可验证奖励（VR）
15:      VR ← R.compute(B, responses)
16:
17:      # 拒绝采样：移除全部成功或全部失败的 prompt 组
18:      valid_mask ← filter_extreme_groups(VR, responses)
19:      responses ← responses[valid_mask]
20:
21:      # 计算旧策略 log 概率
22:      old_log_prob ← π_θ.compute_log_prob(responses)
23:
24:      # 计算参考策略 log 概率（用于 KL 惩罚）
25:      if use_reference_policy then
26:          ref_log_prob ← π_ref.compute_log_prob(responses)
27:          token_level_rewards ← VR - β * KL(old_log_prob, ref_log_prob)
28:      else
29:          token_level_rewards ← VR
30:      end if
31:
32:      # 分组优势估计（GRPO with split）
33:      # 仅使用 on-policy 样本计算基线
34:      A ← compute_grpo_split_advantage(token_level_rewards, 
35:                                       on_policy_mask, group_by_prompt)
36:
37:      # 应用前缀奖励权重
38:      A ← apply_prefix_weight(A, prefix_mask, α, β)
39:
40:      # 混合策略损失计算（on-policy + off-policy）
41:      for micro_batch in split(responses, micro_batch_size) do
42:          # 计算当前策略 log 概率
43:          log_prob ← π_θ.forward(micro_batch)
44:          
45:          # On-policy 损失（标准 PPO clip）
46:          ratio_on ← exp(log_prob - old_log_prob)
47:          if on_policy_reshape != 'no_reshape' then
48:              ratio_on ← reshape_ratio(ratio_on, method=on_policy_reshape)
49:          end if
50:          L_on ← -A * clip(ratio_on, 1-ε, 1+ε)
51:          
52:          # Off-policy 损失（重要性采样 + policy shaping）
53:          ratio_off ← exp(log_prob)  # 相对于隐式行为策略
54:          if off_policy_reshape != 'no_reshape' then
55:              ratio_off ← reshape_ratio(ratio_off, method=off_policy_reshape)
56:          end if
57:          ratio_off ← clip(ratio_off, min_clip, max_clip)
58:          L_off ← -A * ratio_off
59:          
60:          # 混合损失
61:          L_policy ← prefix_mask * L_off + (1 - prefix_mask) * L_on
62:          
63:          # 熵正则化
64:          L_entropy ← -entropy_coeff * H(π_θ)
65:          
66:          # 总损失
67:          L ← mean(L_policy) + L_entropy
68:      end for
69:
70:      # 更新策略
71:      θ ← θ - Adam(∇_θ L)
72:   end for
73: end for
```

## 3. 关键特性

### 3.1 离策略前缀策略
- **动态前缀比例**：支持多种策略（random, linear, fixed）控制使用多少 off-policy 前缀
- **课程学习**：通过 linear 策略，训练初期使用更多 off-policy 引导，后期逐渐增加自主探索
- **灵活配置**：可控制每个 prompt 生成多少个带前缀的样本（n_prefix）

### 3.2 策略塑形方法
支持多种 ratio 变换方法来强调重要动作：
- `no_reshape`：标准重要性采样
- `logp`：使用 log 概率代替概率比
- `p_logp`：概率比与 log 概率的加权组合
- `square_root`：概率比的平方根
- `pow`：概率比的幂次变换
- `p_div_p_α`：概率归一化变换 p/(p+α)

### 3.3 分组优势估计
- **GRPO Split**：改进的 GRPO，仅使用 on-policy 样本计算组内基线
- **降低方差**：通过组内标准化减少优势估计的方差
- **公平比较**：确保 off-policy 样本不影响 on-policy 样本的基线计算

### 3.4 拒绝采样与课程学习
- 自动过滤全部成功或全部失败的 prompt 组
- 提高训练信噪比，避免无信息样本
- 支持基于奖励的样本筛选策略
