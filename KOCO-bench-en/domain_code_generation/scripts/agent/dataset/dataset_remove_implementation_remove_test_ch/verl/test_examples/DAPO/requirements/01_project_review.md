# DAPO 项目总览（Decoupled Clip and Dynamic Sampling Policy Optimization）

## 1. 算法概述

DAPO 基于 PPO 算法框架，通过以下改进提升训练效率与稳定性：

### 1.1 解耦裁剪（Clip-Higher）

传统 PPO 使用单一裁剪参数 ε，限制重要性采样比 ratio 在 [1-ε, 1+ε] 范围内。DAPO 引入非对称裁剪：

```
L_clip = max(-A * r, -A * clip(r, 1-ε_low, 1+ε_high))
```

其中：
- ε_low：负优势（A < 0）方向的裁剪阈值
- ε_high：正优势（A > 0）方向的裁剪阈值
- 通常设置 ε_high > ε_low，允许更激进地提升好的行为，同时保守地抑制坏的行为

### 1.2 动态采样（Dynamic Sampling with Group Filtering）

在标准 PPO 中，每个 prompt 生成 n 个响应后直接用于训练。DAPO 引入组过滤机制：

1. 对于每个 prompt 的 n 个响应，计算某个指标（如准确率 acc、最终奖励 seq_final_reward）
2. 过滤掉所有响应指标完全相同的组（如全部正确或全部错误），因为这些组无法提供有效的对比信号
3. 若过滤后的有效组数量 < train_batch_size，继续采样新批次
4. 重复采样直到达到 train_batch_size 或达到上限 max_num_gen_batches

这一机制显著提升了样本利用效率，尤其在数据难度较大时。

### 1.3 灵活损失聚合

DAPO 支持多种损失聚合模式（loss_agg_mode）：

- **token-mean**（推荐）：在所有 token 上取平均，使每个 token 对损失的贡献均等
- **seq-mean-token-sum**：先在序列内对 token 求和，再在序列间取平均
- **seq-mean-token-mean**：先在序列内对 token 取平均，再在序列间取平均

token-mean 模式使得策略梯度更关注每个 token 的质量，而非序列长度。

### 1.4 超长奖励塑形

为防止模型生成超长但无意义的输出，DAPO 引入线性惩罚：

```
若 response_len > expected_len:
    penalty = min(-(response_len - expected_len) / buffer_len * penalty_factor, 0)
    reward += penalty
```

其中 expected_len = max_response_length - overlong_buffer_len，惩罚在 buffer 区间内线性增加。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: DAPO Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n, filter_groups.enable, filter_groups.metric,
        clip_ratio_low, clip_ratio_high, loss_agg_mode

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       batch_data ← []
4:       num_gen_batches ← 0
5:       # 动态采样循环
6:       while len(batch_data) < train_batch_size do
7:           # 生成 n 个响应
8:           responses ← rollout(π_θ, B, n)
9:           
10:          # 计算奖励
11:          rewards ← R.compute(B, responses)
12:          
13:          # 组过滤（若启用）
14:          if filter_groups.enable then
15:              # 按 prompt 分组，计算每组的指标方差
16:              for each prompt_group in responses do
17:                  metric_vals ← [compute_metric(resp) for resp in prompt_group]
18:                  if std(metric_vals) == 0 then  # 全部相同
19:                      discard prompt_group
20:              end for
21:          end if
22:          
23:          batch_data ← batch_data + filtered_responses
24:          num_gen_batches += 1
25:          
26:          if num_gen_batches >= max_num_gen_batches then
27:              break  # 达到采样上限
28:          end if
29:       end while
30:       
31:       # 裁剪到目标批大小
32:       batch_data ← batch_data[:train_batch_size * n]
33:       
34:       # 计算 old_log_prob 和优势
35:       old_log_prob ← π_θ.compute_log_prob(batch_data)
36:       ref_log_prob ← π_ref.compute_log_prob(batch_data)
37:       advantages ← compute_advantage(rewards, ref_log_prob, ...)
38:       
39:       # PPO 更新（带解耦裁剪）
40:       for ppo_epoch in 1..K do
41:           log_prob ← π_θ.compute_log_prob(batch_data)
42:           ratio ← exp(log_prob - old_log_prob)
43:           pg_loss1 ← -advantages * ratio
44:           pg_loss2 ← -advantages * clip(ratio, 1-ε_low, 1+ε_high)
45:           pg_loss ← agg_loss(max(pg_loss1, pg_loss2), loss_agg_mode)
46:           θ ← θ + Adam(∇_θ pg_loss)
47:       end for
48:   end for
49: end for
```

