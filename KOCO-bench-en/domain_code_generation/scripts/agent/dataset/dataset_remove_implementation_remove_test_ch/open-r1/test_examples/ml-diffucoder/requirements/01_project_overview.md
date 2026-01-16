# DiffuCoder 项目总览（Masked Diffusion Models for Code Generation）

## 1. 算法概述

DiffuCoder 是一个基于掩码扩散模型（Masked Diffusion Models）的代码生成系统，采用"Coupled-GRPO"后训练方法提升代码生成性能。与传统自回归（AR）模型严格从左到右生成不同，扩散语言模型（dLLM）通过迭代去噪过程生成文本，在保持性能的同时提供更快的生成速度。

核心思想：DiffuCoder 基于掩码去噪模型（MDM）架构，通过前向扩散过程将完整序列逐步添加噪声（掩码），再通过反向去噪过程迭代恢复原始序列。在后训练阶段，Coupled-GRPO 通过配对采样方案解决扩散模型中单步损失计算仅在掩码位置的低效问题，确保每个 token 在至少一次前向传播中被计算，从而降低方差并提高训练效率。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: Coupled-GRPO Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, Reward Functions R
Config: num_generations k, num_iterations m, λ pairs, 
        diffusion_steps T, mask_ratio ∈ [0.2, 0.8]

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 生成 k 个响应（扩散生成）
4:       responses ← diffusion_generate(π_θ, B, k, T)
5:       
6:       # 计算奖励（代码执行 + 格式检查）
7:       rewards ← R.compute(B, responses)
8:       
9:       # 计算 leave-one-out 优势
10:      for each prompt p in B do
11:          baseline_p ← (sum(rewards_p) - rewards_p[i]) / (k-1)
12:          advantages_p[i] ← rewards_p[i] - baseline_p
13:      end for
14:      
15:      # Coupled sampling: 为每个迭代生成配对掩码
16:      for iteration i in 1..m do
17:          if random_masking then
18:              seed_i ← random_seed()
19:          else
20:              seed_i ← fixed_seed
21:          
22:          # 生成三个版本的掩码序列
23:          mask_ratio ← uniform(0.2, 0.8)
24:          
25:          # Version 1: 完全掩码所有完成 token
26:          seq_1 ← mask_all_completion_tokens(responses)
27:          
28:          # Version 2: 按 mask_ratio 随机掩码
29:          seq_2 ← random_mask(responses, mask_ratio, seed_i)
30:          
31:          # Version 3: 反向掩码（掩码 Version 2 未掩码的部分）
32:          seq_3 ← reverse_mask(responses, mask_ratio, seed_i)
33:          
34:          # 计算加权对数概率
35:          logits_1, logits_2, logits_3 ← π_θ.forward([seq_1, seq_2, seq_3])
36:          
37:          # 对每个 token，根据其在 seq_2 或 seq_3 中的掩码状态加权
38:          weights ← [1, 1/mask_ratio, 1/(1-mask_ratio)]
39:          per_token_logps ← selective_log_softmax(
40:              [logits_1, logits_2, logits_3], 
41:              responses, 
42:              weights, 
43:              mask_indicator
44:          )
45:          
46:          # 计算 KL 散度（如果使用参考模型）
47:          if β > 0 then
48:              ref_logps ← π_ref.forward([seq_1, seq_2, seq_3])
49:              kl_penalty ← KL(per_token_logps, ref_logps)
50:          
51:          # PPO 风格的裁剪损失
52:          ratio ← exp(per_token_logps - old_logps)
53:          clipped_ratio ← clip(ratio, 1-ε, 1+ε)
54:          loss ← -min(ratio * advantages, clipped_ratio * advantages)
55:          if β > 0 then
56:              loss ← loss + β * kl_penalty
57:          
58:          # 更新策略
59:          θ ← θ - Adam(∇_θ loss)
60:      end for
61:   end for
62: end for
```

要点：
- **Coupled Sampling**：通过配对掩码（互补掩码）确保每个 token 在至少一个前向传播中被计算，降低方差；
- **加权概率**：根据掩码比例对不同版本的对数概率进行加权，平衡不同掩码模式的贡献；
- **Leave-one-out 基线**：按同一 prompt 的 k 个响应为一组，计算 leave-one-out 基线降低方差；
- **随机掩码**：在多次迭代中使用随机掩码种子，增强训练鲁棒性；
- **扩散生成**：使用迭代去噪过程生成代码，支持非严格从左到右的生成模式。
