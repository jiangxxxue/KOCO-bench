# VisualQuality-R1 项目总览


## 1. 算法概述

VisualQuality-R1 是首个通过强化学习排序（Reinforcement Learning to Rank, RL2R）增强的无参考图像质量评估（NR-IQA）模型，能够同时进行质量描述和评分推理。该方法基于 open-r1 框架，采用 Group Relative Policy Optimization（GRPO）算法，通过多模态视觉语言模型实现对图像质量的推理式评估。

核心思想：将图像质量评估任务转化为排序学习问题，通过强化学习优化模型对不同质量图像的相对排序能力。模型为每个图像生成 G 个质量评分响应，利用成对比较的保真度奖励（Fidelity Reward）来衡量模型预测的排序关系与真实质量标签的一致性，并通过 GRPO 算法更新策略模型。

训练过程中，策略模型基于 Qwen2.5-VL-7B-Instruct 多模态架构，对输入图像生成包含推理过程（thinking）和最终评分（answer）的结构化响应。通过计算批次内所有图像对之间的排序保真度，模型学习到更准确的质量感知能力。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: VisualQuality-R1 Training Pipeline
Input: Dataset D (images with MOS scores), Policy π_θ, Reference π_ref
Config: num_generations G, beta β, epsilon ε, num_iterations μ

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 生成 G 个质量评分响应（包含推理过程和评分）
4:       responses ← rollout(π_θ, B, G)
5:       
6:       # 提取每个响应的预测评分
7:       predictions ← extract_scores(responses)
8:       
9:       # 计算批次内的统计量（均值和方差）
10:      for each sample i in B do
11:          μ_i ← mean(predictions_i)
12:          σ²_i ← variance(predictions_i)
13:      end for
14:      
15:      # 计算保真度奖励（基于成对排序）
16:      rewards ← []
17:      for each sample i in B do
18:          for each generation j in 1..G do
19:              r_ij ← 0
20:              for each other sample k ≠ i in B do
21:                  # 计算排序保真度
22:                  Δ ← (pred_ij - μ_k) / √(σ²_i + σ²_k + ε)
23:                  p ← Φ(Δ)  # 标准正态分布累积分布函数
24:                  
25:                  # 根据真实 MOS 确定真实排序关系
26:                  if MOS_i > MOS_k then
27:                      gt ← 1.0
28:                  elif MOS_i < MOS_k then
29:                      gt ← 0.0
30:                  else
31:                      gt ← 0.5
32:                  end if
33:                  
34:                  # 保真度奖励
35:                  r_ij += √(p·gt + ε) + √((1-p)·(1-gt) + ε)
36:              end for
37:              r_ij ← r_ij / (|B| - 1)
38:              rewards.append(r_ij)
39:          end for
40:      end for
41:      
42:      # 计算格式奖励（可选）
43:      format_rewards ← check_format(responses)  # 检查 <think></think> 和 <answer></answer> 格式
44:      
45:      # 融合奖励
46:      total_rewards ← rewards + format_rewards
47:      
48:      # 计算分组优势（Group-wise Advantage）
49:      for each group g (same prompt) do
50:          μ_g ← mean(total_rewards_g)
51:          σ_g ← std(total_rewards_g)
52:      end for
53:      advantages ← (total_rewards - μ_g) / (σ_g + ε)
54:      
55:      # GRPO 更新（μ 次迭代）
56:      for iter in 1..μ do
57:          # 计算策略比率
58:          log_π_θ ← log_prob(π_θ, responses)
59:          log_π_old ← log_prob(π_old, responses)
60:          ratio ← exp(log_π_θ - log_π_old)
61:          
62:          # 裁剪目标
63:          L_clip ← min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
64:          
65:          # KL 散度惩罚
66:          if β > 0 then
67:              log_π_ref ← log_prob(π_ref, responses)
68:              KL ← exp(log_π_ref - log_π_θ) - (log_π_ref - log_π_θ) - 1
69:              L ← -L_clip + β·KL
70:          else
71:              L ← -L_clip
72:          end if
73:          
74:          # 更新策略
75:          θ ← θ - ∇_θ L
76:      end for
77:   end for
78: end for
```

要点：
- 排序学习：通过成对比较学习图像质量的相对关系，而非绝对评分；
- 保真度奖励：基于正态分布假设，衡量预测排序与真实排序的一致性；
- 批次内对比：每个样本与批次内所有其他样本进行比较，充分利用批次信息；
- 多次生成：每个图像生成 G 个响应，通过方差估计预测的不确定性；
- GRPO 优化：采用分组相对策略优化，通过裁剪和 KL 惩罚稳定训练。
