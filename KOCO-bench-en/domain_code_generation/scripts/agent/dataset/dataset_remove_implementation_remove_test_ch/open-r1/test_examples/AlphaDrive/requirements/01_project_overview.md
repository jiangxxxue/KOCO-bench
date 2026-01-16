# AlphaDrive 项目总览


## 1. 算法概述

AlphaDrive 是首个将基于 GRPO（Group Relative Policy Optimization）的强化学习与规划推理相结合应用于自动驾驶的方法。该方法通过视觉-语言模型（VLM）理解驾驶场景，并生成包含推理过程的驾驶决策，显著提升了规划性能和训练效率。

核心思想：利用 GRPO 算法训练视觉-语言模型，使其能够：
1. 理解多模态驾驶场景（图像 + 导航指令 + 当前速度）
2. 生成结构化的推理过程（在 `<think>` 标签内）
3. 输出可执行的驾驶决策（在 `<answer>` 标签内，包含速度规划和路径规划）

训练过程采用多奖励函数融合机制，包括：
- 格式奖励（format reward）：确保输出符合 `<think>...</think><answer>...</answer>` 格式
- 速度规划奖励（speed reward）：评估速度决策的准确性（KEEP/ACCELERATE/DECELERATE/STOP）
- 路径规划奖励（path reward）：评估路径决策的准确性（STRAIGHT/LEFT_TURN/RIGHT_TURN/LEFT_CHANGE/RIGHT_CHANGE）

## 2. 训练 Pipeline（伪代码）

```
Algorithm: AlphaDrive Training Pipeline
Input: Dataset D (images, navigation, speed, ground truth plans), 
       Policy π_θ (VLM), Reference π_ref, 
       Reward functions R = {R_format, R_speed, R_path}
Config: num_generations G, max_prompt_length, max_completion_length,
        beta (KL penalty coefficient)

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 构建多模态输入（图像 + 文本提示）
4:       prompts ← construct_multimodal_prompts(B)
5:       
6:       # 生成 G 个响应（包含推理过程和决策）
7:       responses ← generate(π_θ, prompts, G)
8:       
9:       # 计算多个奖励函数
10:      R_format ← check_format(responses)  # 检查 <think></think><answer></answer> 格式
11:      R_speed ← evaluate_speed_plan(responses, B.ground_truth_speed)
12:      R_path ← evaluate_path_plan(responses, B.ground_truth_path)
13:      
14:      # 融合奖励
15:      R_total ← R_format + R_speed + R_path
16:      
17:      # 计算分组优势（GRPO）
18:      # 按同一 prompt 的 G 个响应分组
19:      mean_R ← group_mean(R_total, by_prompt)
20:      std_R ← group_std(R_total, by_prompt)
21:      advantages ← (R_total - mean_R) / (std_R + ε)
22:      
23:      # 计算 KL 散度（相对于参考模型）
24:      logp_θ ← log_prob(π_θ, responses)
25:      logp_ref ← log_prob(π_ref, responses)
26:      KL ← compute_kl(logp_θ, logp_ref)
27:      
28:      # 计算损失并更新
29:      loss ← -mean(exp(logp_θ - logp_θ.detach()) * advantages - beta * KL)
30:      θ ← θ + Adam(∇_θ loss)
31:   end for
32: end for
```

要点：
- 多模态输入：结合驾驶场景图像、导航指令和当前速度信息
- 结构化输出：模型生成包含推理过程（`<think>`）和决策（`<answer>`）的结构化响应
- 多奖励融合：同时优化格式正确性、速度规划和路径规划三个维度
- GRPO 优势估计：通过分组相对比较降低方差，提高训练稳定性
- 复杂度加权：不同决策动作具有不同的复杂度权重（如 STOP 比 KEEP 更重要）
- 多样性因子：鼓励模型在批次内生成多样化的决策

