# CANOE 项目总览（Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning）

## 1. 算法概述

CANOE（Contextual fAithfulness via syNthetic tasks and reinfOrcEment learning）是一个系统性框架，旨在提高大语言模型（LLM）在给定上下文中的忠实度（faithfulness），适用于短文本和长文本生成任务，且无需人工标注。该方法通过合成短文本问答（QA）数据构建高质量、易验证的训练数据，并提出Dual-GRPO强化学习方法，同时优化短文本和长文本响应生成。

核心思想：
1. 合成任务构建：通过四种多样化的任务合成短文本QA数据，包括反事实QA（Counterfactual QA）和事实QA（Factual QA），构建可自动验证的训练数据。
2. Dual-GRPO训练：基于GRPO（Group Relative Policy Optimization）算法，设计三种基于规则的奖励函数，同时优化短文本和长文本生成，避免过度优化短文本生成。
3. 多任务奖励融合：结合准确性奖励（accuracy reward）、格式奖励（format reward）、影响力奖励（influence reward）和长度奖励（length reward），全面评估模型生成质量。

## 2. 训练 Pipeline（伪代码）

```
Algorithm: CANOE Training Pipeline (Dual-GRPO)
Input: Dataset D (包含 context 和 question), Policy π_θ, Reference π_ref
Config: num_generations n, reward_funcs=[accuracy, format, influence, length],
        temperature, max_completion_length, do_long_answer=True

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       # 第一阶段：生成短文本答案
4:       prompts ← format_prompts(B, SYSTEM_PROMPT)  # 包含 <context>, <think>, <long_answer>, <answer> 标签
5:       completions ← generate(π_θ, prompts, n)  # 生成 n 个响应
6:       
7:       # 计算短文本奖励
8:       R_accuracy ← accuracy_reward(completions, B.solution)  # 符号验证或字符串匹配
9:       R_format ← format_reward(completions)  # 检查 <think><long_answer><answer> 格式
10:      
11:      # 第二阶段：生成长文本答案（如果启用）
12:      if do_long_answer then
13:          # 提取 <long_answer> 内容替换原始 context
14:          long_answers ← extract_long_answer(completions)
15:          new_contexts ← replace_context(B, long_answers)
16:          completions_long ← generate(π_θ, new_contexts, 1)  # 每个样本生成1个长答案
17:          
18:          # 计算长文本奖励
19:          R_influence ← influence_reward(completions_long, B.solution)  # 长答案准确性
20:          R_length ← length_reward(completions, B.context)  # 长度约束：0.2 < len(long_answer)/len(context) < 0.8
21:      else
22:          R_influence ← 0
23:          R_length ← 0
24:      
25:      # 融合所有奖励
26:      R_total ← R_accuracy + R_format + R_influence + R_length
27:      
28:      # 计算 log probabilities 和 KL 散度
29:      logps ← compute_logps(π_θ, completions)
30:      ref_logps ← compute_logps(π_ref, completions)
31:      KL ← compute_kl(logps, ref_logps)
32:      
33:      # GRPO 优势估计（组内相对优势）
34:      A ← compute_grpo_advantage(R_total, grouping_by_prompt)
35:      A ← whiten(A)  # 标准化
36:      
37:      # 策略梯度更新
38:      L ← -A * logps + β * KL  # β 控制 KL 惩罚
39:      θ ← θ - Adam(∇_θ L)
40:   end for
41: end for
```

要点：
- 双阶段生成：先生成包含短答案的完整响应，再基于长答案重新生成以评估影响力；
- 多奖励融合：四种奖励函数协同工作，确保生成质量、格式规范和上下文忠实度；
- GRPO优势估计：按同一prompt的n个响应分组，计算组内相对优势，降低方差；
- 长度约束：确保长答案长度在合理范围内（20%-80%的上下文长度），避免过长或过短。
