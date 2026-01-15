# AlphaDrive 算法核心函数描述

# FUNCTION: plan_speed_reward

## 功能概述
计算速度规划决策的奖励分数，综合考虑准确性（F1 分数）、决策复杂度和多样性因子。

## 函数签名
```python
def plan_speed_reward(completions, solution, diversity_weight=0.4, complexity_weights=None, **kwargs)
```

## 输入参数
- `completions`: 模型生成的响应列表，每个元素为包含 `content` 字段的字典
- `solution`: ground truth 答案列表，包含 `<answer>` 标签
- `diversity_weight`: 多样性因子权重，默认 0.4
- `complexity_weights`: 各决策动作的复杂度权重字典，默认为：
  ```python
  {
      "ACCELERATE": 0.9,
      "DECELERATE": 1.0,
      "STOP": 1.0,
      "KEEP": 0.8
  }
  ```

## 详细描述

函数评估自动驾驶速度规划决策质量的奖励机制，从三个维度打分。首先是准确性，通过对比预测决策与标准答案，计算F1分数。其次是复杂度，不同决策有不同权重，需要对预测的所有决策动作的复杂度权重求平均。最后是多样性，当所有决策动作都是首次出现，则获得奖励，重复模式则受到惩罚。最终的奖励分数是准确性与复杂度的乘积，再加上多样性。

## 输出
- `rewards`: 浮点数列表，每个元素对应一个 completion 的奖励分数
 
## 函数实现
code/src/r1-v/src/open_r1/grpo.py:line 54-109

## 测试代码
code/test_code/test_plan_speed_reward.py


---

# FUNCTION: plan_path_reward

## 功能概述
计算路径规划决策的奖励分数，综合考虑准确性（F1 分数）、决策复杂度和多样性因子。

## 函数签名
```python
def plan_path_reward(completions, solution, diversity_weight=0.4, complexity_weights=None, **kwargs)
```

## 输入参数
- `completions`: 模型生成的响应列表，每个元素为包含 `content` 字段的字典
- `solution`: ground truth 答案列表，包含 `<answer>` 标签
- `diversity_weight`: 多样性因子权重，默认 0.4
- `complexity_weights`: 各决策动作的复杂度权重字典，默认为：
  ```python
  {
      "LEFT_TURN": 1.0,
      "RIGHT_TURN": 1.0,
      "LEFT_CHANGE": 1.0,
      "RIGHT_CHANGE": 1.0,
      "STRAIGHT": 0.8
  }
  ```

## 详细描述

函数评估自动驾驶路径规划决策质量的奖励机制，从三个维度打分。首先是准确性，通过对比预测决策与标准答案，计算F1分数。其次是复杂度，不同决策有不同权重，需要对预测的所有决策动作的复杂度权重求平均。最后是多样性，当所有决策动作都是首次出现，则获得奖励，重复模式则受到惩罚。最终的奖励分数是准确性与复杂度的乘积，再加上多样性。

## 输出
- `rewards`: 浮点数列表，每个元素对应一个 completion 的奖励分数

## 函数实现
code/src/r1-v/src/open_r1/grpo.py:line 112-168

## 测试代码
code/test_code/test_plan_path_reward.py


---

# FUNCTION: plan_format_reward

## 功能概述
验证模型输出是否符合预期的结构化格式：`<think>...</think><answer>...</answer>`。

## 函数签名
```python
def plan_format_reward(completions, **kwargs)
```

## 输入参数
- `completions`: 模型生成的响应列表，每个元素为包含 `content` 字段的字典

## 详细描述

函数实现了一个格式检查奖励机制。它验证模型输出是否严格遵循 <think>思考内容</think><answer>答案内容</answer> 的格式规范。标签之间可以有空白字符，标签内容可以是任意字符。如果输出完全符合这个格式，给予满分奖励（1.0），否则不给分（0.0）。

## 输出
- `rewards`: 浮点数列表，每个元素为 1.0（格式正确）或 0.0（格式错误）

## 函数实现
code/src/r1-v/src/open_r1/grpo.py:line 171-178

## 测试代码
code/test_code/test_plan_format_reward.py


---

# FUNCTION: Qwen2VLGRPOTrainer.compute_loss

## 功能概述
GRPO 训练的核心函数，计算策略优化损失。该函数处理多模态输入、生成响应、计算奖励、估计优势并更新策略。

## 函数签名
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
```

## 输入参数
- `model`: 策略模型（VLM）
- `inputs`: 批次数据，包含 `prompt`（对话）、`image`（图像路径）等字段
- `return_outputs`: 是否返回输出（GRPO 不支持，必须为 False）
- `num_items_in_batch`: 批次中的样本数量（可选）

## 详细描述
训练流程实现了基于多模态输入的 GRPO 强化学习训练。首先加载驾驶场景的图像和文本描述，将其转换为模型输入格式。对每个场景生成多个候选驾驶决策响应，并处理生成序列的结束符掩码。

计算当前模型和参考模型对生成内容的对数概率，用于后续的 KL 散度约束。将生成的决策文本解码后，调用配置的奖励评估函数进行打分。奖励函数分为两类：预训练的奖励模型（直接对提示词和响应的组合进行评分）和自定义评估函数（需要接收输入数据中的额外字段，如场景参数、标准答案、复杂度权重等）。

汇总所有奖励函数的分数后，使用 GRPO 的组内标准化方法计算相对优势值。

最终将每个 token 的对数概率乘以对应的优势值，减去 KL 散度惩罚项，对生成部分的所有 token 求平均。同时记录生成长度、各奖励函数分数、总奖励、奖励标准差和 KL 散度等训练指标。

## 输出
- `loss`: 标量张量，用于反向传播和参数更新

## 函数实现
code/src/r1-v/src/open_r1/trainer/grpo_trainer.py:line 363-500

## 测试代码
code/test_code/test_compute_loss.py


