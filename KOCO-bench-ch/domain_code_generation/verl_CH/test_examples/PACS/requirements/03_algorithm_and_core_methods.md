# PACS 算法核心函数描述

# FUNCTION: compute_reward

## 功能概述
计算策略改进的"奖励"，用于衡量当前策略相对于参考策略的对数概率变化。

## 函数签名
def compute_reward(
    old_log_prob,
    log_prob,
    response_mask,
    beta=1.0,
    reward_method="1",
):

## 输入参数
- `old_log_prob`: 参考策略的 token 级对数概率，形状 `[batch_size, response_length]`
- `log_prob`: 当前策略的 token 级对数概率，形状 `[batch_size, response_length]`
- `response_mask`: 响应部分的掩码，形状 `[batch_size, response_length]`
- `beta`: 奖励缩放因子，控制策略更新激进程度（默认 1.0）
- `reward_method`: 奖励计算方法，'1' 或 '2'

## 详细描述
函数根据指定的方法计算强化学习中的奖励值。
方法 1 计算新旧对数概率差的总和，衡量策略改进幅度。
方法 2 计算归一化的当前对数概率总和。
如果指定了无效的方法编号，函数会抛出异常。

## 输出
- `reward`: 策略改进奖励，形状 `[batch_size, 1]`

## 函数实现
code/src/pacs/pacs_core_algos.py:line 7-21

## 测试代码
code/test_code/test_compute_reward.py

---

# FUNCTION: compute_weight

## 功能概述
计算样本权重，用于处理类别不平衡问题（正负样本比例不均）。

## 函数签名
def compute_weight(score, rollout_n, mode="question"):

## 输入参数
- `score`: 可验证奖励（标签），形状 `[batch_size, 1]`
- `rollout_n`: 每个 prompt 生成的响应数量
- `mode`: 权重计算模式，'question'/'only_positive'/'only_negative'

## 详细描述
函数根据不同模式为样本分配训练权重。在 "question" 模式下，函数按 rollout_n 大小将样本分组，对每组内标签不平衡的情况使用类别平衡权重，使正负样本对训练的贡献相等。在 "only_positive" 模式下，只有正样本（标签为 1）被赋予权重 1.0，负样本权重为 0。在 "only_negative" 模式下则相反，只有负样本（标签为 0）被赋予权重 1.0。如果指定了无效的模式，函数会抛出异常。

## 输出
- `weight`: 样本权重，形状 `[batch_size, 1]`

## 函数实现
code/src/pacs/pacs_core_algos.py:line 59-86

## 测试代码
code/test_code/test_compute_weight.py

---

# FUNCTION: compute_pacs_loss

## 功能概述
计算 PACS 的核心损失函数，将优势函数作为 logit，可验证奖励作为标签，使用二元交叉熵损失。

## 函数签名
def compute_pacs_loss(
    prompts,
    old_log_prob,
    log_prob,
    token_level_scores,
    response_mask,
    rollout_n,
    algorithm_config=None,
):

## 输入参数
- `prompts`: 输入提示，形状 `[batch_size, prompt_length]`
- `old_log_prob`: 参考策略的对数概率，形状 `[batch_size, response_length]`
- `log_prob`: 当前策略的对数概率，形状 `[batch_size, response_length]`
- `token_level_scores`: 可验证奖励（标签），形状 `[batch_size, response_length]`
- `response_mask`: 响应掩码，形状 `[batch_size, response_length]`
- `rollout_n`: 每个 prompt 生成的响应数量
- `algorithm_config`: 算法配置对象，包含 `beta`, `reward_method`, `adv_estimator`, `use_weight`, `weight_mode` 等

## 详细描述

函数需要验证每 rollout_n 个样本对应同一个 prompt，在此基础上计算奖励值、优势值（根据指定的优势估计器）。最后，函数将 token 级别的评分求和，使用二元交叉熵损失将优势值作为监督信号、评分作为预测值来计算损失。如果启用样本权重，会根据评分标签计算平衡权重以处理样本不平衡问题。

## 输出
- `pacs_loss`: PACS 损失值，标量

## 函数实现
code/src/pacs/pacs_core_algos.py:line 89-138

## 测试代码
code/test_code/test_compute_pacs_loss.py
