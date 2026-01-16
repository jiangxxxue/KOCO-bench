# ARES 算法核心函数描述

# FUNCTION: compute_score

## 功能概述
计算 ARES 的分层熵奖励，根据样本难度、准确率和高窗口熵（HWE）token 数量动态调整奖励信号。


## 函数签名
def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, alpha_length_bonus: float = 0.5, alpha_entropy: float = 0.5) -> List[Dict[str, float]]:

## 输入参数
- `reward_inputs`: 奖励输入字典列表，每个字典包含：
  - `response`: 模型生成的响应文本
  - `ground_truth`: 标准答案
  - `difficulty`: 样本难度标签（"easy" / "medium" / "hard"）
  - `high_entropy_token_num`: 响应中 HWE token 的数量
  - `target_high_entropy_token_num`: 目标 HWE token 数量（难度相关）
  - `alpha_entropy`: 熵奖励系数（难度相关）
  - `accuracy`: 准确率标签（可选，若未提供则从响应中提取并评估）
- `format_weight`: 格式奖励权重，默认 0.1
- `alpha_length_bonus`: 长度奖励系数（保留参数，实际使用 alpha_entropy）
- `alpha_entropy`: 默认熵奖励系数，默认 0.5（若样本未指定则使用）

## 详细描述
该函数实现 ARES 的核心奖励塑形机制，根据任务难度和模型表现自适应调整推理资源分配。函数中计算了格式奖励、准确率奖励以及熵奖励。对于简单问题，当答案正确时会惩罚过长的推理链，而当答案错误时则给予适度的正向奖励以鼓励模型探索更多可能性。中等难度的问题要求推理长度更接近目标值，无论推理过长还是过短都会在答案正确时受到惩罚，但答案错误时仍然鼓励较长的推理。困难问题则采用相反的策略，当答案正确时，会奖励接近或超过目标的推理链，只有推理明显不足时才会施加惩罚，而答案错误时同样鼓励更长的推理来寻找正确路径。

整个奖励机制通过Huber损失函数来平滑处理惩罚项，通过Sigmoid函数来实现奖励的饱和效应，并为不同难度设置了不同的容忍边界和奖励上限。

以下是可能用到的难度相关的超参数：
- `MARGIN_FRAC`：容忍带宽度（easy: 15%, medium: 25%, hard: 35%）
- `KAPPA`：Huber 函数转折点（easy: 2.0, medium: 3.0, hard: 4.0）
- `TEMP`：Sigmoid 温度参数（easy: 2.0, medium: 2.5, hard: 3.0）
- `CAP_SCALE`：熵奖励上限系数（easy: 1.0, medium: 1.0, hard: 1.2）

## 输出
- `scores`: 评分字典列表，每个字典包含：
  - `overall`: 总分 = accuracy + entropy_score
  - `accuracy`: 准确率分数（0.0 或 1.0）
  - `format`: 格式分数（0.0 或 1.0）
  - `high_entropy_token_num_score`: 熵奖励分数

## 函数实现
code/examples/reward_function/aepo.py:line 58-133

## 测试代码
code/test_code/test_compute_score.py

---

# FUNCTION: RayPPOTrainer.update_difficulty_and_skip_gid_set

## 功能概述
根据当前 batch 的 rollout 结果更新样本难度字典和自适应熵系数，实现在线难度分桶和资源预算调整。

## 函数签名
def update_difficulty_and_skip_gid_set(self, batch: DataProto) -> None:

## 输入参数
- `batch`: DataProto 对象，包含：
  - `global_id`: 样本全局 ID（用于跨 batch 跟踪）
  - `accuracy`: 准确率列表（每个响应的准确率）
  - `entropies`: 平均熵列表（每个响应的平均 token 熵）
  - `high_entropy_token_num`: HWE token 数量列表（每个响应的 HWE token 数）

## 详细描述
该函数负责在训练过程中动态更新问题难度分类和自适应调整熵惩罚系数。函数需要按照全局标识符聚合每个问题的准确率、熵和高熵token数量。根据准确率将问题划分为简单、中等和困难三个级别。根据高熵token数量，函数需要计算每个难度级别的目标推理长度，并根据实际生成长度与目标值的偏差来调整熵惩罚系数。调整熵惩罚系数时可以应用梯度上升。对于简单和中等问题，如果推理过长则增大惩罚力度，而对于困难问题则在推理不足时增大鼓励力度。

根据准确率阈值分配难度：
  - `acc_rate ≥ 2/3` → "easy"
  - `1/3 ≤ acc_rate < 2/3` → "medium"
  - `acc_rate < 1/3` → "hard"

（可选）函数将"全部正确且平均熵极低"的样本加入 `self.skip_gid_set`，后续训练将跳过这些过于简单的样本，聚焦于有挑战性的数据。

## 输出
- 无直接返回值，但更新以下类属性：
  - `self.difficulty_dict_train`: 样本难度字典 `{global_id: difficulty}`
  - `self.target_high_entropy_token_num_dict`: 目标 HWE token 数 `{difficulty: target_num}`
  - `self.alpha_entropy_dict`: 熵系数字典 `{difficulty: alpha}`
  - `self.skip_gid_set`: 跳过样本集合（可选）

## 函数实现
code/verl/trainer/ray_trainer.py:line 775-825

## 测试代码
code/test_code/test_update_difficulty.py

---


# FUNCTION: TokenLevelAdaptiveKLController.update

## 功能概述
实现 token 级自适应 KL 控制器的更新逻辑，根据每个难度桶的实际 KL 散度与目标的偏差，使用对偶上升（dual ascent）机制动态调整 KL 系数。

## 函数签名
def update(self, current_kls: Dict[str, float], n_steps: Dict[str, int]):

## 输入参数
- `current_kls`: 当前步每个难度桶的平均 KL 散度，字典格式 `{difficulty: kl_value}`
  - 例如：`{"easy": 0.06, "medium": 0.12, "hard": 0.18}`
- `n_steps`: 每个难度桶的样本数（用于加权，实际实现中可能简化）

## 详细描述
Token 级自适应 KL 控制允许在不同难度任务上使用不同的 KL 预算，并通过对偶上升机制自动调整每个难度桶的 KL 散度系数。当实际 KL 散度超过目标值时增大惩罚系数，低于目标值时减小系数。

## 输出
- 无直接返回值，但更新类属性：
  - `self.kl_coefs`: 更新后的 KL 系数字典 `{difficulty: beta}`
  - `self.lambdas`: 更新后的拉格朗日乘子字典 `{difficulty: lambda}`

## 函数实现
code/verl/trainer/core_algos.py:line 77-86

## 测试代码
code/test_code/test_kl_controller.py

---


# FUNCTION: apply_kl_penalty

## 功能概述
将 KL 散度惩罚应用于 token 级奖励，并更新 KL 控制器。

## 函数签名
def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):

## 输入参数
- `data`: DataProto 对象，包含：
  - `token_level_scores`: 原始 token 级奖励 [batch_size, response_length]
  - `old_log_probs`: Actor 旧策略对数概率 [batch_size, response_length]
  - `ref_log_probs`: Ref 策略对数概率 [batch_size, response_length]
  - `response_mask`: 响应掩码 [batch_size, response_length]
- `kl_ctrl`: KL 控制器（FixedKLController 或 AdaptiveKLController）
- `kl_penalty`: KL 惩罚类型（"kl", "abs", "mse", "low_var_kl", "full"）

## 详细描述
该函数实现 token 级 KL 惩罚的计算和应用，根据 `kl_penalty` 类型计算 token 级 KL 散度，然后从原始奖励中减去 KL 惩罚项得到 token 级别奖励，并求得序列级 KL 散度和 batch 级 KL 散度。

函数中还需要实现更新 KL 控制器。

## 输出
- `data`: 更新后的 DataProto，其中 `data.batch["token_level_rewards"]` 已应用 KL 惩罚
- `metrics`: 指标字典 `{"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}`

## 函数实现
code/verl/trainer/ray_trainer.py:line 107-124

## 测试代码
code/test_code/test_apply_kl_penalty.py
