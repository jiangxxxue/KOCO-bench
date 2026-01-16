# LUFFY 算法核心函数描述

# FUNCTION: compute_grpo_outcome_advantage_split

## 功能概述
计算分组 GRPO（Group Relative Policy Optimization）的优势估计，关键创新在于仅使用 on-policy 样本计算组内基线。

## 函数签名
def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):

## 输入参数
- `token_level_rewards`: token 级奖励张量，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `eos_mask`: 有效 token 掩码，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `index`: 样本分组索引，同一 prompt 的 n 个响应共享相同索引。**注意：此参数应为 Python 列表或可索引序列（如 `[0, 0, 1, 1]`），而非 `torch.Tensor`，因为索引值会被用作字典的键**
- `on_policy_mask`: 布尔掩码，形状为 `[batch_size]`，类型为 `torch.Tensor`，标记哪些样本是 on-policy 生成的
- `epsilon`: 数值稳定性常数，默认为 1e-6
- `use_std`: 是否使用标准差进行归一化，默认为 True

## 详细描述
该函数需要从 token 级奖励张量，对每个响应序列沿时间维度求和，得到序列级总分数。

主要实现对同一 prompt 的多个响应进行分组归一化（包括 on-policy 和 off-policy），计算相对优势值。归一化时仅选取标记为 on-policy 的样本来计算该组的均值和标准差。若某组仅有一个 on-policy 样本，则将均值设为 0、标准差设为 1；若有多个样本，则正常计算统计量。对于标准差为 0 的情况，将其替换为 1 以避免除零错误。

若启用标准差归一化，则计算 (score - mean) / (std + epsilon)；否则仅减去均值。最终需要将序列级优势值扩展为token级优势张量。

## 输出
- `advantages`: token 级优势张量，形状为 `[batch_size, response_length]`
- `returns`: token 级回报张量，形状为 `[batch_size, response_length]`，与优势相同

## 函数实现
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 12-67

## 测试代码
code/test_code/test_compute_grpo_outcome_advantage_split.py
---

# FUNCTION: compute_token_on_off_policy_loss

## 功能概述
计算混合策略损失，结合 on-policy 的 PPO clip 损失和 off-policy 的重要性采样损失，并支持多种策略塑形（policy shaping）方法来强调低概率但关键的动作。

## 函数签名
def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False, 
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):

## 输入参数
- `old_log_prob`: 旧策略的 log 概率，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `log_prob`: 当前策略的 log 概率，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `advantages`: 优势估计，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `eos_mask`: 有效 token 掩码，形状为 `[batch_size, response_length]`，类型为 `torch.Tensor`
- `cliprange`: PPO clip 范围，默认为 0.2
- `clip_upper_bound`: PPO clip 上界，默认为 100.0
- `prefix_mask`: 标记 off-policy token 的掩码，形状为 `[batch_size, response_length]`，**类型必须为 `torch.Tensor` 且 dtype 为 `torch.bool`，因为代码中会使用按位取反操作符 `~prefix_mask` 来获取 on-policy token 的掩码**
- `off_cliprange`: off-policy clip 范围（当前实现中未使用）
- `off_normalize`: 是否归一化 off-policy ratio，默认为 False
- `off_abs_cliprange`: off-policy 绝对值 clip 范围（当前实现中未使用）
- `off_max_clip`: off-policy ratio 上限，None 表示不限制
- `off_min_clip`: off-policy ratio 下限，None 表示不限制
- `all_max_clip`: 全局概率上限，None 表示不限制
- `off_policy_reshape`: off-policy 策略塑形方法名称，默认为 'no_reshape'
- `off_policy_reshape_weight`: 策略塑形权重参数，默认为 1.0
- `off_policy_reshape_pow_exp`: 幂次变换指数，默认为 0.5
- `on_policy_reshape`: on-policy 策略塑形方法名称，默认为 'no_reshape'
- `on_policy_reshape_weight`: on-policy 策略塑形权重，默认为 1.0
- `on_policy_reshape_pow_exp`: on-policy 幂次变换指数，默认为 0.5
- `target_probs`: 可选的目标概率（用于显式行为策略），默认为 None
- `loss_remove_token_mean`: 是否移除 token 级平均，默认为 False
- `loss_remove_clip`: 是否移除 clip 操作，默认为 False

## 详细描述
该函数计算混合策略梯度损失，核心思想是同时处理 on-policy 和off-policy 两种数据。对于 on-policy 数据，它使用标准的 PPO 算法计算重要性采样比率并进行裁剪；对于 off-policy 数据，它直接使用当前策略的概率（或相对于目标概率的比率）来计算损失，不依赖旧策略。

函数通过 prefix_mask 区分哪些 token 是off-policy的，哪些是on-policy的，然后将两种损失加权组合。此外，函数应当支持多种概率重塑（reshape）策略和裁剪机制。

## 输出
返回字典包含以下键值对：
- `pg_loss`: 总策略梯度损失（标量）
- `off_pg_loss`: off-policy 损失分量（标量）
- `on_pg_loss`: on-policy 损失分量（标量）
- `off_pg_clipfrac`: off-policy clip 比例（标量，当前实现中为 0）
- `on_pg_clipfrac`: on-policy clip 比例（标量）
- `ppo_kl`: PPO KL 散度（标量）
- `off_policy_prob`: off-policy 样本的平均概率（标量）
- `on_policy_prob`: on-policy 样本的平均概率（标量）
- `off_ratio_mean`: off-policy 平均 ratio（标量）
- `off_ratio_max_clip_frac`: 达到上限裁剪的比例（标量）
- `off_ratio_min_clip_frac`: 达到下限裁剪的比例（标量）

## 函数实现
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 69-251

## 测试代码
code/test_code/test_compute_token_on_off_policy_loss.py

---

# FUNCTION: compute_sft_pure_loss

## 函数签名
def compute_sft_pure_loss(log_prob, eos_mask):

## 功能概述
计算纯 SFT（Supervised Fine-Tuning）损失，即负对数似然损失，用于多任务学习中对 off-policy 样本进行监督学习。

## 输入参数
- `log_prob`: 模型预测的 log 概率，形状为 `[batch_size, response_length]`
- `eos_mask`: 有效 token 掩码，形状为 `[batch_size, response_length]`

## 详细描述
计算每个 token 的负 log 概率作为损失值。使用 eos_mask 对损失进行掩码，仅在有效 token 位置计算损失。对掩码后的损失取平均，得到批次级的 SFT 损失。

## 输出
- `sft_loss`: 标量，表示平均 SFT 损失

## 函数实现
code/luffy/verl/verl/mix_src/mix_core_alg.py:line 7-10

## 测试代码
code/test_code/test_compute_sft_pure_loss.py
