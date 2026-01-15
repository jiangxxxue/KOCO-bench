# DAPO 算法核心函数描述


# FUNCTION: agg_loss

## 功能概述

聚合损失矩阵为标量，支持多种聚合模式。DAPO 推荐使用 `token-mean` 模式，使每个 token 对损失的贡献相等，避免长序列主导梯度。

## 函数签名
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):

## 输入参数

- `loss_mat`: 形状 `(batch_size, response_length)` 的损失矩阵
- `loss_mask`: 形状 `(batch_size, response_length)` 的掩码张量（响应部分为 1，padding 为 0）
- `loss_agg_mode`: 字符串，指定聚合模式
  - `"token-mean"`: 所有 token 取平均（**DAPO 推荐**）
  - `"seq-mean-token-sum"`: 先在序列内对 token 求和，再在序列间取平均
  - `"seq-mean-token-mean"`: 先在序列内对 token 取平均，再在序列间取平均
  - `"seq-mean-token-sum"`:  序列求和后用序列长度归一化
## 详细描述

函数的核心功能是将二维损失矩阵聚合成单个标量值，应当支持以下四种聚合策略：token-mean、seq-mean-token-sum、seq-mean-token-mean、seq-mean-token-sum-norm.

## 输出

- 标量损失值，用于反向传播

## 函数实现
code/verl/trainer/ppo/core_algos.py:line 704-737


## 测试代码
code/test_code/test_agg_loss.py

---

# FUNCTION: compute_policy_loss_vanilla

## 功能概述

计算 PPO 策略梯度损失，支持解耦裁剪（DAPO 的核心创新）。通过 `clip_ratio_low` 和 `clip_ratio_high` 两个参数，允许对正负优势进行非对称裁剪，实现"Clip-Higher"机制。

## 函数签名
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

## 输入参数

- `old_log_prob`: 形状 `(batch_size, response_length)` 的旧策略 log 概率
- `log_prob`: 形状 `(batch_size, response_length)` 的当前策略 log 概率
- `advantages`: 形状 `(batch_size, response_length)` 的优势估计
- `response_mask`: 形状 `(batch_size, response_length)` 的响应掩码
- `loss_agg_mode`: 损失聚合模式（默认为 token-mean）
- `config`: Actor 配置对象，包含：
  - `clip_ratio`: 默认裁剪参数（向后兼容）
  - `clip_ratio_low`: 下界裁剪参数（**DAPO 新增**）
  - `clip_ratio_high`: 上界裁剪参数（**DAPO 新增**）
  - `clip_ratio_c`: 双裁剪下界（处理极端负优势，默认 3.0）
- `rollout_log_probs`: 可选，Rollout 策略的 log 概率（用于 off-policy 训练）

## 详细描述

函数实现了增强版的PPO策略损失计算，包括标准PPO裁剪损失计算、双裁剪PPO以及截断重要性采样（可选），最终需要调用相关函数实现损失聚合。

标准PPO裁剪机制支持不对称裁剪，需要计算未裁剪的原始策略梯度损失和不对称裁剪的损失，取两者的最大值作为悲观更新策略。当优势值为负时，应用双裁剪以更保守地降低该动作的概率。如果启用了截断重要性采样且提供了rollout策略的对数概率，函数会计算旧策略相对于rollout策略的重要性比率，将其裁剪到配置的上限，然后乘到策略梯度损失上。

## 输出

返回四个标量张量：
- `pg_loss`: 策略梯度损失
- `pg_clipfrac`: 裁剪比例（被裁剪的 token 占比，用于监控训练动态）
- `ppo_kl`: 近似 KL 散度
- `pg_clipfrac_lower`: 下界裁剪比例（双裁剪生效的 token 占比）

## 函数实现

code/verl/trainer/ppo/core_algos.py:line 816-902

## 测试代码
code/test_code/test_compute_policy_loss_vanilla.py



---

# FUNCTION: DAPORewardManager.__call__

## 功能概述

DAPO 的奖励管理器，负责计算任务奖励并施加超长惩罚。这是 DAPO 区别于其他方法的关键组件之一，通过线性惩罚机制防止模型生成无意义的冗长输出。

## 函数签名
def __call__(self, data: DataProto, return_dict: bool = False):

## 输入参数

- `data`: DataProto 对象，包含：
  - `batch["prompts"]`: 形状 `(bs, prompt_len)` 的 prompt token IDs
  - `batch["responses"]`: 形状 `(bs, response_len)` 的 response token IDs
  - `batch["attention_mask"]`: 形状 `(bs, total_len)` 的注意力掩码
  - `non_tensor_batch["reward_model"]["ground_truth"]`: 真实答案
  - `non_tensor_batch["data_source"]`: 数据来源（如 "gsm8k", "math"）
- `return_dict`: 布尔值，是否返回字典格式（包含额外信息如准确率、超长惩罚等）

## 详细描述

函数负责为生成的响应计算奖励分数。首先检查是否已有预计算的奖励，如果有则直接返回，否则逐个处理每条响应，将token序列解码成文本后根据标准答案计算任务奖励（需要调用相关功能函数）。计算出的分数会被放置在响应序列的最后一个有效token位置。如果启用了长度控制，函数会对超长响应施加额外的负奖励惩罚（线性惩罚机制）。整个过程中会输出部分样本的详细信息用于调试验证，最终返回完整的奖励张量或包含额外评估指标的字典。


## 输出

- 若 `return_dict=False`：返回形状为 `(batch_size, response_length)` 的稀疏奖励张量（仅最后一个有效 token 位置非零）
- 若 `return_dict=True`：返回字典，包含：
  - `"reward_tensor"`: 如上的奖励张量
  - `"reward_extra_info"`: 字典，包含 `"acc"`（准确率列表）、`"overlong_reward"`（超长惩罚列表）、`"overlong"`（是否超长的布尔列表）等

## 函数实现
code/verl/workers/reward_manager/dapo.py:line 53-149


## 测试代码
code/test_code/test_dapo_reward_manager.py


