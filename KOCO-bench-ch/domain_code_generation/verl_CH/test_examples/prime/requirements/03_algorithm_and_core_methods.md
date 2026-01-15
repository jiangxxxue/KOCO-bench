# PRIME算法核心函数描述

# FUNCTION:DataParallelPRIMERewardModel._forward_micro_batch

## 功能概述
计算隐式过程奖励模型的token级奖励分数

## 函数签名
def _forward_micro_batch(self, micro_batch, prompt_length):

## 输入参数
- `micro_batch`: 微批次数据 {input_ids, attention_mask, position_ids, acc}
- `prompt_length`: 提示序列长度

## 详细描述
首先根据是否启用填充移除优化选择计算路径。对于奖励模型和参考模型，分别进行前向传播计算输出logits，然后提取对应token的对数概率。
隐式奖励通过计算两个模型在响应序列上的对数概率差异获得。
过程奖励生成支持两种模式：直接模式、GAE模式。
在token粒度下每个位置获得对应奖励值，在whole粒度下所有奖励累积到最后一个有效token位置。

## 输出
- `token_level_scores`: token级奖励分数 (torch.Tensor, shape: [batch_size, seq_len])
- `q_values`: 原始对数概率差异 (torch.Tensor, shape: [batch_size, seq_len])

## 函数实现
code/recipe/prime/prime_dp_rm.py:line 51-228

## 测试代码
code/test_code/test_forward_micro_batch.py

---

# FUNCTION:compute_ce_dpo_loss_rm

## 功能概述
计算交叉熵风格的DPO损失

## 函数签名
def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):

## 输入参数
- `token_level_scores`: token级奖励
- `acc`: 二元准确率标签
- `response_mask`: 响应掩码  
- `beta`: 温度参数

## 详细描述
通过计算序列级分数，将其转化为概率，最终计算交叉熵损失。

## 输出
标量损失值

## 函数实现
code/recipe/prime/prime_core_algos.py:line 82-85

## 测试代码
code/test_code/test_ce_dpo_loss.py

---

# FUNCTION:compute_detach_dpo_loss_rm

## 功能概述
计算分离式DPO损失，支持BoN（Best-of-N）模式

## 函数签名
def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):

## 输入参数
- 基础: `token_level_scores`, `acc`, `response_mask`, `beta`
- 对比: `Q_bc` (批次Q值矩阵), `acc_bc` (批次准确率矩阵)  
- `bon_mode`: 权重模式 {"none", "bon_rm", "bon_acc"}

## 详细描述

计算基于对比样本的分离式DPO损失，通过在批次内选择相反准确率标签的样本作为对比基准，并支持通过BoN权重机制为高质量样本分配更高的训练权重.

BoN权重包括下述三种模式：none模式、bon_rm模式、bon_acc模式。

## 输出
加权DPO损失值

## 函数实现
code/recipe/prime/prime_core_algos.py:line 88-116

## 测试代码
code/test_code/test_detach_dpo_loss.py

---

# FUNCTION:compute_dpo_accuracy

## 功能概述
计算成对比较的DPO准确率

## 函数签名
def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):

## 输入参数
- `token_level_scores`: token级奖励分数
- `acc`: 准确率标签
- `response_mask`: 响应掩码
- `n_samples`: 每组采样数量

## 详细描述
采用成对比较评估奖励模型准确性。将token级分数聚合为序列级分数，构建样本对进行比较。通过计算分数差异和准确率差异来判断方向一致性。
使用加权平均计算准确率，准确率差异大的样本对权重更高。所有样本准确率相同时返回0.5。

## 输出
加权成对比较准确率

## 函数实现
code/recipe/prime/prime_core_algos.py:line 119-143

## 测试代码
code/test_code/test_dpo_accuracy.py

---

# FUNCTION:compute_dpo_abs_accuracy

## 功能概述
计算绝对准确率指标

## 函数签名
def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):

## 输入参数
- `token_level_scores`: token级奖励分数
- `acc`: 准确率标签
- `response_mask`: 响应掩码

## 详细描述
将聚合的token级分数转换为预测标签，与真实标签比较计算绝对准确率。

## 输出
绝对分类准确率

## 函数实现
code/recipe/prime/prime_core_algos.py:line 146-147

## 测试代码
code/test_code/test_dpo_abs_accuracy.py

---

# FUNCTION:compute_rloo_advantage_return

## 功能概述
实现RLOO优势估计，支持多种奖励源融合

## 函数签名
def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):

## 输入参数
- `data`: 数据协议对象，包含DataProto.batch字段
  - `data.batch["rm_scores"]`: 隐式奖励分数
  - `data.batch["acc"]`: 二元准确率标签
  - `data.batch["prompts"]`: 输入提示序列
  - `data.batch["attention_mask"]`: 注意力掩码
- `response_mask`: 响应掩码，指示有效token位置
- `n_samples`: 每组采样数量，用于RLOO分组计算
- `config`: 奖励权重配置，分层配置对象
  - `config.algorithm.reward_dpo_coef`: DPO奖励权重系数，默认5.0
  - `config.algorithm.reward_gt_coef`: 准确率奖励权重系数，默认5.0

## 详细描述
函数支持多源奖励融合的优势计算。奖励源包括奖励模型分数、准确率奖励。
按n_samples对同一prompt生成的多个response间隔分组。
使用RLOO对每个奖励源计算优势函数并融合多个奖励源，融合奖励后计算累积返回值并白化处理.

## 输出
- `advantages`: 白化优势函数 (torch.Tensor, shape: [batch_size, seq_len])
- `returns`: 累积返回值 (torch.Tensor, shape: [batch_size, seq_len])

## 函数实现
code/recipe/prime/prime_core_algos.py:line 21-79

## 测试代码
code/test_code/test_rloo_advantage_return.py

---

# FUNCTION:RayPRIMETrainer.filter_and_downsample

## 功能概述
实现PRIME算法的智能样本筛选和优先级下采样

## 函数签名
def filter_and_downsample(self, scores, batch: DataProto):

## 输入参数
- `scores`: 验证分数列表，二元准确率标签
- `batch`: 数据批次对象
- `config`: 过滤参数配置

## 详细描述
从过采样候选响应中筛选高质量样本。将验证分数重构为二维矩阵，进行两重过滤：准确率过滤、长度过滤。
准确率过滤是指保留平均准确率在配置范围内的样本组，长度过滤指移除超过最大长度阈值的样本组。
满足过滤条件的样本组优先保留，最终保留1/oversample_factor比例的样本。若所有组都不满足过滤条件，算法仍会保留相应比例的样本，确保训练继续。

## 输出
经过筛选的高质量数据批次

## 函数实现
code/recipe/prime/prime_ray_trainer.py:line 540-572

## 测试代码
code/test_code/test_filter_and_downsample.py
