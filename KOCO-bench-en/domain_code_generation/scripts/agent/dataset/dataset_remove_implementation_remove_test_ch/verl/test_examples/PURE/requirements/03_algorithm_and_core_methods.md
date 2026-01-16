# PURE算法核心函数描述

# FUNCTION:ProcessRewardModelWorker._forward_micro_batch

## 功能概述
计算过程奖励模型（PRM）的 token 级分数，并根据配置执行最小式（min-form）信用分配的近似加权。

## 函数签名
def _forward_micro_batch(self, micro_batch):

## 输入参数
- `micro_batch`: 微批数据，包含 `input_ids`, `attention_mask`, `position_ids`, `responses`, `reward_mask` 等
- `response_length`: 响应序列长度（由批内张量推导）
- `temperature`: 近似 min-form 的温度（当 `reward_model.credit_assignment` 为浮点数时）

## 详细描述
对过程奖励模型(PRM)执行前向传播，通过softmax计算正负类概率差值得到token级分数。使用掩码仅保留响应部分的有效分数。
若启用去填充或序列并行优化，需先移除填充再进行计算，完成后复原。
信用分配加权根据配置模式执行不同策略：min-form、strict min-form、gamma-dacay. 低分token应当获得更高权重。
   
## 输出
- `rm_score`: token 级过程奖励分数（形状 `[batch_size, response_length]`）

## 函数实现
code/verl/workers/fsdp_workers.py:line 1462-1539

## 测试代码
code/test_code/test_forward_micro_batch.py


---

# FUNCTION:ProcessRewardModelWorker.compute_rm_score

## 功能概述
批量构建 PRM 输入（按步骤切分并在每步后追加换行），进行推断与（可选）变长分片/回填，聚合得到批级 token 级 `rm_scores`。

## 函数签名
def compute_rm_score(self, data:DataProto):

## 输入参数
- `data`: `DataProto`，包含 `responses`, `reward_mask` 等；

## 详细描述
对响应序列按步骤切分，为过程奖励模型构建输入格式。
动态批次模式下按token总数限制将数据重组为多个小批次并记录原始顺序，固定批次模式下按样本数均匀分批。
对每个小批次调用前向函数计算token级分数，拼接所有批次结果。动态批次模式下根据记录的索引将结果还原为原始顺序。

## 输出
- `DataProto`：`batch['rm_scores']` 形状 `[batch_size, response_length]`

## 函数实现
code/verl/workers/fsdp_workers.py:line 1541-1594

## 测试代码
code/test_code/test_compute_rm_score.py

---

# FUNCTION:compute_return

## 功能概述
根据 `method` 计算 token 级回报序列

## 函数签名
def compute_return(token_level_rewards, eos_mask, method='sum', gamma=1.0):

## 输入参数
- `token_level_rewards`: `[batch_size, response_length]`
- `eos_mask`: `[batch_size, response_length]`
- `method`: `'sum'|'min'`
- `gamma`: 折扣因子（`'sum'` 模式下可用）

## 详细描述
根据method计算token级回报序列,使用掩码确保仅处理有效位置。

“method”包括“sum”和“min”.

## 输出
- `returns`: `[batch_size, response_length]`

## 函数实现
code/verl/trainer/ppo/core_algos.py:line 70-88

## 测试代码
code/test_code/test_compute_return.py

