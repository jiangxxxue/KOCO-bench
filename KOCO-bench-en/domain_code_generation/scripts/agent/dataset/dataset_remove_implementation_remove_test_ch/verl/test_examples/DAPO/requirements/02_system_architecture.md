# DAPO 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（DAPO 核心）

DAPO 在 verl 框架基础上做了精简的扩展，主要新增代码集中在两处：
- `dapo/` 目录：包含 DAPO 专属的训练入口、配置文件和训练器实现
- `verl/workers/reward_manager/dapo.py`：DAPO 专用的奖励管理器

具体目录结构如下：

- **dapo/**：DAPO 专属模块
  - `config/`：训练配置文件，继承 verl 的 `ppo_trainer.yaml` 并扩展 DAPO 特有参数
  - `main_dapo.py`：训练入口，负责配置加载、Ray 初始化和 Trainer 启动
  - `dapo_ray_trainer.py`：DAPO 训练器，继承 `RayPPOTrainer` 并实现动态采样逻辑
  - `run_*.sh`：训练脚本示例
  - `prepare_dapo_data.sh`：数据准备脚本

- **verl/**：复用的 verl 框架核心（DAPO 在此基础上增强）
  - `trainer/ppo/`：PPO 基础训练器、核心算法（`agg_loss`、`compute_policy_loss_vanilla`）
  - `workers/actor/`：Actor 前向推理与更新逻辑（DAPO 在此处应用解耦裁剪）
  - `workers/reward_manager/dapo.py`：**DAPO 新增的奖励管理器**，实现超长惩罚机制
  - `workers/rollout/`：vLLM 推理后端
  - `utils/`：日志跟踪、数值计算等工具

### 1.2 角色与数据流

#### 1.2.1 核心角色

- **Actor**：策略模型，使用 PPO 更新；支持 FSDP 或 Megatron 并行
- **RewardManager**：奖励计算器，DAPO 使用 `DAPORewardManager` 实现超长惩罚
- **vLLM Rollout**：高效推理引擎，用于生成响应

#### 1.2.2 数据流

DAPO 的训练数据流与标准 PPO 的主要区别在于第 2-4 步的动态采样与组过滤机制：

1. **数据加载**：从训练集加载 prompts
2. **生成响应（动态采样）**：对每个 prompt 生成 n 个响应；若启用组过滤，可能需要多次生成以收集足够的有效样本
3. **计算奖励**：使用 DAPORewardManager 计算奖励，包括基础任务奖励和超长惩罚
4. **组过滤**：按 prompt 分组，过滤掉指标方差为 0 的组（全对或全错），保留有对比信号的组
5. **计算 old_log_prob 和优势**：使用 Actor 和 Ref/Critic 计算优势函数
6. **PPO 更新**：应用解耦裁剪和灵活损失聚合进行策略梯度更新
7. **保存检查点与日志**：定期保存模型和记录训练指标

## 2. 训练管线与配置要点

### 2.1 训练入口与初始化

**入口**：`dapo/main_dapo.py`

训练初始化流程包含以下步骤：

1. **配置加载**：使用 Hydra 加载 `dapo/config/dapo_trainer.yaml`，该配置继承 verl 的 `ppo_trainer.yaml` 并扩展 DAPO 特有参数
2. **Ray 初始化**：连接或启动 Ray 分布式集群
3. **Worker 实例化**：根据配置实例化各类 Worker（ActorRolloutRef、Critic、RewardModel 等）
4. **RewardManager 加载**：加载 DAPORewardManager，并传入超长惩罚配置 `overlong_buffer_cfg`
5. **Trainer 实例化**：创建 `RayDAPOTrainer` 实例，注入所有依赖项（workers、reward_fn 等）

### 2.2 核心训练循环与动态采样

**训练循环入口**：`dapo_ray_trainer.py:fit` 方法

DAPO 的核心训练循环与标准 PPO 的主要区别在于**动态采样与组过滤**机制。训练流程如下：

#### 2.2.1 动态采样循环

对于每个训练批次，DAPO 不是一次性生成所有样本，而是采用循环采样策略：

1. **生成 n 个响应**：对当前批次的 prompts，使用 vLLM 生成 n 个候选响应
2. **计算奖励**：调用 DAPORewardManager 计算每个响应的奖励（包含超长惩罚）
3. **组过滤**（若启用）：
   - 按 prompt 分组，计算每组内指定指标（如 acc、seq_final_reward）的方差
   - 过滤掉方差为 0 的组（即所有响应完全相同的组，如全对或全错）
   - 保留有对比信号的组
4. **检查终止条件**：
   - 若收集到足够的有效 prompts（达到 `train_batch_size`），结束采样
   - 若采样批次数达到上限（`max_num_gen_batches`），结束采样
   - 否则继续循环，采样新批次

#### 2.2.2 PPO 更新流程

动态采样完成后，进入标准 PPO 更新流程：

1. **计算 old_log_prob**：使用当前 Actor 计算每个 token 的 log 概率
2. **计算 ref_log_prob**（若使用）：使用参考模型计算 log 概率，用于 KL 散度计算
3. **计算 values**（若使用 critic）：使用 Critic 估计价值函数
4. **计算优势**：根据配置的优势估计方法（GAE/GRPO/RLOO）计算优势函数
5. **更新 critic**（若使用）：使用 TD 误差更新 Critic
6. **更新 actor**：应用解耦裁剪和灵活损失聚合进行策略梯度更新
7. **验证与保存**：定期在验证集上评估，保存检查点

### 2.3 解耦裁剪与损失聚合机制

DAPO 在 Actor 更新阶段引入了两项关键改进：

#### 2.3.1 解耦裁剪（Decoupled Clipping）

传统 PPO 使用单一裁剪参数 ε，将重要性采样比限制在 [1-ε, 1+ε]。DAPO 引入 `clip_ratio_low` 和 `clip_ratio_high` 两个参数，允许对正负优势采用不同的裁剪策略。在计算策略梯度损失时，根据优势的符号选择不同的裁剪阈值。

#### 2.3.2 灵活损失聚合（Flexible Loss Aggregation）

DAPO 支持三种损失聚合模式，通过 `loss_agg_mode` 参数控制：
- `token-mean`：在所有 token 上取平均，每个 token 贡献相等
- `seq-mean-token-sum`：先在序列内对 token 求和，再在序列间取平均
- `seq-mean-token-mean`：先在序列内对 token 取平均，再在序列间取平均

这两项改进在 `verl/workers/actor/dp_actor.py` 的 `update_actor` 方法中实现，通过调用 `verl/trainer/ppo/core_algos.py` 中的 `compute_policy_loss_vanilla` 和 `agg_loss` 函数完成。

## 3. 关键模块说明

### 3.1 `dapo/main_dapo.py`

**功能**：DAPO 的训练入口，负责整个训练流程的启动与协调。

**主要职责**：
- 配置加载：使用 Hydra 加载训练配置
- Ray 初始化：连接或启动分布式集群
- Worker 实例化：根据配置创建各类 Worker
- RewardManager 加载：实例化 DAPORewardManager，传入超长惩罚配置
- Trainer 启动：创建并启动 `RayDAPOTrainer`

**关键配置项**：
- `reward_model.reward_manager = "dapo"`：指定使用 DAPORewardManager
- `reward_model.overlong_buffer`：超长惩罚相关配置
- `algorithm.filter_groups`：动态采样与组过滤配置

### 3.2 `dapo/dapo_ray_trainer.py:RayDAPOTrainer`

**功能**：DAPO 训练器，继承自 verl 的 `RayPPOTrainer` 基类。

**核心改动**：在 `fit` 方法中加入动态采样与组过滤逻辑。与标准 PPO 训练器的主要区别在于：
- **动态采样循环**：不是一次性生成所有样本，而是循环采样直到收集到足够的有效组
- **组过滤机制**：按 prompt 分组，计算每组内指定指标的方差，过滤掉方差为 0 的组（全对或全错）
- **批次拼接**：将多次采样的有效样本拼接成完整的训练批次

**关键方法**：
- `fit`：主训练循环，实现动态采样与组过滤
- `_validate`：验证集评估
- `_save_checkpoint`：保存检查点

### 3.3 `verl/workers/reward_manager/dapo.py:DAPORewardManager`

**功能**：DAPO 专用的奖励管理器，在基础任务奖励上叠加超长惩罚。

**核心逻辑**：
1. **解码响应**：将 token IDs 解码为文本
2. **计算基础奖励**：调用具体任务的评分函数（如数学题的正确性检验）
3. **超长惩罚**（若启用）：
   - 若响应长度超过预期长度（`expected_len = max_resp_len - overlong_buffer_len`）
   - 在 buffer 区间内线性增加惩罚：`penalty = -(exceed_len / buffer_len) * penalty_factor`
   - 惩罚上限为 `-penalty_factor`
4. **奖励赋值**：将最终奖励放在响应最后一个 token 的位置

**配置参数**：
- `overlong_buffer.enable`：是否启用超长惩罚
- `overlong_buffer.len`：buffer 长度
- `overlong_buffer.penalty_factor`：惩罚强度
- `overlong_buffer.log`：是否记录超长信息

### 3.4 `verl/trainer/ppo/core_algos.py`

**功能**：实现 PPO 的核心算法函数，DAPO 在此基础上扩展了解耦裁剪和灵活聚合。

#### 3.4.1 `agg_loss` 函数

**功能**：将损失矩阵聚合为标量损失。

**输入**：
- `loss_mat`：形状为 `(bs, response_length)` 的损失矩阵
- `loss_mask`：形状为 `(bs, response_length)` 的掩码
- `loss_agg_mode`：聚合模式（`token-mean`、`seq-mean-token-sum`、`seq-mean-token-mean`）

**输出**：标量损失

**实现逻辑**：根据不同的 `loss_agg_mode` 采用不同的聚合策略，控制 token 和序列维度的平均/求和方式。

#### 3.4.2 `compute_policy_loss_vanilla` 函数

**功能**：计算 PPO 策略梯度损失，支持解耦裁剪。

**输入**：
- `old_log_prob`、`log_prob`：旧策略和当前策略的 log 概率
- `advantages`：优势函数
- `response_mask`：响应掩码
- `loss_agg_mode`：损失聚合模式
- `config`：Actor 配置，包含 `clip_ratio_low` 和 `clip_ratio_high`

**输出**：
- `pg_loss`：策略梯度损失
- `pg_clipfrac`：裁剪比例（用于监控）
- `ppo_kl`：KL 散度估计
- `pg_clipfrac_lower`：下界裁剪比例

**核心逻辑**：
1. 计算重要性采样比 `ratio = exp(log_prob - old_log_prob)`
2. 使用解耦裁剪：根据优势符号选择不同的裁剪阈值（`clip_ratio_low` 或 `clip_ratio_high`）
3. 应用双裁剪机制处理极端负优势
4. 调用 `agg_loss` 聚合损失

