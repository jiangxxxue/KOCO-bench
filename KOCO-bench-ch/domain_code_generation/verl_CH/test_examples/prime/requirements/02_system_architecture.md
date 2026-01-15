# PRIME算法系统架构与模块设计

## 1. PRIME模块文件结构

### 1.1 文件组织
```
recipe/prime/
├── main_prime.py              # PRIME算法启动入口
├── prime_ray_trainer.py       # PRIME训练器实现
├── prime_core_algos.py        # PRIME核心算法函数
├── prime_dp_rm.py             # PRIME奖励模型实现
├── prime_fsdp_workers.py      # PRIME工作器实现
└── config/
    └── prime_trainer.yaml     # PRIME配置文件
```

### 1.2 模块职责划分
- main_prime.py: PRIME算法的启动逻辑和环境配置
- prime_ray_trainer.py: 继承PPO训练器并实现PRIME功能
- prime_core_algos.py: PRIME算法的纯函数实现集合
- prime_dp_rm.py: PRIME隐式奖励模型的核心实现
- prime_fsdp_workers.py: PRIME奖励模型的分布式工作器封装

## 2. PRIME模块设计

### 2.1 PRIME训练器 (prime_ray_trainer.py)

核心类: `RayPRIMETrainer`

主要功能:
- 禁用Critic: `self.use_critic = False` - 移除价值函数网络
- 过采样数据加载: 数据批次大小乘以过采样因子进行加载
- 智能过滤策略: `filter_and_downsample()` - 基于准确率和长度的双重过滤
- 奖励模型联合更新: 在优势计算前同步更新奖励模型参数

PRIME函数:
- `filter_and_downsample()`: 实现过采样后的智能筛选和优先级排序
- `_create_dataloader()`: 支持过采样因子的数据加载器定制

训练循环实现:
- 数据验证后进行过滤下采样
- 奖励模型更新与分数计算的灵活调度
- 多源奖励的优势计算集成

### 2.2 PRIME奖励模型 (prime_dp_rm.py)

核心类: `DataParallelPRIMERewardModel`

主要方法:
- `_forward_micro_batch()`: 隐式奖励计算的核心实现
- `prime_norm()`: PRIME的反向累积和归一化
- `compute_rm_score()`: 包含PRIME归一化的奖励分数计算
- `update_rm()`: 支持多种PRIME损失函数的模型更新

隐式奖励计算实现:
- 同时计算奖励模型和参考模型的对数概率
- 通过对数概率差异生成隐式奖励信号
- 支持GAE时序建模和直接奖励两种模式
- 灵活的奖励粒度分配（token级或整体级）

### 2.3 PRIME算法函数库 (prime_core_algos.py)

算法函数:
- `compute_rloo_advantage_return()`: 多源奖励的RLOO优势计算
- `compute_ce_dpo_loss_rm()`: CE风格的隐式奖励损失
- `compute_detach_dpo_loss_rm()`: 支持BoN模式的分离式DPO损失
- `compute_dpo_accuracy()`: 成对比较的奖励模型准确率评估
- `compute_dpo_abs_accuracy()`: 绝对准确率的简化评估

算法实现:
- 多源奖励融合: 同时处理rm_scores和accuracy两种奖励源
- RLOO白化: 优势函数标准化为均值≈0、标准差≈1的分布
- BoN模式支持: 根据样本排名分配不同训练权重
- 成对比较评估: 通过样本对比较评估奖励模型性能
- GAE时序建模: 支持GAE模式的反向累积奖励计算

### 2.4 PRIME工作器 (prime_fsdp_workers.py)

核心类: `PRIMERewardModelWorker`

主要接口:
- `compute_rm_score()`: 分布式环境下的PRIME奖励分数计算
- `update_rm()`: 分布式环境下的PRIME奖励模型更新
- `init_model()`: PRIME奖励模型和参考模型的初始化

集成实现:
- 封装`DataParallelPRIMERewardModel`为分布式工作器
- 集成PRIME的DPO准确率评估指标
- 支持PRIME配置的ulysses序列并行和动态批次处理

### 2.5 PRIME配置系统 (config/prime_trainer.yaml)

配置结构说明：PRIME算法采用分层配置结构，算法相关参数位于 `algorithm` 子节点下。

PRIME核心配置:
- `algorithm.reward_dpo_coef`: DPO奖励权重系数，默认值5.0，用于多源奖励融合
- `algorithm.reward_gt_coef`: 准确率奖励权重系数，默认值5.0，用于多源奖励融合
- `prime_granularity`: 奖励粒度控制（token/whole）
- `prime_norm`: PRIME归一化模式（batch_norm/none）
- `reward_manager`: 指定使用prime奖励管理器
- `beta_train`: PRIME温度参数，默认值0.05
- `loss_type`: PRIME损失类型（ce/dpo/bon_acc/bon_rm）

数据处理配置:
- `oversample_factor`: 过采样倍数配置，默认值4.0
- `filter_accuracy`: 准确率过滤开关，默认值True
- `accuracy_lower_bound`: 准确率过滤下界，默认值0.2
- `accuracy_upper_bound`: 准确率过滤上界，默认值0.8
- `filter_truncate`: 长度过滤开关，默认值True
- `max_response_length`: 最大响应长度阈值

算法配置:
- `adv_estimator`: 固定使用"rloo"优势估计算法
- `n_samples`: 每组采样数量，默认值4

## 3. PRIME数据协议与模块交互

### 3.1 PRIME数据协议 (DataProto)

数据协议结构: PRIME算法使用标准化的数据协议，确保各模块间的数据传递一致性。

DataProto.batch字段规范:
```python
data.batch = {
    "rm_scores": torch.Tensor,           # 隐式奖励模型分数 [batch_size, seq_len]
    "acc": torch.Tensor,                 # 二元准确率标签 [batch_size]  
    "prompts": torch.Tensor,             # 输入提示序列 [batch_size, prompt_len]
    "attention_mask": torch.Tensor,      # 注意力掩码 [batch_size, total_len]
    "responses": torch.Tensor,           # 响应序列 [batch_size, response_len]
    "input_ids": torch.Tensor           # 完整输入序列 [batch_size, total_len]
}
```

批次数据组织:
- 分组结构: 数据按n_samples进行分组，每组包含同一prompt的多个响应
- 索引规律: 样本[i*n_samples:(i+1)*n_samples]属于第i组
- RLOO计算: 利用分组结构计算leave-one-out基线

### 3.2 PRIME训练流程
1. 过采样加载: 按过采样因子加载更多数据，维持分组结构
2. PRIME验证: 使用PrimeRewardManager验证响应准确性
3. 智能过滤: 基于准确率和长度过滤，保持组内完整性
4. 隐式奖励: 通过双模型对数概率差异计算奖励
5. PRIME归一化: 应用反向累积和归一化策略
6. 多源优势: 融合rm_scores和accuracy计算RLOO优势函数

### 3.3 PRIME数据流
- 输入: 分组原始数据 × oversample_factor
- 过滤: 基于准确率和长度的双重筛选，保持分组结构
- 下采样: 保留1/oversample_factor的高质量样本组
- 奖励: 生成隐式过程奖励和二元准确率标签
- 优势: 利用分组结构计算多源融合的RLOO优势函数

### 3.4 PRIME更新策略
- 奖励模型先更新: update="before"模式下先更新RM再计算奖励
- 反向更新模式: update="reverse"模式下先计算奖励统计信息，然后更新奖励模型，支持批次内样本对比
- 联合训练: 每个训练步同时更新Actor和奖励模型
- 无Critic依赖: 完全移除价值函数网络的依赖
