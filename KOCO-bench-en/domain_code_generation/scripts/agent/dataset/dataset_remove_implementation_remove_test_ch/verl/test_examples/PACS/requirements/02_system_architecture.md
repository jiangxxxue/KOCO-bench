# PACS 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（PACS 核心）
```
code/src/
├── pacs/
│   ├── pacs_main.py                # 训练入口与角色装配
│   ├── pacs_trainer.py             # 训练循环、指标统计、验证与日志
│   ├── pacs_workers.py             # FSDP Workers（Actor/Rollout/Ref）
│   ├── pacs_actor.py               # Actor 前向/反传与 PACS 损失计算
│   ├── pacs_core_algos.py          # 核心算法：奖励计算、优势估计、权重计算、PACS 损失
│   ├── pacs_compute_metrics.py     # 指标计算与统计
│   └── config/ppo_trainer.yaml     # 训练配置（FSDP、vLLM、PACS 参数等）
├── verl/                           # verl 框架（版本 432f9e）
│   ├── trainer/                    # 基础训练器
│   ├── workers/                    # 基础 workers
│   ├── utils/                      # 工具函数
│   └── ...
├── reward_function.py              # 可验证奖励函数（数学问题验证）
├── inference.py                    # 推理脚本
├── model_merger.py                 # 模型权重转换（FSDP → HF）
└── preprocess_dataset.py           # 数据预处理
```

### 1.2 角色与数据流
- 角色
  - Actor：策略模型，使用 PACS 损失更新；支持 FSDP 并行与 vLLM 推理。
  - Ref：参考模型，用于计算对数概率基线（old_log_prob）。
  - RewardManager：可验证奖励（VR）解析器，用于数学问题验证等。
  - Critic：PACS 不使用 Critic，该角色在 PACS 中为空实现。
- 关键数据
  - prompts/responses：输入与生成序列。
  - token_level_scores：可验证奖励（0 或 1）。
  - old_log_probs：参考策略的对数概率。
  - log_probs：当前策略的对数概率。
  - advantages：优势函数（作为 logit 输入到 BCE 损失）。

## 2. 训练管线与配置要点

### 2.1 训练入口与模式切换
- 入口：`python -m pacs.pacs_main`
- 关键配置（`configs/ppo_trainer.yaml` 与命令行参数）
  - PACS 参数：
    - `actor_rollout_ref.pacs.beta`：控制策略更新激进程度（默认 1.0）
    - `actor_rollout_ref.pacs.reward_method`：奖励计算方法（'1' 或 '2'）
    - `actor_rollout_ref.pacs.adv_estimator`：优势估计器（'rloo'/'grpo'/'naive'）
    - `actor_rollout_ref.pacs.use_weight`：是否使用样本权重
    - `actor_rollout_ref.pacs.weight_mode`：权重模式（'question'/'only_positive'/'only_negative'）
  - Rollout 参数：
    - `actor_rollout_ref.rollout.n`：每个 prompt 生成的响应数量（默认 8）
    - `actor_rollout_ref.rollout.name`：推理后端（'vllm' 或 'hf'）
  - 模型参数：
    - `actor_rollout_ref.model.path`：预训练模型路径
    - `actor_rollout_ref.model.use_remove_padding`：是否使用去填充优化

### 2.2 PACS 损失计算
- 配置项：`actor_rollout_ref.pacs`
  - `beta`：奖励缩放因子，控制策略更新幅度；
  - `reward_method`：
    - '1'：`reward = beta × sum(log_prob - old_log_prob)`
    - '2'：`reward = beta × sum(log_prob) / response_length`
  - `adv_estimator`：
    - 'rloo'：基于组内 leave-one-out 的优势估计；
    - 'grpo'：基于组内均值和标准差的优势估计；
    - 'naive'：直接使用奖励作为优势。
- 实现位置：`pacs_core_algos.py:compute_pacs_loss`
  - 计算奖励（策略改进度量）；
  - 根据 `adv_estimator` 计算优势函数；
  - 将可验证奖励（token_level_scores）求和作为标签；
  - 可选地计算样本权重（处理类别不平衡）；
  - 使用 `BCEWithLogitsLoss(advantage, score, weight)` 计算损失。

### 2.3 样本权重机制
- 配置项：`actor_rollout_ref.pacs.use_weight` 与 `weight_mode`
  - `use_weight=True`：启用样本权重；
  - `weight_mode`：
    - 'question'：按问题级别平衡正负样本权重（使用 sklearn 的 `compute_class_weight`）；
    - 'only_positive'：仅对正样本赋权重 1，负样本权重 0；
    - 'only_negative'：仅对负样本赋权重 1，正样本权重 0。
- 实现位置：`pacs_core_algos.py:compute_weight`
  - 对每组 n 个响应，根据标签分布计算平衡权重；
  - 返回形状为 `[batch_size, 1]` 的权重张量。

### 2.4 训练循环与指标
- 训练循环：`pacs_trainer.py:PACSTrainer.fit`
  - 生成响应 → 计算可验证奖励 → 计算对数概率 → 计算优势 → 更新 Actor；
  - 支持验证、检查点保存、指标记录等。
- 关键指标：
  - `actor/pacs_loss`：PACS 损失值；
  - `actor/grad_norm`：梯度范数；
  - `critic/score/mean`：可验证奖励均值；
  - `critic/rewards/mean`：策略改进奖励均值；
  - `response_length/mean`：响应长度统计。
