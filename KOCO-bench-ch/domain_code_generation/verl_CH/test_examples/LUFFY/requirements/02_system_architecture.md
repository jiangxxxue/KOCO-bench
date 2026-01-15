# LUFFY 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织
```
code/luffy/
├── verl/verl/mix_src/              # LUFFY 核心实现（基于 verl 的扩展）
│   ├── main_mix_ppo.py             # 训练入口，配置角色与奖励管理
│   ├── mix_trainer.py              # 混合训练器，实现 on/off-policy 数据流
│   ├── mix_trainer_acc_rebatch.py  # 支持累积梯度的训练器变体
│   ├── mix_core_alg.py             # 核心算法：GRPO split、混合策略损失
│   ├── mix_actor.py                # Actor 模块，实现混合策略更新
│   ├── mix_fsdp_worker.py          # FSDP Worker，处理模型并行
│   ├── mix_vllm_rollout.py         # vLLM Rollout，支持前缀采样
│   ├── rl_dataset_with_target.py   # 数据集加载器，支持 off-policy target
│   ├── math_verify_reward.py       # 数学验证奖励函数
│   ├── reward_with_format.py       # 带格式检查的奖励函数
│   └── config/                     # 训练配置文件
│       └── mix_ppo_trainer.yaml    # 主配置文件
├── deepscaler/                     # 奖励计算与系统提示
│   ├── rewards/                    # 奖励函数实现
│   ├── system_prompts.py           # 不同模型的系统提示
│   └── utils.py                    # 工具函数
├── scripts/                        # 训练与评估脚本
│   ├── train/                      # 训练脚本
│   ├── eval/                       # 评估脚本
│   └── data/                       # 数据处理脚本
└── data/                           # 训练数据
    ├── prepare_train.py            # 准备训练数据
    └── prepare_sft.py              # 准备 SFT 数据
```

### 1.2 核心模块说明

#### 1.2.1 MIXRayPPOTrainer（mix_trainer.py）
混合训练器，继承自 verl 的 RayPPOTrainer，扩展以支持 on/off-policy 混合训练：
- **数据加载**：使用 `RLHFDatasetWithTarget` 加载包含 off-policy target 的数据
- **混合采样**：协调 on-policy 和 off-policy 样本的生成
- **拒绝采样**：基于奖励分布过滤极端样本
- **优势计算**：调用 `compute_grpo_split_advantage` 进行分组优势估计
- **前缀权重**：应用 α/β 权重调整 on/off-policy 样本的优势

#### 1.2.2 MIXvLLMRollout（mix_vllm_rollout.py）
扩展的 vLLM Rollout 模块，支持前缀采样：
- **前缀策略**：
  - `random`：随机采样前缀比例
  - `linear`：线性递减前缀比例（课程学习）
  - `linear_max`：分段线性递减
  - `reverse_linear`：反向线性增长
  - `fixed`：固定前缀比例集合
- **前缀生成**：根据 target 和 prefix_ratio 截取前缀
- **前缀掩码**：生成 `prefix_mask` 标记哪些 token 来自 off-policy
- **灵活配置**：支持 `n_prefix`（off-policy 样本数）和 `n`（总样本数）的独立控制

#### 1.2.3 MIXDataParallelPPOActor（mix_actor.py）
扩展的 Actor 模块，实现混合策略更新：
- **混合损失计算**：调用 `compute_token_on_off_policy_loss` 计算 on/off-policy 损失
- **策略塑形**：支持多种 ratio 变换方法（`off_policy_reshape`, `on_policy_reshape`）
- **自适应温度**：可选的自适应熵系数调整
- **多任务学习**：可选的 SFT 多任务损失（`use_sft_multitask_loss`）

#### 1.2.4 核心算法函数（mix_core_alg.py）

**compute_grpo_outcome_advantage_split**：
- 分组 GRPO 优势估计，仅使用 on-policy 样本计算基线
- 输入：token 级奖励、EOS 掩码、样本索引、on-policy 掩码
- 输出：优势和回报（形状 `[batch_size, response_length]`）

**compute_token_on_off_policy_loss**：
- 计算混合策略损失（on-policy PPO + off-policy 重要性采样）
- 支持多种策略塑形方法
- 支持 ratio clipping（`off_max_clip`, `off_min_clip`）
- 返回详细的损失分解和统计信息

**compute_sft_pure_loss**：
- 计算纯 SFT 损失（用于多任务学习）
- 输入：log 概率和 EOS 掩码
- 输出：平均负 log 似然损失

### 1.3 数据流与角色交互

```
训练循环数据流：
1. DataLoader → Batch (prompts + optional targets)
2. MIXvLLMRollout → 生成响应（混合 on/off-policy）
   - 对每个 prompt 生成 n 个响应
   - 其中 n_prefix 个使用 off-policy 前缀
   - 生成 prefix_mask 标记 off-policy token
3. RewardManager → 计算可验证奖励
   - 使用 math_verify 或其他奖励函数
   - 返回 token 级奖励（最后一个有效 token）
4. 拒绝采样 → 过滤极端样本
   - 移除全部成功或全部失败的 prompt 组
5. Actor → 计算 old_log_prob
6. RefPolicy → 计算 ref_log_prob（可选）
7. 优势计算 → compute_grpo_split_advantage
   - 按 prompt 分组
   - 仅用 on-policy 样本计算基线
   - 应用前缀权重
8. Actor 更新 → 混合策略损失
   - On-policy: PPO clip loss
   - Off-policy: 重要性采样 + policy shaping
   - 熵正则化
9. 指标记录 → WandB/TensorBoard
```

## 2. 配置系统

### 2.1 关键配置项（mix_ppo_trainer.yaml）

#### 2.1.1 Rollout 配置
```yaml
actor_rollout_ref:
  rollout:
    n: 8                          # 每个 prompt 的总样本数
    n_prefix: 4                   # 使用 off-policy 前缀的样本数（-1 表示全部）
    prefix_strategy: 'random'     # 前缀策略：random/linear/fixed/reverse_linear
    min_prefix_ratio: 0.0         # 最小前缀比例
    max_prefix_ratio: 0.8         # 最大前缀比例
    prefix_steps: 300             # 线性策略的步数
    prefix_share_across_samples: false  # 是否在同一 prompt 的样本间共享前缀比例
    max_prefix_len: 8192          # 最大前缀长度
    prefix_reward_weight_alpha: 1.0     # off-policy 优势权重
    prefix_reward_weight_beta: 1.0      # on-policy 优势权重
```

#### 2.1.2 Actor 配置
```yaml
actor_rollout_ref:
  actor:
    use_off_policy_loss: true           # 启用混合损失
    off_policy_reshape: 'p_div_p_0.5'   # off-policy 策略塑形方法
    off_policy_max_clip: 10.0           # off-policy ratio 上限
    off_policy_min_clip: -1             # off-policy ratio 下限（-1 表示不限制）
    on_policy_reshape: 'no_reshape'     # on-policy 策略塑形方法
    clip_ratio: 0.2                     # PPO clip 范围
    clip_upper_bound: 100.0             # PPO clip 上限
    entropy_coeff: 0.0                  # 熵系数
    use_adaptive_temperature: false     # 是否使用自适应熵系数
    use_kl_loss: false                  # 是否使用 KL 损失
    use_sft_multitask_loss: false       # 是否使用 SFT 多任务损失
```

#### 2.1.3 算法配置
```yaml
algorithm:
  adv_estimator: 'grpo_split'     # 优势估计器：grpo_split
  grpo_use_std: true              # GRPO 是否使用标准差归一化
  kl_ctrl:
    type: 'fixed'                 # KL 控制类型：fixed/adaptive
    kl_coef: 0.001                # KL 系数
```

#### 2.1.4 数据配置
```yaml
data:
  train_files: ['path/to/train.parquet']
  val_files: ['path/to/val.parquet']
  train_batch_size: 256           # 训练批大小
  prompt_key: 'prompt'            # prompt 字段名
  max_prompt_length: 2048         # 最大 prompt 长度
  max_response_length: 8192       # 最大响应长度
  max_target_length: 8192         # 最大 target 长度
  sample_target_ratio: 1.0        # target 采样比例
  filter_targets: true            # 是否过滤过长的 target
  reward_impl_version: 3          # 奖励实现版本
  add_tgt_with_acc: false         # 是否添加带准确率的 target
```

### 2.2 训练模式切换

#### 2.2.1 LUFFY-Zero（纯 RL，使用 off-policy 引导）
```yaml
reward_model:
  enable: false                   # 不使用 PRM
actor_rollout_ref:
  rollout:
    n_prefix: 4                   # 使用部分 off-policy 样本
  actor:
    use_off_policy_loss: true     # 启用混合损失
```

#### 2.2.2 SFT+RL（先 SFT 后 RL）
```yaml
# 第一阶段：SFT（使用 OpenRLHF）
# 第二阶段：RL（使用 heldout 数据）
data:
  train_files: ['path/to/heldout.parquet']
```

#### 2.2.3 RL w/ SFT Loss（多任务学习）
```yaml
actor_rollout_ref:
  actor:
    use_sft_multitask_loss: true  # 启用 SFT 多任务损失
    sft_loss_coef: 0.1            # SFT 损失系数
```
