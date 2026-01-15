# ARES 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（ARES 核心）
```
code/
├── verl/
│   ├── trainer/
│   │   ├── main.py                      # 训练入口
│   │   ├── ray_trainer.py               # ARES 训练器实现，包含难度分桶、熵感知 rollout
│   │   ├── core_algos.py                # 核心算法函数（PPO、KL、优势估计）
│   │   ├── config.py                    # 配置管理
│   │   ├── data_loader.py               # 数据加载器（支持动态采样过滤）
│   │   └── metrics.py                   # 指标计算与统计
│   ├── utils/
│   │   └── seqlen_balancing.py          # 序列长度均衡
│   └── models/
│       └── transformers/
│           └── qwen2_vl.py              # Qwen2-VL 模型适配（支持多模态）
├── examples/
│   ├── reward_function/
│   │   └── aepo.py                      # AEPO 奖励函数实现
│   └── format_prompt/
│       └── math.jinja                   # 提示词格式模板
└── experiments/
    └── AEPO/
        ├── config.yaml                  # AEPO 完整训练配置
        └── train.sh                     # 训练启动脚本
```

### 1.2 角色与数据流
- **角色**
  - **Actor**：策略模型，使用 PPO 更新；支持 FSDP 并行与动态 KL 损失。
  - **Ref**：参考模型，用于 KL 散度计算与对比学习。
  - **Rollout**：vLLM 推理引擎，生成多个候选响应，计算 token 级熵与 HWE token。
  - **RewardManager**：奖励函数管理器，计算准确率奖励与熵塑形奖励。

- **关键数据**
  - `prompts/responses`：输入提示与生成序列。
  - `entropies`：token 级熵序列，用于识别高不确定性区域。
  - `high_entropy_token_num`：每个响应中 HWE token 的数量，用于熵奖励计算。
  - `difficulty`：样本难度标签（easy/medium/hard），动态分桶得到。
  - `alpha_entropy`：难度相关的熵奖励系数，通过自适应机制动态调整。
  - `target_high_entropy_token_num`：目标 HWE token 数，作为推理资源预算。

## 2. 训练管线与配置要点

### 2.1 训练入口与模式切换
- **入口**：`python -m verl.trainer.main`
- **核心配置**（`experiments/AEPO/config.yaml`）
  - **Bootstrap 模式**：`data.bootstrap=True` 启用难度预估阶段
  - **在线过滤**：`algorithm.online_filtering=True` 启用动态样本过滤
  - **Rollout 配置**：`worker.rollout.n=8` 设置每个 prompt 的采样数
  - **熵窗口**：`worker.rollout.window_size=4` 设置滑动窗口大小
  - **KL 控制**：`algorithm.kl_coef` 设置基础 KL 系数，`worker.actor.dynamic_kl_loss=True` 启用动态 KL

### 2.2 难度分桶与自适应机制
- **Bootstrap 阶段**（训练前）
  - 对整个训练集进行一轮 rollout，计算每个 prompt 的平均准确率。
  - 根据准确率阈值分桶：`acc ≥ 2/3` → easy，`1/3 ≤ acc < 2/3` → medium，`acc < 1/3` → hard。
  - 初始化每个难度桶的目标 HWE token 数和熵系数 α。

- **在线更新**（训练中）
  - 每个训练 batch 后，根据当前 rollout 更新 `difficulty_dict`。
  - 计算每个难度桶的平均 HWE token 数偏差：`Δ = mean(N_HWE) - target`。
  - 使用梯度上升调整熵系数：`α_d ← α_d + lr · Δ`，使生成的 HWE token 数逐渐接近目标。
  - 支持对偶上升（dual ascent）机制动态调整 KL 预算：`λ_d ← max(0, λ_d + η · (D_KL^(d) - ε_d))`。

### 2.3 HWE（高窗口熵）机制
- **窗口熵计算**
  - 对每个 token，计算其后续 `window_size` 个 token 的平均熵。
  - 窗口平滑可以过滤短期波动，识别持续的高不确定性区域。

- **HWE token 识别**
  - 使用全局阈值 τ（通过 percentile 动态估计）区分高熵与低熵 token。
  - HWE token 标记模型在推理过程中遇到的困难决策点。
  - 在这些位置，模型可以触发更多探索（通过降低 KL 约束）。

- **动态 KL 控制**
  - 在 HWE token 位置，降低 token 级 KL 系数（例如 `β_low = 0.5 * β_normal`）。
  - 允许模型在高不确定性区域偏离参考策略，进行更自由的探索。
  - 在非 HWE 区域，保持正常 KL 约束，防止模型过度发散。

### 2.4 熵奖励塑形策略
- **Easy 任务**
  - **目标**：减少冗余推理，提高效率。
  - **策略**：若答案正确且 HWE token 数超过目标，施加 Huber 惩罚；若答案错误但探索较多，给予小幅奖励。
  - **公式**：对于正确答案，`R_entropy = -min(cap, huber_penalty(over) / normalization * cap)`。

- **Medium 任务**
  - **目标**：平衡探索与效率。
  - **策略**：若答案正确，惩罚偏离目标 HWE token 数（过多或过少）；若答案错误，鼓励适度探索。
  - **公式**：对于正确答案，`R_entropy = -min(cap, huber_penalty(abs(delta)) / normalization * cap)`。

- **Hard 任务**
  - **目标**：鼓励深入探索，找到正确解。
  - **策略**：若答案正确，奖励充分探索（HWE token 数接近或超过目标）；若答案错误，也奖励探索尝试。
  - **公式**：对于正确答案，`R_entropy = sigmoid_saturate(delta + margin) * cap`。

- **自适应函数**
  - **Huber 惩罚**：在接近目标时二次惩罚，远离目标时线性惩罚，避免梯度爆炸。
  - **Sigmoid 饱和**：平滑的 [0,1] 增长函数，用于奖励探索，避免奖励无限增长。
  - **Margin 机制**：根据难度设置容忍带（easy: 15%, medium: 25%, hard: 35%），在 margin 内不施加惩罚。

### 2.5 优势估计与损失
- **优势估计器**：`algorithm.adv_estimator = grpo` 或 `rloo`，使用组内对比降低方差。
- **PPO 损失**：支持标准 PPO clip 和双 clip 变体（dual-clip），配置 `clip_ratio_low/high/dual`。
- **KL 损失**：若 `use_kl_loss=True`，使用 KL 散度作为独立损失项；若 `False`，则将 KL 直接加到奖励中。
- **动态 KL 损失**：`worker.actor.dynamic_kl_loss=True` 时，根据 HWE token 位置动态调整 token 级 KL 权重。

## 3. ARES 数据协议与模块交互

### 3.1 ARES 数据协议（DataProto）

数据协议结构：ARES 在标准 PPO 数据协议基础上扩展了熵相关字段。

DataProto.batch 字段规范：
```python
data.batch = {
    "prompts": torch.Tensor,                     # 输入提示序列 [batch_size, prompt_len]
    "responses": torch.Tensor,                   # 响应序列 [batch_size, response_len]
    "attention_mask": torch.Tensor,              # 注意力掩码 [batch_size, total_len]
    "response_mask": torch.Tensor,               # 响应掩码 [batch_size, response_len]
    "token_level_scores": torch.Tensor,          # token 级奖励分数 [batch_size, response_len]
    "old_log_probs": torch.Tensor,               # Actor 旧策略对数概率 [batch_size, response_len]
    "ref_log_probs": torch.Tensor,               # Ref 策略对数概率 [batch_size, response_len]
    "advantages": torch.Tensor,                  # 优势函数 [batch_size, response_len]
    "returns": torch.Tensor,                     # 回报值 [batch_size, response_len]
}

data.non_tensor_batch = {
    "uid": np.ndarray,                           # 样本唯一标识符（用于分组）
    "global_id": np.ndarray,                     # 全局样本 ID（用于难度跟踪）
    "ground_truth": np.ndarray,                  # 标准答案
    "entropies": np.ndarray,                     # token 级熵序列（每个样本的平均熵）
    "high_entropy_token_num": np.ndarray,        # 每个样本的 HWE token 数量
    "entropy_percentile_thresholds": np.ndarray, # 熵阈值（用于 HWE 判定）
    "difficulty": List[str],                     # 样本难度标签 ["easy", "medium", "hard"]
    "target_high_entropy_token_num": List[int],  # 目标 HWE token 数（难度相关）
    "alpha_entropy": List[float],                # 熵奖励系数（难度相关）
    "accuracy": List[float],                     # 准确率标签 [0.0, 1.0]
}

data.meta_info = {
    "min_pixels": int,                           # 多模态图像最小像素
    "max_pixels": int,                           # 多模态图像最大像素
    "video_fps": int,                            # 视频帧率
    "high_entropy_threshold": float,             # 全局 HWE 阈值
    "global_token_num": List[int],               # 每个样本的总 token 数
}
```

批次数据组织：
- **分组结构**：数据按 `n_samples` 进行分组，每组包含同一 prompt 的多个响应。
- **索引规律**：样本 `[i*n:(i+1)*n]` 属于第 i 组，便于 GRPO/RLOO 优势计算。
- **难度跟踪**：通过 `global_id` 跨 batch 跟踪样本难度演化。

### 3.2 ARES 训练流程
1. **Bootstrap**：对整个数据集进行一轮 rollout，初始化 `difficulty_dict` 和 `target_entropy_dict`。
2. **生成 rollout**：使用 vLLM 引擎为每个 prompt 生成 n 个响应，同时计算 token 级熵与 HWE token。
3. **在线过滤**：根据 `filter_key` 和 `filter_low/high` 阈值过滤极端样本（全部正确或全部错误的组）。
4. **难度更新**：根据当前 batch 的 rollout 准确率更新 `difficulty_dict` 和 `target_entropy_dict`。
5. **熵奖励**：根据难度、准确率、HWE token 数计算分层熵奖励。
6. **动态 KL**：在 HWE token 位置降低 KL 权重，允许更多探索。
7. **优势计算**：使用 GRPO/RLOO 计算组内对比优势函数。
8. **PPO 更新**：使用 clipped surrogate objective 更新 Actor 参数。

### 3.3 ARES 数据流
- **输入**：原始数据集（prompt + ground_truth + 可选的多模态数据）
- **Rollout**：生成 n 个响应 × rollout_batch_size 个 prompt，计算熵统计
- **过滤**：保留准确率在 [filter_low, filter_high] 区间的样本组
- **奖励**：计算准确率奖励 + 熵塑形奖励，减去 KL 惩罚
- **优势**：利用分组结构计算 GRPO/RLOO 优势函数
- **更新**：PPO 更新 Actor，同时自适应调整熵系数和目标 token 数

### 3.4 ARES 自适应更新策略
- **熵系数调整**：通过梯度上升机制，根据当前 HWE token 数与目标的偏差调整 α_entropy。
- **目标更新**：定期根据当前难度桶的统计数据更新 target_high_entropy_token_num。
- **阈值估计**：使用 EMA（指数移动平均）平滑 HWE 阈值，避免短期波动。
- **动态采样**：可选地从训练集中移除"过于简单"的样本（全部正确且熵极低），聚焦于有挑战性的数据。

