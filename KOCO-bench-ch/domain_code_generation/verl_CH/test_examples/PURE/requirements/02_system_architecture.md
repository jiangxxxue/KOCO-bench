# PURE 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（PURE/verl版本核心）
```
code/verl/
├── trainer/
│   ├── main_ppo.py                 # 训练入口与角色装配
│   ├── config/ppo_trainer.yaml     # 关键训练配置（FSDP、vLLM、PRM、RLOO等）
│   └── ppo/ray_trainer.py          # 训练循环、指标统计、验证与日志
├── workers/
│   ├── fsdp_workers.py             # FSDP Workers，含 ProcessRewardModelWorker（PRM推断与min-form加权）
│   ├── megatron_workers.py         # Megatron并行支持（可选）
│   ├── rollout/
│   │   ├── vllm_rollout/*.py       # vLLM 推理与采样（支持温度/TopK/TopP，n采样）
│   │   └── hf_rollout.py           # HF 推理后端（备用）
│   ├── actor/dp_actor.py           # Actor前向/反传（温度作用于logits）
│   └── reward_manager/
│       ├── prime.py                # 可验证奖励（VR）规则/解析器管理
│       ├── naive.py                # 简单奖励管理（baseline）
│       └── blank.py                # 空管理器（仅PRM，不使用VR）
├── utils/
│   ├── logger/wandb_ws.py          # 自定义wandb工作区，多面板指标
│   ├── torch_functional.py         # 数值函数实现（logprobs/entropy/clamp等）
│   └── seqlen_balancing.py         # 长度均衡/分片工具
└── ...
```

### 1.2 角色与数据流
- 角色
  - Actor：策略模型，使用 PPO 更新；支持 FSDP 并行与 vLLM 推理。
  - Ref：参考模型，用于 KL/对比与 log_prob 计算。
  - PRM：过程奖励模型（`type: prm`），在本仓库中仅推断模式；用于生成 token/step 级过程奖励。
  - RewardManager：可验证奖励（VR）解析器与融合器（`prime/blank/naive`）。
- 关键数据
  - prompts/responses：输入与生成序列。
  - verifiable_rewards：按样本的结果级奖励（0/1 等）。
  - rm_scores：PRM 产生的 token 级过程奖励（经最小式加权后聚合）。

## 2. 训练管线与配置要点

### 2.1 训练入口与模式切换
- 入口：`python -m verl.trainer.main_ppo`
- 奖励模式开关（`verl/trainer/config/ppo_trainer.yaml`）
  - PURE-VR：`reward_model.enable=False` 且 `reward_model.reward_manager=prime`
  - PURE-PRM：`reward_model.enable=True` 且 `reward_model.reward_manager=blank`
  - PURE-PRM+VR：`reward_model.enable=True` 且 `reward_model.reward_manager=prime`


### 2.2 信用分配（Min-form）与 PRM 推断
- 配置项：`reward_model.credit_assignment`
  - 'gamma-decay'：兼容旧式 γ 衰减；
  - 'strict min-form'：严格最小式（无需温度）；
  - 浮点温度（>0）：启用近似最小式权重 `softmax(-rm_score/T)`（推荐）。
- 实现位置：`ProcessRewardModelWorker._forward_micro_batch`（`fsdp_workers.py`）
  - 对 PRM logits 取 softmax 后计算二分类差分 `(p_positive - p_negative)` 得到 token 级 `rm_score`；
  - 使用 `reward_mask` 选取响应 token；
  - 若开启近似最小式，则对 `rm_score` 施加 `softmax(-rm_score/T)` 权重并逐 token 重加权；
  - 聚合后返回 `rm_scores` 给上层优势估计。

### 2.3 优势估计与损失
- `algorithm.adv_estimator = rloo`：基于组内 leave-one-out 的优势估计，配合 `adv_norm=True` 做白化；
- PPO 相关：`clip_ratio`、`entropy_coeff`、`ppo_epochs` 等在 `actor_rollout_ref.actor` 下配置；
- KL 相关：可选 GRPO 风格 `use_kl_loss=True` 或固定系数 `algorithm.kl_ctrl`；



