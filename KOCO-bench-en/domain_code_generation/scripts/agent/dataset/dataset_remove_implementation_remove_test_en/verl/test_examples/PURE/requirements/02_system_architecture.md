# PURE System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (PURE/verl version core)
```
code/verl/
├── trainer/
│   ├── main_ppo.py                 # Training entry point and role assembly
│   ├── config/ppo_trainer.yaml     # Key training configuration (FSDP, vLLM, PRM, RLOO, etc.)
│   └── ppo/ray_trainer.py          # Training loop, metric statistics, validation and logging
├── workers/
│   ├── fsdp_workers.py             # FSDP Workers, including ProcessRewardModelWorker (PRM inference and min-form weighting)
│   ├── megatron_workers.py         # Megatron parallelism support (optional)
│   ├── rollout/
│   │   ├── vllm_rollout/*.py       # vLLM inference and sampling (supports temperature/TopK/TopP, n sampling)
│   │   └── hf_rollout.py           # HF inference backend (fallback)
│   ├── actor/dp_actor.py           # Actor forward/backward (temperature applied to logits)
│   └── reward_manager/
│       ├── prime.py                # Verifiable reward (VR) rule/parser manager
│       ├── naive.py                # Simple reward manager (baseline)
│       └── blank.py                # Empty manager (PRM only, no VR)
├── utils/
│   ├── logger/wandb_ws.py          # Custom wandb workspace, multi-panel metrics
│   ├── torch_functional.py         # Numerical function implementations (logprobs/entropy/clamp, etc.)
│   └── seqlen_balancing.py         # Length balancing/sharding tools
└── ...
```

### 1.2 Roles and Data Flow
- Roles
  - Actor: Policy model, updated using PPO; supports FSDP parallelism and vLLM inference.
  - Ref: Reference model, used for KL/contrast and log_prob computation.
  - PRM: Process reward model (`type: prm`), inference-only mode in this repository; used to generate token/step-level process rewards.
  - RewardManager: Verifiable reward (VR) parser and fusion module (`prime/blank/naive`).
- Key Data
  - prompts/responses: Input and generated sequences.
  - verifiable_rewards: Outcome-level rewards per sample (0/1, etc.).
  - rm_scores: Token-level process rewards produced by PRM (aggregated after min-form weighting).

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Point and Mode Switching
- Entry point: `python -m verl.trainer.main_ppo`
- Reward mode switches (`verl/trainer/config/ppo_trainer.yaml`)
  - PURE-VR: `reward_model.enable=False` and `reward_model.reward_manager=prime`
  - PURE-PRM: `reward_model.enable=True` and `reward_model.reward_manager=blank`
  - PURE-PRM+VR: `reward_model.enable=True` and `reward_model.reward_manager=prime`


### 2.2 Credit Assignment (Min-form) and PRM Inference
- Configuration option: `reward_model.credit_assignment`
  - 'gamma-decay': Compatible with legacy γ decay;
  - 'strict min-form': Strict minimum form (no temperature required);
  - Float temperature (>0): Enable approximate min-form weights `softmax(-rm_score/T)` (recommended).
- Implementation location: `ProcessRewardModelWorker._forward_micro_batch` (`fsdp_workers.py`)
  - Apply softmax to PRM logits, then compute binary classification difference `(p_positive - p_negative)` to obtain token-level `rm_score`;
  - Use `reward_mask` to select response tokens;
  - If approximate min-form is enabled, apply `softmax(-rm_score/T)` weights to `rm_score` and reweight token-wise;
  - Aggregate and return `rm_scores` to upper-level advantage estimation.

### 2.3 Advantage Estimation and Loss
- `algorithm.adv_estimator = rloo`: Group-based leave-one-out advantage estimation, combined with `adv_norm=True` for whitening;
- PPO-related: `clip_ratio`, `entropy_coeff`, `ppo_epochs`, etc. configured under `actor_rollout_ref.actor`;
- KL-related: Optional GRPO-style `use_kl_loss=True` or fixed coefficient `algorithm.kl_ctrl`;




