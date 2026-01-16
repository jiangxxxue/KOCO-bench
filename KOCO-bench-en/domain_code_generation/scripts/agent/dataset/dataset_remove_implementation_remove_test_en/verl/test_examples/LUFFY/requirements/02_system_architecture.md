# LUFFY System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization
```
code/luffy/
├── verl/verl/mix_src/              # LUFFY core implementation (extension based on verl)
│   ├── main_mix_ppo.py             # Training entry point, configures roles and reward management
│   ├── mix_trainer.py              # Mixed trainer, implements on/off-policy data flow
│   ├── mix_trainer_acc_rebatch.py  # Trainer variant supporting gradient accumulation
│   ├── mix_core_alg.py             # Core algorithms: GRPO split, mixed policy loss
│   ├── mix_actor.py                # Actor module, implements mixed policy update
│   ├── mix_fsdp_worker.py          # FSDP Worker, handles model parallelism
│   ├── mix_vllm_rollout.py         # vLLM Rollout, supports prefix sampling
│   ├── rl_dataset_with_target.py   # Dataset loader, supports off-policy targets
│   ├── math_verify_reward.py       # Math verification reward function
│   ├── reward_with_format.py       # Reward function with format checking
│   └── config/                     # Training configuration files
│       └── mix_ppo_trainer.yaml    # Main configuration file
├── deepscaler/                     # Reward computation and system prompts
│   ├── rewards/                    # Reward function implementations
│   ├── system_prompts.py           # System prompts for different models
│   └── utils.py                    # Utility functions
├── scripts/                        # Training and evaluation scripts
│   ├── train/                      # Training scripts
│   ├── eval/                       # Evaluation scripts
│   └── data/                       # Data processing scripts
└── data/                           # Training data
    ├── prepare_train.py            # Prepare training data
    └── prepare_sft.py              # Prepare SFT data
```

### 1.2 Core Module Descriptions

#### 1.2.1 MIXRayPPOTrainer (mix_trainer.py)
Mixed trainer, inheriting from verl's RayPPOTrainer, extended to support on/off-policy mixed training:
- **Data loading**: Uses `RLHFDatasetWithTarget` to load data containing off-policy targets
- **Mixed sampling**: Coordinates generation of on-policy and off-policy samples
- **Rejection sampling**: Filters extreme samples based on reward distribution
- **Advantage computation**: Calls `compute_grpo_split_advantage` for group advantage estimation
- **Prefix weighting**: Applies α/β weights to adjust advantages for on/off-policy samples

#### 1.2.2 MIXvLLMRollout (mix_vllm_rollout.py)
Extended vLLM Rollout module supporting prefix sampling:
- **Prefix strategies**:
  - `random`: Randomly sample prefix ratio
  - `linear`: Linearly decrease prefix ratio (curriculum learning)
  - `linear_max`: Piecewise linear decrease
  - `reverse_linear`: Reverse linear increase
  - `fixed`: Fixed prefix ratio set
- **Prefix generation**: Truncate prefix based on target and prefix_ratio
- **Prefix mask**: Generate `prefix_mask` marking which tokens come from off-policy
- **Flexible configuration**: Support independent control of `n_prefix` (off-policy sample count) and `n` (total sample count)

#### 1.2.3 MIXDataParallelPPOActor (mix_actor.py)
Extended Actor module implementing mixed policy update:
- **Mixed loss computation**: Calls `compute_token_on_off_policy_loss` to compute on/off-policy losses
- **Policy shaping**: Supports multiple ratio transformation methods (`off_policy_reshape`, `on_policy_reshape`)
- **Adaptive temperature**: Optional adaptive entropy coefficient adjustment
- **Multi-task learning**: Optional SFT multi-task loss (`use_sft_multitask_loss`)

#### 1.2.4 Core Algorithm Functions (mix_core_alg.py)

**compute_grpo_outcome_advantage_split**:
- Group GRPO advantage estimation, using only on-policy samples to compute baseline
- Input: Token-level rewards, EOS mask, sample indices, on-policy mask
- Output: Advantages and returns (shape `[batch_size, response_length]`)

**compute_token_on_off_policy_loss**:
- Compute mixed policy loss (on-policy PPO + off-policy importance sampling)
- Support multiple policy shaping methods
- Support ratio clipping (`off_max_clip`, `off_min_clip`)
- Return detailed loss decomposition and statistics

**compute_sft_pure_loss**:
- Compute pure SFT loss (for multi-task learning)
- Input: Log probabilities and EOS mask
- Output: Average negative log likelihood loss

### 1.3 Data Flow and Role Interaction

```
Training Loop Data Flow:
1. DataLoader → Batch (prompts + optional targets)
2. MIXvLLMRollout → Generate responses (mixed on/off-policy)
   - Generate n responses per prompt
   - n_prefix of them use off-policy prefix
   - Generate prefix_mask marking off-policy tokens
3. RewardManager → Compute verifiable rewards
   - Use math_verify or other reward functions
   - Return token-level rewards (last valid token)
4. Rejection sampling → Filter extreme samples
   - Remove all-success or all-failure prompt groups
5. Actor → Compute old_log_prob
6. RefPolicy → Compute ref_log_prob (optional)
7. Advantage computation → compute_grpo_split_advantage
   - Group by prompt
   - Use only on-policy samples to compute baseline
   - Apply prefix weights
8. Actor update → Mixed policy loss
   - On-policy: PPO clip loss
   - Off-policy: Importance sampling + policy shaping
   - Entropy regularization
9. Metric logging → WandB/TensorBoard
```

## 2. Configuration System

### 2.1 Key Configuration Items (mix_ppo_trainer.yaml)

#### 2.1.1 Rollout Configuration
```yaml
actor_rollout_ref:
  rollout:
    n: 8                          # Total samples per prompt
    n_prefix: 4                   # Samples using off-policy prefix (-1 means all)
    prefix_strategy: 'random'     # Prefix strategy: random/linear/fixed/reverse_linear
    min_prefix_ratio: 0.0         # Minimum prefix ratio
    max_prefix_ratio: 0.8         # Maximum prefix ratio
    prefix_steps: 300             # Steps for linear strategy
    prefix_share_across_samples: false  # Whether to share prefix ratio across samples from same prompt
    max_prefix_len: 8192          # Maximum prefix length
    prefix_reward_weight_alpha: 1.0     # off-policy advantage weight
    prefix_reward_weight_beta: 1.0      # on-policy advantage weight
```

#### 2.1.2 Actor Configuration
```yaml
actor_rollout_ref:
  actor:
    use_off_policy_loss: true           # Enable mixed loss
    off_policy_reshape: 'p_div_p_0.5'   # off-policy policy shaping method
    off_policy_max_clip: 10.0           # off-policy ratio upper limit
    off_policy_min_clip: -1             # off-policy ratio lower limit (-1 means no limit)
    on_policy_reshape: 'no_reshape'     # on-policy policy shaping method
    clip_ratio: 0.2                     # PPO clip range
    clip_upper_bound: 100.0             # PPO clip upper bound
    entropy_coeff: 0.0                  # Entropy coefficient
    use_adaptive_temperature: false     # Whether to use adaptive entropy coefficient
    use_kl_loss: false                  # Whether to use KL loss
    use_sft_multitask_loss: false       # Whether to use SFT multi-task loss
```

#### 2.1.3 Algorithm Configuration
```yaml
algorithm:
  adv_estimator: 'grpo_split'     # Advantage estimator: grpo_split
  grpo_use_std: true              # Whether GRPO uses standard deviation normalization
  kl_ctrl:
    type: 'fixed'                 # KL control type: fixed/adaptive
    kl_coef: 0.001                # KL coefficient
```

#### 2.1.4 Data Configuration
```yaml
data:
  train_files: ['path/to/train.parquet']
  val_files: ['path/to/val.parquet']
  train_batch_size: 256           # Training batch size
  prompt_key: 'prompt'            # Prompt field name
  max_prompt_length: 2048         # Maximum prompt length
  max_response_length: 8192       # Maximum response length
  max_target_length: 8192         # Maximum target length
  sample_target_ratio: 1.0        # Target sampling ratio
  filter_targets: true            # Whether to filter overly long targets
  reward_impl_version: 3          # Reward implementation version
  add_tgt_with_acc: false         # Whether to add targets with accuracy
```

### 2.2 Training Mode Switching

#### 2.2.1 LUFFY-Zero (Pure RL with off-policy guidance)
```yaml
reward_model:
  enable: false                   # Don't use PRM
actor_rollout_ref:
  rollout:
    n_prefix: 4                   # Use some off-policy samples
  actor:
    use_off_policy_loss: true     # Enable mixed loss
```

#### 2.2.2 SFT+RL (SFT first, then RL)
```yaml
# Stage 1: SFT (using OpenRLHF)
# Stage 2: RL (using heldout data)
data:
  train_files: ['path/to/heldout.parquet']
```

#### 2.2.3 RL w/ SFT Loss (Multi-task learning)
```yaml
actor_rollout_ref:
  actor:
    use_sft_multitask_loss: true  # Enable SFT multi-task loss
    sft_loss_coef: 0.1            # SFT loss coefficient
```

