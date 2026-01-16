# PACS System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (PACS Core)
```
code/src/
├── pacs/
│   ├── pacs_main.py                # Training entry point and role assembly
│   ├── pacs_trainer.py             # Training loop, metric statistics, validation and logging
│   ├── pacs_workers.py             # FSDP Workers (Actor/Rollout/Ref)
│   ├── pacs_actor.py               # Actor forward/backward and PACS loss computation
│   ├── pacs_core_algos.py          # Core algorithms: reward computation, advantage estimation, weight computation, PACS loss
│   ├── pacs_compute_metrics.py     # Metric computation and statistics
│   └── config/ppo_trainer.yaml     # Training configuration (FSDP, vLLM, PACS parameters, etc.)
├── verl/                           # verl framework (version 432f9e)
│   ├── trainer/                    # Base trainer
│   ├── workers/                    # Base workers
│   ├── utils/                      # Utility functions
│   └── ...
├── reward_function.py              # Verifiable reward function (math problem verification)
├── inference.py                    # Inference script
├── model_merger.py                 # Model weight conversion (FSDP → HF)
└── preprocess_dataset.py           # Data preprocessing
```

### 1.2 Roles and Data Flow
- Roles
  - Actor: Policy model, updated using PACS loss; supports FSDP parallelism and vLLM inference.
  - Ref: Reference model, used to compute log probability baseline (old_log_prob).
  - RewardManager: Verifiable reward (VR) parser, used for math problem verification, etc.
  - Critic: PACS doesn't use Critic; this role has empty implementation in PACS.
- Key Data
  - prompts/responses: Input and generated sequences.
  - token_level_scores: Verifiable rewards (0 or 1).
  - old_log_probs: Reference policy log probabilities.
  - log_probs: Current policy log probabilities.
  - advantages: Advantage function (used as logit input to BCE loss).

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Point and Mode Switching
- Entry point: `python -m pacs.pacs_main`
- Key configuration (`configs/ppo_trainer.yaml` and command-line arguments)
  - PACS parameters:
    - `actor_rollout_ref.pacs.beta`: Controls aggressiveness of policy updates (default 1.0)
    - `actor_rollout_ref.pacs.reward_method`: Reward computation method ('1' or '2')
    - `actor_rollout_ref.pacs.adv_estimator`: Advantage estimator ('rloo'/'grpo'/'naive')
    - `actor_rollout_ref.pacs.use_weight`: Whether to use sample weights
    - `actor_rollout_ref.pacs.weight_mode`: Weight mode ('question'/'only_positive'/'only_negative')
  - Rollout parameters:
    - `actor_rollout_ref.rollout.n`: Number of responses generated per prompt (default 8)
    - `actor_rollout_ref.rollout.name`: Inference backend ('vllm' or 'hf')
  - Model parameters:
    - `actor_rollout_ref.model.path`: Pretrained model path
    - `actor_rollout_ref.model.use_remove_padding`: Whether to use depadding optimization

### 2.2 PACS Loss Computation
- Configuration: `actor_rollout_ref.pacs`
  - `beta`: Reward scaling factor, controls policy update magnitude;
  - `reward_method`:
    - '1': `reward = beta × sum(log_prob - old_log_prob)`
    - '2': `reward = beta × sum(log_prob) / response_length`
  - `adv_estimator`:
    - 'rloo': Group-based leave-one-out advantage estimation;
    - 'grpo': Group-based advantage estimation using mean and standard deviation;
    - 'naive': Directly use reward as advantage.
- Implementation location: `pacs_core_algos.py:compute_pacs_loss`
  - Compute reward (policy improvement measure);
  - Compute advantage function based on `adv_estimator`;
  - Sum verifiable rewards (token_level_scores) as labels;
  - Optionally compute sample weights (handle class imbalance);
  - Compute loss using `BCEWithLogitsLoss(advantage, score, weight)`.

### 2.3 Sample Weighting Mechanism
- Configuration: `actor_rollout_ref.pacs.use_weight` and `weight_mode`
  - `use_weight=True`: Enable sample weighting;
  - `weight_mode`:
    - 'question': Balance positive and negative sample weights at question level (using sklearn's `compute_class_weight`);
    - 'only_positive': Only assign weight 1 to positive samples, negative samples get weight 0;
    - 'only_negative': Only assign weight 1 to negative samples, positive samples get weight 0.
- Implementation location: `pacs_core_algos.py:compute_weight`
  - For each group of n responses, compute balanced weights based on label distribution;
  - Return weight tensor with shape `[batch_size, 1]`.

### 2.4 Training Loop and Metrics
- Training loop: `pacs_trainer.py:PACSTrainer.fit`
  - Generate responses → Compute verifiable rewards → Compute log probabilities → Compute advantages → Update Actor;
  - Supports validation, checkpoint saving, metric logging, etc.
- Key metrics:
  - `actor/pacs_loss`: PACS loss value;
  - `actor/grad_norm`: Gradient norm;
  - `critic/score/mean`: Verifiable reward mean;
  - `critic/rewards/mean`: Policy improvement reward mean;
  - `response_length/mean`: Response length statistics.

