# ARES System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (ARES Core)
```
code/
├── verl/
│   ├── trainer/
│   │   ├── main.py                      # Training entry point
│   │   ├── ray_trainer.py               # ARES trainer implementation, including difficulty bucketing, entropy-aware rollout
│   │   ├── core_algos.py                # Core algorithm functions (PPO, KL, advantage estimation)
│   │   ├── config.py                    # Configuration management
│   │   ├── data_loader.py               # Data loader (supports dynamic sampling filtering)
│   │   └── metrics.py                   # Metric computation and statistics
│   ├── utils/
│   │   └── seqlen_balancing.py          # Sequence length balancing
│   └── models/
│       └── transformers/
│           └── qwen2_vl.py              # Qwen2-VL model adaptation (supports multimodal)
├── examples/
│   ├── reward_function/
│   │   └── aepo.py                      # AEPO reward function implementation
│   └── format_prompt/
│       └── math.jinja                   # Prompt format template
└── experiments/
    └── AEPO/
        ├── config.yaml                  # AEPO complete training configuration
        └── train.sh                     # Training launch script
```

### 1.2 Roles and Data Flow
- **Roles**
  - **Actor**: Policy model, updated using PPO; supports FSDP parallelism and dynamic KL loss.
  - **Ref**: Reference model, used for KL divergence computation and contrastive learning.
  - **Rollout**: vLLM inference engine, generates multiple candidate responses, computes token-level entropy and HWE tokens.
  - **RewardManager**: Reward function manager, computes accuracy rewards and entropy shaping rewards.

- **Key Data**
  - `prompts/responses`: Input prompts and generated sequences.
  - `entropies`: Token-level entropy sequences, used to identify high-uncertainty regions.
  - `high_entropy_token_num`: Number of HWE tokens in each response, used for entropy reward computation.
  - `difficulty`: Sample difficulty labels (easy/medium/hard), obtained through dynamic bucketing.
  - `alpha_entropy`: Difficulty-related entropy reward coefficient, dynamically adjusted through adaptive mechanism.
  - `target_high_entropy_token_num`: Target HWE token count, serving as reasoning resource budget.

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Point and Mode Switching
- **Entry point**: `python -m verl.trainer.main`
- **Core Configuration** (`experiments/AEPO/config.yaml`)
  - **Bootstrap mode**: `data.bootstrap=True` enables difficulty estimation phase
  - **Online filtering**: `algorithm.online_filtering=True` enables dynamic sample filtering
  - **Rollout configuration**: `worker.rollout.n=8` sets sampling count per prompt
  - **Entropy window**: `worker.rollout.window_size=4` sets sliding window size
  - **KL control**: `algorithm.kl_coef` sets base KL coefficient, `worker.actor.dynamic_kl_loss=True` enables dynamic KL

### 2.2 Difficulty Bucketing and Adaptive Mechanism
- **Bootstrap Phase** (before training)
  - Perform one rollout pass over the entire training set, computing average accuracy for each prompt.
  - Bucket based on accuracy thresholds: `acc ≥ 2/3` → easy, `1/3 ≤ acc < 2/3` → medium, `acc < 1/3` → hard.
  - Initialize target HWE token count and entropy coefficient α for each difficulty bucket.

- **Online Update** (during training)
  - After each training batch, update `difficulty_dict` based on current rollout.
  - Compute average HWE token count deviation for each difficulty bucket: `Δ = mean(N_HWE) - target`.
  - Use gradient ascent to adjust entropy coefficient: `α_d ← α_d + lr · Δ`, gradually bringing generated HWE token count closer to target.
  - Support dual ascent mechanism to dynamically adjust KL budget: `λ_d ← max(0, λ_d + η · (D_KL^(d) - ε_d))`.

### 2.3 HWE (High Window Entropy) Mechanism
- **Window Entropy Computation**
  - For each token, compute the average entropy of the subsequent `window_size` tokens.
  - Window smoothing filters short-term fluctuations, identifying sustained high-uncertainty regions.

- **HWE Token Identification**
  - Use global threshold τ (dynamically estimated via percentile) to distinguish high-entropy from low-entropy tokens.
  - HWE tokens mark difficult decision points encountered by the model during reasoning.
  - At these positions, the model can trigger more exploration (by reducing KL constraints).

- **Dynamic KL Control**
  - At HWE token positions, reduce token-level KL coefficient (e.g., `β_low = 0.5 * β_normal`).
  - Allow the model to deviate from reference policy in high-uncertainty regions for freer exploration.
  - In non-HWE regions, maintain normal KL constraints to prevent excessive model divergence.

### 2.4 Entropy Reward Shaping Strategy
- **Easy Tasks**
  - **Goal**: Reduce redundant reasoning, improve efficiency.
  - **Strategy**: If answer is correct and HWE token count exceeds target, apply Huber penalty; if answer is incorrect but exploration is high, give small reward.
  - **Formula**: For correct answers, `R_entropy = -min(cap, huber_penalty(over) / normalization * cap)`.

- **Medium Tasks**
  - **Goal**: Balance exploration and efficiency.
  - **Strategy**: If answer is correct, penalize deviation from target HWE token count (too many or too few); if answer is incorrect, encourage moderate exploration.
  - **Formula**: For correct answers, `R_entropy = -min(cap, huber_penalty(abs(delta)) / normalization * cap)`.

- **Hard Tasks**
  - **Goal**: Encourage deep exploration to find correct solution.
  - **Strategy**: If answer is correct, reward sufficient exploration (HWE token count close to or exceeding target); if answer is incorrect, also reward exploration attempts.
  - **Formula**: For correct answers, `R_entropy = sigmoid_saturate(delta + margin) * cap`.

- **Adaptive Functions**
  - **Huber penalty**: Quadratic penalty near target, linear penalty far from target, avoiding gradient explosion.
  - **Sigmoid saturation**: Smooth [0,1] growth function, used to reward exploration, avoiding unbounded reward growth.
  - **Margin mechanism**: Set tolerance bands based on difficulty (easy: 15%, medium: 25%, hard: 35%), no penalty within margin.

### 2.5 Advantage Estimation and Loss
- **Advantage estimator**: `algorithm.adv_estimator = grpo` or `rloo`, using within-group contrast to reduce variance.
- **PPO loss**: Supports standard PPO clip and dual-clip variants, configured via `clip_ratio_low/high/dual`.
- **KL loss**: If `use_kl_loss=True`, use KL divergence as independent loss term; if `False`, add KL directly to reward.
- **Dynamic KL loss**: When `worker.actor.dynamic_kl_loss=True`, dynamically adjust token-level KL weights based on HWE token positions.

## 3. ARES Data Protocol and Module Interaction

### 3.1 ARES Data Protocol (DataProto)

Data protocol structure: ARES extends the standard PPO data protocol with entropy-related fields.

DataProto.batch field specification:
```python
data.batch = {
    "prompts": torch.Tensor,                     # Input prompt sequences [batch_size, prompt_len]
    "responses": torch.Tensor,                   # Response sequences [batch_size, response_len]
    "attention_mask": torch.Tensor,              # Attention mask [batch_size, total_len]
    "response_mask": torch.Tensor,               # Response mask [batch_size, response_len]
    "token_level_scores": torch.Tensor,          # Token-level reward scores [batch_size, response_len]
    "old_log_probs": torch.Tensor,               # Actor old policy log probabilities [batch_size, response_len]
    "ref_log_probs": torch.Tensor,               # Ref policy log probabilities [batch_size, response_len]
    "advantages": torch.Tensor,                  # Advantage function [batch_size, response_len]
    "returns": torch.Tensor,                     # Return values [batch_size, response_len]
}

data.non_tensor_batch = {
    "uid": np.ndarray,                           # Sample unique identifier (for grouping)
    "global_id": np.ndarray,                     # Global sample ID (for difficulty tracking)
    "ground_truth": np.ndarray,                  # Ground truth answer
    "entropies": np.ndarray,                     # Token-level entropy sequences (average entropy per sample)
    "high_entropy_token_num": np.ndarray,        # HWE token count per sample
    "entropy_percentile_thresholds": np.ndarray, # Entropy thresholds (for HWE determination)
    "difficulty": List[str],                     # Sample difficulty labels ["easy", "medium", "hard"]
    "target_high_entropy_token_num": List[int],  # Target HWE token count (difficulty-related)
    "alpha_entropy": List[float],                # Entropy reward coefficient (difficulty-related)
    "accuracy": List[float],                     # Accuracy labels [0.0, 1.0]
}

data.meta_info = {
    "min_pixels": int,                           # Multimodal image minimum pixels
    "max_pixels": int,                           # Multimodal image maximum pixels
    "video_fps": int,                            # Video frame rate
    "high_entropy_threshold": float,             # Global HWE threshold
    "global_token_num": List[int],               # Total token count per sample
}
```

Batch data organization:
- **Grouping structure**: Data is grouped by `n_samples`, each group containing multiple responses from the same prompt.
- **Indexing pattern**: Samples `[i*n:(i+1)*n]` belong to group i, facilitating GRPO/RLOO advantage computation.
- **Difficulty tracking**: Track sample difficulty evolution across batches via `global_id`.

### 3.2 ARES Training Flow
1. **Bootstrap**: Perform one rollout pass over entire dataset, initialize `difficulty_dict` and `target_entropy_dict`.
2. **Generate rollout**: Use vLLM engine to generate n responses per prompt, computing token-level entropy and HWE tokens.
3. **Online filtering**: Filter extreme samples (all correct or all incorrect groups) based on `filter_key` and `filter_low/high` thresholds.
4. **Difficulty update**: Update `difficulty_dict` and `target_entropy_dict` based on current batch rollout accuracy.
5. **Entropy reward**: Compute hierarchical entropy rewards based on difficulty, accuracy, and HWE token count.
6. **Dynamic KL**: Reduce KL weight at HWE token positions, allowing more exploration.
7. **Advantage computation**: Use GRPO/RLOO to compute within-group contrastive advantage function.
8. **PPO update**: Update Actor parameters using clipped surrogate objective.

### 3.3 ARES Data Flow
- **Input**: Raw dataset (prompt + ground_truth + optional multimodal data)
- **Rollout**: Generate n responses × rollout_batch_size prompts, compute entropy statistics
- **Filtering**: Retain sample groups with accuracy in [filter_low, filter_high] range
- **Reward**: Compute accuracy reward + entropy shaping reward, subtract KL penalty
- **Advantage**: Use grouping structure to compute GRPO/RLOO advantage function
- **Update**: PPO update Actor, while adaptively adjusting entropy coefficients and target token counts

### 3.4 ARES Adaptive Update Strategy
- **Entropy coefficient adjustment**: Through gradient ascent mechanism, adjust α_entropy based on deviation between current HWE token count and target.
- **Target update**: Periodically update target_high_entropy_token_num based on current difficulty bucket statistics.
- **Threshold estimation**: Use EMA (exponential moving average) to smooth HWE threshold, avoiding short-term fluctuations.
- **Dynamic sampling**: Optionally remove "too easy" samples (all correct with extremely low entropy) from training set, focusing on challenging data.


