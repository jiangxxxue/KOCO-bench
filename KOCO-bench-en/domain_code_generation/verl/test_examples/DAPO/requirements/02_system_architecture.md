# DAPO System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (DAPO Core)

DAPO makes streamlined extensions to the verl framework, with new code concentrated in two areas:
- `dapo/` directory: Contains DAPO-specific training entry point, configuration files, and trainer implementation
- `verl/workers/reward_manager/dapo.py`: DAPO-specific reward manager

Specific directory structure:

- **dapo/**: DAPO-specific modules
  - `config/`: Training configuration files, inheriting verl's `ppo_trainer.yaml` and extending DAPO-specific parameters
  - `main_dapo.py`: Training entry point, responsible for configuration loading, Ray initialization, and Trainer startup
  - `dapo_ray_trainer.py`: DAPO trainer, inheriting `RayPPOTrainer` and implementing dynamic sampling logic
  - `run_*.sh`: Training script examples
  - `prepare_dapo_data.sh`: Data preparation script

- **verl/**: Reused verl framework core (DAPO enhances on this foundation)
  - `trainer/ppo/`: PPO base trainer, core algorithms (`agg_loss`, `compute_policy_loss_vanilla`)
  - `workers/actor/`: Actor forward inference and update logic (DAPO applies decoupled clipping here)
  - `workers/reward_manager/dapo.py`: **DAPO's new reward manager**, implementing overlong penalty mechanism
  - `workers/rollout/`: vLLM inference backend
  - `utils/`: Log tracking, numerical computation, and other utilities

### 1.2 Roles and Data Flow

#### 1.2.1 Core Roles

- **Actor**: Policy model, updated using PPO; supports FSDP or Megatron parallelism
- **RewardManager**: Reward calculator, DAPO uses `DAPORewardManager` to implement overlong penalty
- **vLLM Rollout**: Efficient inference engine for generating responses

#### 1.2.2 Data Flow

The main difference between DAPO's training data flow and standard PPO lies in the dynamic sampling and group filtering mechanism in steps 2-4:

1. **Data Loading**: Load prompts from training set
2. **Generate Responses (Dynamic Sampling)**: Generate n responses for each prompt; if group filtering is enabled, may need multiple generations to collect sufficient valid samples
3. **Compute Rewards**: Use DAPORewardManager to compute rewards, including base task rewards and overlong penalty
4. **Group Filtering**: Group by prompt, filter out groups with zero metric variance (all correct or all incorrect), retaining groups with contrastive signals
5. **Compute old_log_prob and Advantages**: Use Actor and Ref/Critic to compute advantage function
6. **PPO Update**: Apply decoupled clipping and flexible loss aggregation for policy gradient update
7. **Save Checkpoints and Logs**: Periodically save model and record training metrics

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Point and Initialization

**Entry point**: `dapo/main_dapo.py`

Training initialization process includes the following steps:

1. **Configuration Loading**: Use Hydra to load `dapo/config/dapo_trainer.yaml`, which inherits verl's `ppo_trainer.yaml` and extends DAPO-specific parameters
2. **Ray Initialization**: Connect to or start Ray distributed cluster
3. **Worker Instantiation**: Instantiate various Workers (ActorRolloutRef, Critic, RewardModel, etc.) based on configuration
4. **RewardManager Loading**: Load DAPORewardManager and pass in overlong penalty configuration `overlong_buffer_cfg`
5. **Trainer Instantiation**: Create `RayDAPOTrainer` instance, injecting all dependencies (workers, reward_fn, etc.)

### 2.2 Core Training Loop and Dynamic Sampling

**Training loop entry point**: `dapo_ray_trainer.py:fit` method

The main difference between DAPO's core training loop and standard PPO lies in the **dynamic sampling and group filtering** mechanism. Training flow:

#### 2.2.1 Dynamic Sampling Loop

For each training batch, DAPO doesn't generate all samples at once but adopts a cyclic sampling strategy:

1. **Generate n responses**: For current batch prompts, use vLLM to generate n candidate responses
2. **Compute rewards**: Call DAPORewardManager to compute reward for each response (including overlong penalty)
3. **Group filtering** (if enabled):
   - Group by prompt, compute variance of specified metric (e.g., acc, seq_final_reward) within each group
   - Filter out groups with zero variance (i.e., groups where all responses are identical, such as all correct or all incorrect)
   - Retain groups with contrastive signals
4. **Check termination condition**:
   - If sufficient valid prompts collected (reaching `train_batch_size`), end sampling
   - If sampling batch count reaches limit (`max_num_gen_batches`), end sampling
   - Otherwise continue loop, sampling new batches

#### 2.2.2 PPO Update Flow

After dynamic sampling completes, enter standard PPO update flow:

1. **Compute old_log_prob**: Use current Actor to compute log probability for each token
2. **Compute ref_log_prob** (if used): Use reference model to compute log probability for KL divergence calculation
3. **Compute values** (if using critic): Use Critic to estimate value function
4. **Compute advantages**: Compute advantage function based on configured advantage estimation method (GAE/GRPO/RLOO)
5. **Update critic** (if used): Update Critic using TD error
6. **Update actor**: Apply decoupled clipping and flexible loss aggregation for policy gradient update
7. **Validation and saving**: Periodically evaluate on validation set, save checkpoints

### 2.3 Decoupled Clipping and Loss Aggregation Mechanism

DAPO introduces two key improvements in the Actor update phase:

#### 2.3.1 Decoupled Clipping

Traditional PPO uses a single clipping parameter ε, limiting importance sampling ratio to [1-ε, 1+ε]. DAPO introduces `clip_ratio_low` and `clip_ratio_high` parameters, allowing different clipping strategies for positive and negative advantages. When computing policy gradient loss, different clipping thresholds are selected based on the sign of advantage.

#### 2.3.2 Flexible Loss Aggregation

DAPO supports three loss aggregation modes, controlled by `loss_agg_mode` parameter:
- `token-mean`: Average over all tokens, each token contributes equally
- `seq-mean-token-sum`: Sum tokens within sequences first, then average across sequences
- `seq-mean-token-mean`: Average tokens within sequences first, then average across sequences

These two improvements are implemented in the `update_actor` method of `verl/workers/actor/dp_actor.py`, completed by calling `compute_policy_loss_vanilla` and `agg_loss` functions in `verl/trainer/ppo/core_algos.py`.

## 3. Key Module Descriptions

### 3.1 `dapo/main_dapo.py`

**Function**: DAPO's training entry point, responsible for starting and coordinating the entire training process.

**Main Responsibilities**:
- Configuration loading: Use Hydra to load training configuration
- Ray initialization: Connect to or start distributed cluster
- Worker instantiation: Create various Workers based on configuration
- RewardManager loading: Instantiate DAPORewardManager, pass in overlong penalty configuration
- Trainer startup: Create and start `RayDAPOTrainer`

**Key Configuration Items**:
- `reward_model.reward_manager = "dapo"`: Specify using DAPORewardManager
- `reward_model.overlong_buffer`: Overlong penalty related configuration
- `algorithm.filter_groups`: Dynamic sampling and group filtering configuration

### 3.2 `dapo/dapo_ray_trainer.py:RayDAPOTrainer`

**Function**: DAPO trainer, inheriting from verl's `RayPPOTrainer` base class.

**Core Changes**: Add dynamic sampling and group filtering logic in `fit` method. Main differences from standard PPO trainer:
- **Dynamic sampling loop**: Instead of generating all samples at once, cyclically sample until collecting sufficient valid groups
- **Group filtering mechanism**: Group by prompt, compute metric variance within each group, filter out groups with zero variance (all correct or all incorrect)
- **Batch concatenation**: Concatenate valid samples from multiple samplings into complete training batch

**Key Methods**:
- `fit`: Main training loop, implementing dynamic sampling and group filtering
- `_validate`: Validation set evaluation
- `_save_checkpoint`: Save checkpoints

### 3.3 `verl/workers/reward_manager/dapo.py:DAPORewardManager`

**Function**: DAPO-specific reward manager, adding overlong penalty on top of base task rewards.

**Core Logic**:
1. **Decode responses**: Decode token IDs to text
2. **Compute base rewards**: Call task-specific scoring function (e.g., correctness verification for math problems)
3. **Overlong penalty** (if enabled):
   - If response length exceeds expected length (`expected_len = max_resp_len - overlong_buffer_len`)
   - Linearly increase penalty within buffer interval: `penalty = -(exceed_len / buffer_len) * penalty_factor`
   - Penalty cap is `-penalty_factor`
4. **Reward assignment**: Place final reward at the position of the last token in response

**Configuration Parameters**:
- `overlong_buffer.enable`: Whether to enable overlong penalty
- `overlong_buffer.len`: Buffer length
- `overlong_buffer.penalty_factor`: Penalty strength
- `overlong_buffer.log`: Whether to log overlong information

### 3.4 `verl/trainer/ppo/core_algos.py`

**Function**: Implements PPO's core algorithm functions, DAPO extends with decoupled clipping and flexible aggregation.

#### 3.4.1 `agg_loss` Function

**Function**: Aggregate loss matrix to scalar loss.

**Input**:
- `loss_mat`: Loss matrix with shape `(bs, response_length)`
- `loss_mask`: Mask with shape `(bs, response_length)`
- `loss_agg_mode`: Aggregation mode (`token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`)

**Output**: Scalar loss

**Implementation Logic**: Adopt different aggregation strategies based on different `loss_agg_mode`, controlling averaging/summing methods for token and sequence dimensions.

#### 3.4.2 `compute_policy_loss_vanilla` Function

**Function**: Compute PPO policy gradient loss, supporting decoupled clipping.

**Input**:
- `old_log_prob`, `log_prob`: Old policy and current policy log probabilities
- `advantages`: Advantage function
- `response_mask`: Response mask
- `loss_agg_mode`: Loss aggregation mode
- `config`: Actor configuration, including `clip_ratio_low` and `clip_ratio_high`

**Output**:
- `pg_loss`: Policy gradient loss
- `pg_clipfrac`: Clipping fraction (for monitoring)
- `ppo_kl`: KL divergence estimate
- `pg_clipfrac_lower`: Lower bound clipping fraction

**Core Logic**:
1. Compute importance sampling ratio `ratio = exp(log_prob - old_log_prob)`
2. Use decoupled clipping: Select different clipping thresholds based on advantage sign (`clip_ratio_low` or `clip_ratio_high`)
3. Apply dual clipping mechanism to handle extreme negative advantages
4. Call `agg_loss` to aggregate loss


