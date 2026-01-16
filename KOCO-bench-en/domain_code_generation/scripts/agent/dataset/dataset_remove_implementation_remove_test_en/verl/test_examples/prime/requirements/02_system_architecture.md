# PRIME Algorithm System Architecture and Module Design

## 1. PRIME Module File Structure

### 1.1 File Organization
```
recipe/prime/
├── main_prime.py              # PRIME algorithm entry point
├── prime_ray_trainer.py       # PRIME trainer implementation
├── prime_core_algos.py        # PRIME core algorithm functions
├── prime_dp_rm.py             # PRIME reward model implementation
├── prime_fsdp_workers.py      # PRIME worker implementation
└── config/
    └── prime_trainer.yaml     # PRIME configuration file
```

### 1.2 Module Responsibilities
- main_prime.py: PRIME algorithm startup logic and environment configuration
- prime_ray_trainer.py: Inherits PPO trainer and implements PRIME functionality
- prime_core_algos.py: Collection of pure function implementations for PRIME algorithm
- prime_dp_rm.py: Core implementation of PRIME implicit reward model
- prime_fsdp_workers.py: Distributed worker wrapper for PRIME reward model

## 2. PRIME Module Design

### 2.1 PRIME Trainer (prime_ray_trainer.py)

Core class: `RayPRIMETrainer`

Main features:
- Disable Critic: `self.use_critic = False` - Remove value function network
- Oversampling data loading: Data batch size multiplied by oversample factor for loading
- Intelligent filtering strategy: `filter_and_downsample()` - Dual filtering based on accuracy and length
- Joint reward model update: Synchronously update reward model parameters before advantage computation

PRIME functions:
- `filter_and_downsample()`: Implements intelligent filtering and priority ranking after oversampling
- `_create_dataloader()`: Customized data loader supporting oversample factor

Training loop implementation:
- Filter and downsample after data verification
- Flexible scheduling of reward model update and score computation
- Multi-source reward advantage computation integration

### 2.2 PRIME Reward Model (prime_dp_rm.py)

Core class: `DataParallelPRIMERewardModel`

Main methods:
- `_forward_micro_batch()`: Core implementation of implicit reward computation
- `prime_norm()`: PRIME's reverse cumulative sum normalization
- `compute_rm_score()`: Reward score computation with PRIME normalization
- `update_rm()`: Model update supporting multiple PRIME loss functions

Implicit reward computation implementation:
- Simultaneously compute log probabilities of reward model and reference model
- Generate implicit reward signals through log probability differences
- Support both GAE temporal modeling and direct reward modes
- Flexible reward granularity allocation (token-level or whole-level)

### 2.3 PRIME Algorithm Function Library (prime_core_algos.py)

Algorithm functions:
- `compute_rloo_advantage_return()`: Multi-source reward RLOO advantage computation
- `compute_ce_dpo_loss_rm()`: CE-style implicit reward loss
- `compute_detach_dpo_loss_rm()`: Detached DPO loss supporting BoN mode
- `compute_dpo_accuracy()`: Reward model accuracy evaluation through pairwise comparison
- `compute_dpo_abs_accuracy()`: Simplified absolute accuracy evaluation

Algorithm implementation:
- Multi-source reward fusion: Simultaneously process rm_scores and accuracy reward sources
- RLOO whitening: Normalize advantage function to distribution with mean≈0, std≈1
- BoN mode support: Assign different training weights based on sample ranking
- Pairwise comparison evaluation: Evaluate reward model performance through sample comparison
- GAE temporal modeling: Support GAE mode reverse cumulative reward computation

### 2.4 PRIME Worker (prime_fsdp_workers.py)

Core class: `PRIMERewardModelWorker`

Main interfaces:
- `compute_rm_score()`: PRIME reward score computation in distributed environment
- `update_rm()`: PRIME reward model update in distributed environment
- `init_model()`: Initialization of PRIME reward model and reference model

Integration implementation:
- Wrap `DataParallelPRIMERewardModel` as distributed worker
- Integrate PRIME's DPO accuracy evaluation metrics
- Support PRIME configuration's ulysses sequence parallelism and dynamic batch processing

### 2.5 PRIME Configuration System (config/prime_trainer.yaml)

Configuration structure description: PRIME algorithm adopts hierarchical configuration structure, algorithm-related parameters are located under `algorithm` subnode.

PRIME core configuration:
- `algorithm.reward_dpo_coef`: DPO reward weight coefficient, default 5.0, used for multi-source reward fusion
- `algorithm.reward_gt_coef`: Accuracy reward weight coefficient, default 5.0, used for multi-source reward fusion
- `prime_granularity`: Reward granularity control (token/whole)
- `prime_norm`: PRIME normalization mode (batch_norm/none)
- `reward_manager`: Specify using prime reward manager
- `beta_train`: PRIME temperature parameter, default 0.05
- `loss_type`: PRIME loss type (ce/dpo/bon_acc/bon_rm)

Data processing configuration:
- `oversample_factor`: Oversample multiplier configuration, default 4.0
- `filter_accuracy`: Accuracy filtering switch, default True
- `accuracy_lower_bound`: Accuracy filtering lower bound, default 0.2
- `accuracy_upper_bound`: Accuracy filtering upper bound, default 0.8
- `filter_truncate`: Length filtering switch, default True
- `max_response_length`: Maximum response length threshold

Algorithm configuration:
- `adv_estimator`: Fixed to use "rloo" advantage estimation algorithm
- `n_samples`: Sample count per group, default 4

## 3. PRIME Data Protocol and Module Interaction

### 3.1 PRIME Data Protocol (DataProto)

Data protocol structure: PRIME algorithm uses standardized data protocol to ensure data transfer consistency between modules.

DataProto.batch field specification:
```python
data.batch = {
    "rm_scores": torch.Tensor,           # Implicit reward model scores [batch_size, seq_len]
    "acc": torch.Tensor,                 # Binary accuracy labels [batch_size]  
    "prompts": torch.Tensor,             # Input prompt sequences [batch_size, prompt_len]
    "attention_mask": torch.Tensor,      # Attention mask [batch_size, total_len]
    "responses": torch.Tensor,           # Response sequences [batch_size, response_len]
    "input_ids": torch.Tensor           # Complete input sequences [batch_size, total_len]
}
```

Batch data organization:
- Grouping structure: Data is grouped by n_samples, each group contains multiple responses from the same prompt
- Indexing pattern: Samples [i*n_samples:(i+1)*n_samples] belong to group i
- RLOO computation: Use grouping structure to compute leave-one-out baseline

### 3.2 PRIME Training Flow
1. Oversampling loading: Load more data by oversample factor, maintain grouping structure
2. PRIME verification: Use PrimeRewardManager to verify response accuracy
3. Intelligent filtering: Filter based on accuracy and length, maintain group integrity
4. Implicit reward: Compute rewards through dual-model log probability differences
5. PRIME normalization: Apply reverse cumulative sum normalization strategy
6. Multi-source advantage: Fuse rm_scores and accuracy to compute RLOO advantage function

### 3.3 PRIME Data Flow
- Input: Grouped raw data × oversample_factor
- Filtering: Dual filtering based on accuracy and length, maintain grouping structure
- Downsampling: Retain 1/oversample_factor of high-quality sample groups
- Reward: Generate implicit process rewards and binary accuracy labels
- Advantage: Use grouping structure to compute multi-source fused RLOO advantage function

### 3.4 PRIME Update Strategy
- Reward model update first: In update="before" mode, update RM first then compute rewards
- Reverse update mode: In update="reverse" mode, compute reward statistics first, then update reward model, supporting within-batch sample comparison
- Joint training: Simultaneously update Actor and reward model at each training step
- No Critic dependency: Completely remove dependency on value function network

