# CTRL System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization
```
code/ctrl/
├── sft/                                # SFT stage training scripts
│   └── zero3_accelerate.yaml          # DeepSpeed Zero3 configuration
├── rl/                                 # RL stage core modules (verl-based)
│   ├── __init__.py
│   ├── critic_ray_trainer.py          # Training loop and critique-revision mechanism
│   ├── critic_rm.py                    # Sandbox execution-based reward model
│   ├── critic_fsdp_workers.py         # FSDP Worker (Actor/Critic/Ref)
│   └── critic_vllm_rollout.py         # vLLM inference engine
├── gen/                                # Generation and inference tools
│   ├── api.py                          # API interface
│   ├── prompt.py                       # Prompt templates
│   ├── tree.py                         # Search tree structure
│   └── config/                         # Inference configuration
│       ├── critic/                     # Critique generation configuration
│       └── generator/                  # Code generation configuration
├── eval/                               # Evaluation tools
│   ├── eval_utils.py                   # Evaluation utility functions
│   └── sandbox_utils.py                # Sandbox execution interface
└── scripts/
    ├── run_sft.py                      # SFT training script
    ├── run_rl.py                       # RL training script
    ├── run_gen.py                      # Generation script
    └── rl_config/
        └── trainer.yaml                # RL training configuration
```

### 1.2 Roles and Data Flow (RL Stage)

#### Core Roles
- **Actor (Policy Model)**: Generates critique text, updated using PPO algorithm
- **Rollout (Inference Engine)**: Efficient batch generation based on vLLM
- **Reference/Proxy (Reference Model)**: Used to generate revised code and compute KL divergence
- **RewardModel (Reward Function)**: Verifiable reward computation based on sandbox execution

Note: CTRL uses **GRPO** algorithm, so no independent Critic model (value network) is needed.

#### Data Flow
```
1. Input batch (prompts)
   ↓
2. Actor generates critiques
   ↓
3. Build revision prompts (original problem + original code + critique)
   ↓
4. Reference generates revised code (revisions)
   ↓
5. Sandbox executes revised code (sandbox execution)
   ↓
6. Compute verifiable rewards (pass rate)
   ↓
7. Assign rewards to token level (last token)
   ↓
8. Compute KL penalty (old_log_prob vs ref_log_prob)
   ↓
9. GRPO advantage estimation (group-based baseline)
   ↓
10. PPO update Actor
```

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Points and Modes

#### SFT Stage
- **Entry point**: `python scripts/run_sft.py`
- **Framework**: Supervised fine-tuning based on DeepSpeed Zero3
- **Data**: Synthetic critique data (generated from execution feedback)
- **Objective**: Let model initially learn critique format and content

#### RL Stage
- **Entry point**: `python scripts/run_rl.py`
- **Framework**: Distributed RL training based on verl
- **Algorithm**: PPO + GRPO (no Critic needed)
- **Configuration file**: `scripts/rl_config/trainer.yaml`

### 2.2 Key Configuration Items

#### Data Configuration (`data`)
```yaml
train_files: Training data path (parquet format)
val_files: Validation data path
prompt_key: Prompt field name in data
max_prompt_length: Maximum prompt length
max_response_length: Maximum response length (critique length)
train_batch_size: Training batch size
```

#### Actor-Rollout-Ref Configuration (`actor_rollout_ref`)
```yaml
hybrid_engine: True                     # Use hybrid engine (Actor + Rollout + Ref)
model.path: Model path
actor.ppo_mini_batch_size: PPO mini-batch size
actor.ppo_epochs: PPO training epochs
actor.clip_ratio: PPO clip ratio
actor.entropy_coeff: Entropy regularization coefficient
actor.use_kl_loss: False                # GRPO doesn't use KL loss
rollout.name: vllm                      # Use vLLM inference
rollout.temperature: 1.0                # Sampling temperature
rollout.n: 1                            # Number of samples per prompt (GRPO can set >1)
```

#### Algorithm Configuration (`algorithm`)
```yaml
adv_estimator: grpo                     # Use GRPO advantage estimation
gamma: 1.0                              # Discount factor
lam: 1.0                                # GAE parameter (not used in GRPO)
kl_ctrl.kl_coef: 0.001                  # KL divergence penalty coefficient
```

### 2.3 Critique-Revision Mechanism Implementation

#### Core Flow (`critic_ray_trainer.py: fit()`)

Training loop executes complete critique-revision flow in each batch:

**Critique Generation Phase**: Calls `actor_rollout_wg.generate_sequences()` to use Actor model to generate critique text. After generation completes, parses critique content through `reward_fn.parse_response()`, extracting problem ID, critique text, and valid response length. If multi-sample generation is configured (n > 1), batch is expanded accordingly. For parallel efficiency optimization, performs sequence length balancing on batch to ensure similar total token counts across GPUs.

**Build Revision Prompts**: Obtains original problem and test sample information from `reward_fn.parse_query()`. Calls `build_revision_messages()` to combine problem, initial code, and critique into dialogue format. Uses tokenizer's `apply_chat_template()` method to format prompts, generating input sequences conforming to model training format.

**Generate Revised Code**: Calls `actor_rollout_wg.generate_sequences_ref()` to use reference model or proxy model to generate revisions. This method internally temporarily increases maximum generation length to 1024 tokens to accommodate complete code, using greedy decoding (do_sample=False) to ensure generation stability. After generation completes, decodes token sequences to text through tokenizer's `batch_decode()`.

**Asynchronous Sandbox Execution**: Creates asynchronous event loop, calls `reward_fn.get_reward_all()` to concurrently execute tests for critique-revision pairs. While waiting for sandbox results, computes reference policy log probabilities in parallel (compute_ref_log_prob), fully utilizing wait time. After completion, obtains reward list for all samples, with each element representing test pass rate for corresponding revised code.

**Reward Assignment**: Creates zero tensor reward_tensor with same shape as responses. Iterates through each sample in batch, assigning sequence-level reward to last valid token position of critique, keeping other positions at zero. This assignment method reflects that quality of entire critique sequence is determined by final result.

### 2.4 Verifiable Reward Computation (`critic_rm.py`)

#### Reward Model Design
RewardModelForCritic class implements verifiable reward computation based on sandbox execution. Input is critique text, revised code, and test samples; output is float between 0 and 1, representing test case pass rate.

#### Reward Computation Flow (`reward_revision`)

**Critique Validity Check**: First checks if critique text contains explicit judgment identifiers, such as "Overall judgment: Correct" or "Overall judgment: Incorrect". These identifiers are necessary components of critique format, indicating critic made explicit correctness judgment. If critique lacks these identifiers, directly returns zero reward, encouraging model to generate structured, explicit critiques.

**Code Extraction and Normalization**: Calls `sanitize()` function to extract code blocks from revision text, removing markdown format markers. Uses `normalize_code()` for Abstract Syntax Tree (AST) level code normalization: parses code to generate AST, unifies variable naming (e.g., v_0, v_1), removes redundant whitespace, regenerates standard format code string. Normalized code is semantically equivalent to original code but improves cache hit rate.

**Cache Query**: Uses combination of normalized code and test cases as key to look up historical execution results in code_to_reward dictionary. If match found, directly returns cached reward value, skipping sandbox execution. This is particularly effective in later training stages, as model may generate duplicate or similar code.

**Sandbox Execution**: Constructs corresponding execution requests based on dataset type. MBPP+ and APPS use RunCodeRequest, concatenating code with test cases for execution. LiveCodeBench uses SubmitRequest with LiveCodeBenchDataset type, providing problem metadata. CodeContests uses CommonOJDataset for OJ evaluation. Asynchronously calls `submit_to_sandbox()` to send request to sandbox service, waiting for execution results.

**Reward Aggregation**: Computes reward value based on execution results. If configured to run all test cases, computes average pass rate; otherwise uses binary reward (1 if all pass, 0 otherwise). Finally updates computation result to cache dictionary for subsequent queries.

#### Supported Dataset Formats
- CodeContests: Contest-style evaluation, uses CommonOJDataset execution
- LiveCodeBench: Real-time programming challenges, uses LiveCodeBenchDataset execution  
- MBPP+/APPS: Unit test style, directly concatenates test code for execution

#### Performance Optimization
Code hash caching mechanism avoids repeated execution of same code. Normalization unifies code style through AST transformation, enabling functionally identical but differently written code to be recognized as equivalent. Asynchronous concurrent execution fully utilizes I/O wait time, semaphore control prevents sandbox service overload.

### 2.5 Hybrid Engine Implementation (`critic_fsdp_workers.py`)

#### ActorRolloutRefWorker
This is a unified Worker that can play different roles based on `role` parameter:
- `role='actor_rollout_ref'`: Simultaneously supports Actor, Rollout, and Ref functionality
- Uses FSDP for parameter management and communication
- Supports parameter/gradient/optimizer offloading to save memory

#### Key Methods
- `generate_sequences()`: Use vLLM to generate critiques
- `generate_sequences_ref()`: Use Proxy model to generate revisions (longer max_tokens)
- `compute_log_prob()`: Compute old_log_probs
- `compute_ref_log_prob()`: Compute ref_log_probs (for KL penalty)
- `update_actor()`: PPO update

#### Proxy Model Mechanism

Configuration file can specify different path for Proxy model (proxy.model.path) than Actor, allowing use of stronger base model to generate revisions. When generating revisions, use_proxy parameter instructs system to switch models. In internal implementation, calls rollout_sharding_manager's update_module method to bind inference engine to proxy_module_fsdp parameters. This design allows Actor to focus on learning critique while revision quality is guaranteed by stronger model, simplifying training logic.


