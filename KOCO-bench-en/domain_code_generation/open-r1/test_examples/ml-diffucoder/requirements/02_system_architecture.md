# DiffuCoder System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (ml-diffucoder Core)
```
code/src/open_r1/
├── grpo.py                         # Training entry and main flow
├── coupled_grpo.py                 # DiffuGRPOTrainer core implementation
├── configs.py                      # Training configuration (diffusion generation, GRPO parameters)
├── rewards.py                      # Reward function registry (code execution, format checking, etc.)
└── utils/
    ├── code_providers.py           # Code execution providers (E2B, local, MorphCloud)
    └── ioi.py                      # IOI problem evaluation tools

recipes/
├── config_coupled_code.yaml        # Training configuration file
├── process_data.py                 # Data preprocessing script
└── accelerate_configs/             # Distributed training configuration (DDP, FSDP, ZeRO)

tests/
└── test_code_reward.py             # Code reward function tests
```

### 1.2 Roles and Data Flow
- **Roles**
  - **Policy Model (π_θ)**: Policy model based on diffusion architecture, updated using Coupled-GRPO; supports FSDP/DDP parallel training.
  - **Reference Model (π_ref)**: Reference model, used to calculate KL divergence; can optionally synchronize policy model parameters (`sync_ref_model`).
  - **Reward Functions**: Collection of reward functions, including code execution reward (`code_reward`), format reward (`code_format_reward`), etc.
  - **Code Execution Provider**: Code execution backend, supports E2B, local execution, MorphCloud, etc.

- **Key Data**
  - **prompts/completions**: Input prompts and generated code responses.
  - **rewards**: Reward scores per sample (code execution pass rate, format correctness, etc.).
  - **per_token_logps**: Token-level log probabilities, calculated through coupled sampling.
  - **advantages**: Advantage estimates, calculated based on leave-one-out baseline.

## 2. Training Pipeline and Configuration Key Points

### 2.1 Training Entry and Mode Switching
- **Entry**: `python -m open_r1.grpo --config recipes/config_coupled_code.yaml`
- **Reward Mode Configuration** (`recipes/config_coupled_code.yaml`)
  - **Code Execution Reward**: `reward_funcs: [code, code_format]`, weights `reward_weights: [2.0, 0.5]`
  - **Code Execution Provider**: `code_provider: e2b` (supports `e2b`, `local`, `morph`)
  - **Parallel Execution Count**: `parallel_code_exec_per_proc: 2` (parallel code executions per process)

### 2.2 Coupled Sampling and Masking Strategy
- **Configuration Items**: `random_masking`, `p_mask_prompt`, `diffusion_steps`
  - `random_masking: True`: Use random mask seeds in multiple iterations to enhance robustness;
  - `p_mask_prompt: 0.0`: Mask probability for prompt tokens (default no masking of prompts);
  - `diffusion_steps: 256`: Number of diffusion generation steps.

- **Implementation Location**: `DiffuGRPOTrainer.forward_process` (`coupled_grpo.py`)
  - Generate three versions of masked sequences for each response:
    1. **Version 1**: Fully mask all completion tokens;
    2. **Version 2**: Mask completion tokens by random ratio `mask_ratio ∈ [0.2, 0.8]`;
    3. **Version 3**: Reverse mask (mask unmasked parts in Version 2).
  - Calculate weighted log probabilities: `weights = [1, 1/mask_ratio, 1/(1-mask_ratio)]`;
  - Use `selective_log_softmax` function to efficiently calculate weighted log probabilities, avoiding memory overflow.

### 2.3 Advantage Estimation and Loss
- **Advantage Estimation**: `leave-one-out` baseline
  - For k responses of same prompt, calculate `baseline = (sum(rewards) - reward_i) / (k-1)`;
  - Advantage `advantage_i = reward_i - baseline`;
  - Optional normalization: `advantages = advantages / (std + 1e-4)`.

- **PPO Loss**:
  - Calculate probability ratio `ratio = exp(logps - old_logps)`;
  - Clip ratio `clipped_ratio = clip(ratio, 1-ε, 1+ε)`;
  - Loss `loss = -min(ratio * advantages, clipped_ratio * advantages)`.

- **KL Divergence**:
  - If `β > 0`, calculate KL divergence `kl = exp(ref_logps - logps) - (ref_logps - logps) - 1`;
  - Total loss `loss = loss + β * kl`.

### 2.4 Diffusion Generation Configuration
- **Generation Parameters**: `max_completion_length`, `diffusion_steps`, `generation_temperature`
  - `max_completion_length: 256`: Maximum generation length;
  - `diffusion_steps: 256`: Diffusion steps (more steps → higher generation quality, but slower speed);
  - `generation_temperature: 1.0`: Generation temperature (affects sampling diversity and generation order).

- **Generation Algorithm**: `alg="entropy"`, `alg_temp=0.0`
  - Entropy-based mask selection strategy, prioritizing denoising of high-uncertainty tokens.

## 3. Reward Function System

### 3.1 Code Execution Reward (`code_reward`)
- **Function**: Execute generated code and calculate test case pass rate;
- **Flow**:
  1. Extract code blocks (use regex to match ` ```python\n...\n``` `)
  2. Build evaluation script (containing code and test cases)
  3. Execute script through code execution provider (E2B/local/MorphCloud)
  4. Parse pass rate `__PASS_RATE__` from output
  5. Return pass rate as reward (range [0, 1])

### 3.2 Code Format Reward (`code_format_reward`)
- **Function**: Check if code format is correct (contains code block markers, syntax is correct);
- **Flow**:
  1. Check if matches format pattern ` ^...```python\n...\n```\n$ `
  2. Extract code block
  3. For Python code, use `ast.parse` to check syntax
  4. Return reward: format correct and syntax correct → 1.0; format correct but syntax error → 0.5; format incorrect → 0.0

### 3.3 Other Reward Functions
- **IOI Code Reward** (`ioi_code_reward`): Code evaluation for IOI competition problems;
- **Binary Code Reward** (`binary_code_reward`): Binarize code execution reward (pass rate > 0.99 → 1.0, otherwise → 0.0);
- **Overlong Punishment** (`soft_overlong_punishment`): Soft penalty for overly long generations.

## 4. Code Execution Providers

### 4.1 E2B Provider
- **Features**: Cloud sandbox execution, supports multiple languages (Python, JavaScript, R, Java, Bash, C++);
- **Configuration**: Requires E2B API Key (environment variable `E2B_API_KEY`);
- **Routing**: Supports local routing server (`e2b_router_url`) to improve concurrent performance.

### 4.2 Local Provider
- **Features**: Local execution, no external dependencies;
- **Limitations**: Lower security, only suitable for trusted code.

### 4.3 MorphCloud Provider
- **Features**: Supports IOI problem evaluation, provides high-performance computing resources;
- **Configuration**: Requires MorphCloud API Key.
