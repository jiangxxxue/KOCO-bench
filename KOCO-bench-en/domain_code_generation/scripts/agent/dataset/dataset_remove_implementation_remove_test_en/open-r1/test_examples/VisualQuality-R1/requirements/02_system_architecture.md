# VisualQuality-R1 System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (VisualQuality-R1 Core)
```
code/src/open-r1-multimodal/
├── src/open_r1/
│   ├── grpo_jsonl.py               # Training entry, data loading and reward function definition
│   ├── trainer/
│   │   ├── grpo_trainer.py         # VLMGRPOTrainer core training logic
│   │   ├── grpo_config.py          # GRPO training configuration parameters
│   │   └── vllm_grpo_trainer.py    # vLLM accelerated version (optional)
│   ├── vlm_modules/
│   │   ├── vlm_module.py           # VLM base module interface
│   │   ├── qwen_module.py          # Qwen2.5-VL model adaptation
│   │   └── internvl_module.py      # InternVL model adaptation (optional)
│   ├── utils/
│   │   ├── evaluation.py           # Evaluation tools
│   │   ├── math.py                 # Mathematical calculation tools
│   │   └── callbacks.py            # Training callbacks
│   └── qwen2_5vl_monkey_patch.py   # Qwen2.5-VL Flash Attention optimization
├── run_scripts/
│   └── KADID-10K/
│       ├── one_node_run_kadid.sh   # Single-node training script
│       └── multi_run_kadid.sh      # Multi-node training script
├── configs/
│   ├── qwen2vl_sft_config.yaml     # Model configuration
│   └── zero3.json                  # DeepSpeed ZeRO-3 configuration
└── datasets/
    ├── make_data.py                # Dataset preprocessing script
    └── KADID-10K/                  # Example dataset
```

### 1.2 Roles and Data Flow
- Roles
  - Policy Model (π_θ): Policy model based on Qwen2.5-VL-7B-Instruct, updated using GRPO; supports LoRA fine-tuning and vision module freezing.
  - Reference Model (π_ref): Reference model, used to calculate KL divergence; when using LoRA, can be obtained by disabling adapter.
  - VLM Module: Multimodal model adaptation layer, processes image and text inputs, generates structured responses.
  - Reward Functions: Collection of reward functions, including fidelity reward (accuracy_reward) and format reward (format_reward).

- Key Data
  - prompts: Multimodal inputs containing images and text prompts.
  - responses: Generated structured responses, containing `<think>reasoning process</think>` and `<answer>score</answer>`.
  - predictions: Quality scores extracted from responses (floats between 1-5).
  - fidelity_rewards: Fidelity rewards calculated based on pairwise ranking.
  - format_rewards: Rewards for checking if response format meets requirements.

## 2. Training Pipeline and Configuration Key Points

### 2.1 Training Entry and Mode Switching
- Entry: `python -m open_r1.grpo_jsonl` (launch distributed training via torchrun)
- Key configuration (`run_scripts/KADID-10K/one_node_run_kadid.sh`)
  - Model path: `--model_name_or_path` specifies Qwen2.5-VL-7B-Instruct path
  - Data path: `--data_file_paths` and `--image_folders` specify training data
  - Generation count: `--num_generations` sets number of responses generated per image (default 6)
  - Vision module: `--freeze_vision_modules` controls whether to freeze vision encoder
  - Question template: `--question_template scoring` specifies using scoring task template

### 2.2 Reward Function Design
- Fidelity Reward
  - Location: `grpo_jsonl.py::fidelity_reward` and `grpo_jsonl.py::accuracy_reward`
  - Calculation flow:
    1. Extract predicted score from each response (parse from `<answer>` tag)
    2. Calculate mean μ and variance σ² of G predictions for each sample
    3. For each pair of samples (i, k) in batch, calculate ranking probability: p = Φ((pred_i - μ_k) / √(σ²_i + σ²_k))
    4. Determine ground truth ranking relationship gt based on true MOS (1.0 means i better than k, 0.0 means k better than i, 0.5 means equal)
    5. Calculate fidelity: r = √(p·gt) + √((1-p)·(1-gt))
    6. Average over all other samples to get final reward

- Format Reward
  - Location: `grpo_jsonl.py::format_reward`
  - Check if response conforms to `<think>...</think><answer>...</answer>` format
  - Returns 1.0 if format is correct, otherwise 0.0

### 2.3 GRPO Training Loop
- Configuration: `GRPOConfig` (`trainer/grpo_config.py`)
  - `num_generations`: Number of responses generated per prompt (must be divisible by global batch size)
  - `num_iterations`: Number of update iterations per batch (μ)
  - `beta`: KL divergence coefficient (controls deviation from reference model)
  - `epsilon` / `epsilon_high`: PPO clipping range
  - `max_completion_length`: Maximum length of generated response

- Implementation location: `VLMGRPOTrainer` (`trainer/grpo_trainer.py`)
  - Generation and scoring: `_generate_and_score_completions` method
    - Use vLLM or HuggingFace generator to generate G responses
    - Call reward functions to calculate reward for each response
    - Calculate group-wise advantage
  - Loss calculation: `compute_loss` method
    - Calculate policy ratio ratio = exp(log_π_θ - log_π_old)
    - Apply PPO clipping: L_clip = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
    - Add KL penalty: L = -L_clip + β·KL
  - Sampler: `RepeatRandomSampler`
    - Ensures each GPU's batch can come from different datasets
    - Supports grouping by image prefix (for datasets like KADID-10K)
    - Repeat sampling to support multiple iterations

### 2.4 Data Preprocessing
- Data format: JSONL format, each line contains:
  - `id`: Sample ID
  - `dataset_name`: Dataset name
  - `image`: Image filename (single or multiple)
  - `conversations`: Question and answer in conversation format
    - `from: "human"`: Contains task description and question
    - `from: "gpt"`: Contains MOS score (normalized to 1-5)

- Normalization: `make_data.py::normalize`
  - Normalize original MOS scores to [1, 5] interval
  - Formula: normalized_mos = (mos - min_mos) / (max_mos - min_mos) × 4 + 1

- Data split: Split train/validation/test sets by reference image (6:2:2)

### 2.5 Multimodal Processing
- VLM Module interface: `VLMBaseModule` (`vlm_modules/vlm_module.py`)
  - `prepare_prompt`: Convert data to model input format
  - `prepare_model_inputs`: Process images and text, generate model input tensors
  - `get_vision_modules_keywords`: Return vision module keywords (for freezing or LoRA)
  - `get_custom_multimodal_keywords`: Return multimodal input keywords (e.g., `pixel_values`, `image_grid_thw`)

- Qwen2.5-VL adaptation: `Qwen2VLModule` (`vlm_modules/qwen_module.py`)
  - Supports dynamic resolution and multi-image input
  - Configure `max_pixels` and `min_pixels` to control image resolution
  - Use Flash Attention 2 to accelerate training

### 2.6 Optimization Techniques
- LoRA fine-tuning: Automatically identify all linear layers (excluding vision modules) as target_modules
- Vision module freezing: Freeze vision encoder through `freeze_vision_modules=True`, reduce memory usage
- Gradient checkpointing: Enable through `gradient_checkpointing=True`, reduce memory peak
- DeepSpeed ZeRO-3: Support large model training, configuration file `local_scripts/zero3.json`
- Flash Attention 2: Optimize Qwen2.5-VL attention calculation through monkey patch
