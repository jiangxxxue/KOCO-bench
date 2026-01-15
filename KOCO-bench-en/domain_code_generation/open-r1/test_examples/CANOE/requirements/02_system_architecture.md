# CANOE System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (CANOE Core)
```
code/train/
├── src/open_r1/
│   ├── grpo.py                      # GRPO training entry and Dual-GRPO implementation
│   ├── configs.py                   # Training configuration classes (GRPOConfig extension)
│   ├── sft.py                       # Supervised fine-tuning script
│   ├── generate.py                  # Data generation script
│   └── utils/
│       ├── callbacks.py             # Training callback functions
│       ├── evaluation.py            # Evaluation tools
│       └── hub.py                   # HuggingFace Hub integration
├── TRL/                             # Modified TRL library
│   └── trl/trainer/
│       ├── grpo_trainer.py          # GRPO Trainer core implementation (supports Dual-GRPO)
│       └── grpo_config.py           # GRPO configuration
├── recipes/                         # Training configuration files
│   ├── LLama/
│   │   └── LLama-Instruct/
│   │       └── llama3_8b_2epoch_10k.yaml
│   ├── Qwen/
│   │   └── Qwen2.5-Instruct/
│   │       ├── qwen_7b_2epoch_10k.yaml
│   │       └── qwen_14b_2epoch_10k.yaml
│   └── accelerate_configs/          # DeepSpeed/DDP configuration
│       ├── zero2.yaml
│       └── zero3.yaml
├── train_data/                      # Training data directory
│   └── final_10k.jsonl             # Synthesized training data
└── scripts/
    ├── generate_reasoning.py        # Inference generation script
    └── run_benchmarks.py            # Benchmark testing script

code/eval/                           # Evaluation code
├── ConFiQA_FiQA/                   # Counterfactual and factual QA evaluation
├── CNQ/                            # Counterfactual multiple-choice evaluation
├── FaithEval/                      # Counterfactual QA evaluation
├── FollowRAG/                      # RAG generation evaluation
├── CLAPNQ/                         # Long-context QA evaluation
└── XSum_WiKiLarge/                 # Summarization and simplification evaluation
```

### 1.2 Core Modules and Data Flow

#### 1.2.1 Training Module
- **GRPO Trainer**: Based on TRL's GRPOTrainer, extended to support Dual-GRPO
  - First stage: Generate short-text answers (containing <think>, <long_answer>, <answer> tags)
  - Second stage: Extract <long_answer> to replace context, regenerate to evaluate influence
  - Supports vLLM accelerated inference (use_vllm=True)
  - Supports DeepSpeed ZeRO-2/ZeRO-3 parallel training

- **Reward Function Module**:
  - `accuracy_reward`: Verify answer correctness (symbolic verification + string matching)
  - `format_reward`: Check output format (<think><long_answer><answer> structure)
  - `influence_reward`: Evaluate the influence of long answer on final answer
  - `length_reward`: Constrain long answer length within reasonable range

#### 1.2.2 Data Format
- **Input Data**:
  ```json
  {
    "problem": "<context>Context content</context>\n\nQuestion description",
    "solution": "<answer>Correct answer</answer>"
  }
  ```

- **System Prompt** (SYSTEM_PROMPT):
  ```
  A conversation between User and Assistant. The user gives an instruction 
  that consists of two parts: a passage and the actual instruction, 
  separated by two newline characters.
  
  The passage is provided within <context> and </context> tags. 
  The Assistant need to refer to the given passage and complete the instruction.
  
  The response must be structured and include the following three sections:
  - Reasoning Process: <think>reasoning process</think>
  - Long Answer: <long_answer>detailed answer</long_answer>
  - Short Answer: <answer>concise answer</answer>
  ```

- **Generated Output**:
  ```
  <think>reasoning steps...</think>
  <long_answer>complete, grammatically and semantically complete sentences...</long_answer>
  <answer>concise direct answer</answer>
  ```

## 2. Training Pipeline and Configuration Key Points

### 2.1 Training Entry and Mode
- Entry: `accelerate launch src/open_r1/grpo.py --config recipes/xxx.yaml`
- Key configuration parameters (`recipes/xxx.yaml`):
  ```yaml
  # Model configuration
  model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  
  # GRPO configuration
  use_vllm: true                      # Use vLLM acceleration
  vllm_gpu_memory_utilization: 0.7    # GPU memory utilization
  num_generations: 7                  # Generate 7 responses per prompt
  do_long_answer: true                # Enable Dual-GRPO (long answer generation)
  
  # Training hyperparameters
  learning_rate: 1.0e-06
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_prompt_length: 1024
  max_completion_length: 1024
  
  # Reward functions
  reward_funcs: [accuracy, format, influence, length]
  ```

### 2.2 Dual-GRPO Implementation Details
- **First Stage Generation** (grpo.py):
  - Use standard GRPO to generate n short-text responses
  - Calculate accuracy_reward and format_reward
  
- **Second Stage Generation** (extension in grpo_trainer.py):
  - Triggered when `do_long_answer=True`
  - Extract `<long_answer>` content from first stage generation
  - Use regular expression to replace original context:
    ```python
    pattern = re.compile(r'<context>.*?</context>', flags=re.DOTALL)
    new_context = pattern.sub(f'<context>{long_answer}</context>', original_context)
    ```
  - Regenerate based on new context (same temperature, n=1)
  - Calculate influence_reward and length_reward

- **Reward Fusion**:
  - All reward functions return binary rewards of 0.0 or 1.0
  - Total reward = accuracy + format + influence + length (range 0-4)
  - GRPO uses total reward to calculate group-wise relative advantage
