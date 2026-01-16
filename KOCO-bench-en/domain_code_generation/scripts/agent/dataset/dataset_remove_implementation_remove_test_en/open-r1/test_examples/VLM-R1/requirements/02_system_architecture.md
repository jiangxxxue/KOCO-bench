# VLM-R1 System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization (VLM-R1 Core)
```
code/
├── src/
│   ├── open-r1-multimodal/
│   │   └── src/open_r1/
│   │       ├── grpo_jsonl.py           # GRPO training main entry (supports JSONL data format)
│   │       ├── grpo_rec.py             # REC task-specific training script
│   │       ├── grpo.py                 # General GRPO training script
│   │       ├── configs.py              # Training configuration classes (GRPOConfig, SFTConfig)
│   │       ├── sft.py                  # Supervised fine-tuning script
│   │       ├── generate.py             # Inference generation script
│   │       ├── evaluate.py             # Evaluation tools
│   │       ├── qwen2_5vl_monkey_patch.py  # Qwen2.5-VL model patch
│   │       ├── trainer/
│   │       │   ├── grpo_trainer.py     # VLM GRPO Trainer implementation
│   │       │   └── grpo_config.py      # GRPO configuration
│   │       ├── vlm_modules/            # Vision-language model modules
│   │       │   ├── vlm_module.py       # VLM base class (abstract interface)
│   │       │   ├── qwen_module.py      # Qwen2-VL/Qwen2.5-VL module
│   │       │   ├── internvl_module.py  # InternVL module
│   │       │   └── glm_module.py       # GLM-V module
│   │       └── utils/
│   │           ├── callbacks.py        # Training callback functions
│   │           ├── evaluation.py       # Evaluation tools
│   │           ├── hub.py              # HuggingFace Hub integration
│   │           └── pycocotools/        # COCO evaluation tools
│   └── eval/                           # Evaluation scripts
│       ├── test_rec_r1.py              # REC task evaluation (R1 model)
│       ├── test_rec_baseline.py        # REC task evaluation (SFT baseline)
│       ├── test_rec_r1_internvl.py     # REC task evaluation (InternVL)
│       └── test_od_r1.py               # OVD task evaluation
├── run_scripts/                        # Training scripts
│   ├── run_grpo_rec.sh                 # REC task full fine-tuning
│   ├── run_grpo_rec_lora.sh            # REC task LoRA fine-tuning
│   ├── run_grpo_rec_internvl.sh        # InternVL REC training
│   ├── run_grpo_rec_more_params.sh     # More parameter configuration examples
│   ├── run_grpo_gui.sh                 # GUI defect detection (multi-image input)
│   ├── run_grpo_gui_defect_detection.sh # GUI defect detection specific
│   ├── multinode_training_demo.sh      # Multi-node training example
│   └── multinode_training_args.yaml    # Multi-node training configuration
├── ascend_inference/                   # Huawei Ascend inference adaptation
│   ├── 910B/                           # Atlas 800T A2 adaptation
│   │   ├── vllm_ascend/                # vLLM-Ascend framework
│   │   └── xllm/                       # xLLM framework
│   └── 300IDuo/                        # Atlas 300I Duo adaptation
├── assets/                             # Documentation resources
│   └── add_new_model.md                # Guide for adding new models
└── README.md                           # Project documentation
```

### 1.2 Core Modules and Data Flow

#### 1.2.1 VLM Module Abstraction Layer
VLM-R1 implements a unified interface for different vision-language models through the `VLMBaseModule` abstract base class:

- **Model Initialization**:
  - `get_model_class()`: Returns model class (e.g., Qwen2VLForConditionalGeneration)
  - `get_processing_class()`: Returns processor class (e.g., AutoProcessor)
  - `post_model_init()`: Custom processing after model initialization

- **Model Configuration**:
  - `get_vision_modules_keywords()`: Returns vision module keywords (for freezing)
  - `get_custom_multimodal_keywords()`: Returns multimodal-specific parameters
  - `get_custom_processing_keywords()`: Returns processor configuration parameters

- **Input Processing**:
  - `prepare_prompt()`: Prepare prompt text
  - `prepare_model_inputs()`: Process multimodal inputs (image + text)

- **Task-Specific Methods**:
  - `get_question_template()`: Returns question template based on task type
  - `select_reward_func()`: Selects reward function based on task type

#### 1.2.2 Training Module
- **VLMGRPOTrainer**: Extended from TRL's GRPOTrainer, specifically optimized for vision-language models
  - Supports multimodal input processing (lazy loading of image paths)
  - Supports vision module freezing (freeze_vision_modules)
  - Supports LoRA fine-tuning (via peft_config)
  - Supports vLLM accelerated inference (use_vllm=True)
  - Supports DeepSpeed ZeRO-2/ZeRO-3 parallel training

- **Reward Function Module**:
  - `accuracy_reward`: Selects different accuracy verification methods based on task type
    - `default_accuracy_reward`: Symbolic verification + fuzzy matching
    - `mcq_reward`: Multiple-choice option extraction and matching
    - `yes_no_reward`: Yes/no question matching
    - `llm_reward`: Use LLM to evaluate answer similarity
    - `map_reward`: Calculate object detection mAP
    - `od_reward`: Object detection accuracy (mAP or mAP@50)
    - `odLength_reward`: Object detection reward with length penalty
    - `math_reward`: Math problem symbolic verification
    - `numeric_reward`: Numerical exact matching
    - `all_match_reward`: Text exact matching
  - `format_reward`: Check output format (`<think><answer>` structure)
  - `cosine_rewards`: Cosine-based length reward
  - `repetition_rewards`: Repetitive content penalty

#### 1.2.3 Data Format
- **Input Data (JSONL format)**:
  ```json
  {
    "id": 1,
    "image": "path/to/image.jpg",  // or ["path1.jpg", "path2.jpg"] for multi-image
    "conversations": [
      {"from": "human", "value": "<image>Question description"},
      {"from": "gpt", "value": "Standard answer"}
    ],
    "accu_reward_method": "default"  // Optional, specify accuracy reward method
  }
  ```

- **System Prompt** (default):
  ```
  A conversation between User and Assistant. The user asks a question, and the 
  Assistant solves it. The assistant first thinks about the reasoning process in 
  the mind and then provides the user with the answer. The reasoning process and 
  answer are enclosed within <think> </think> and <answer> </answer> tags, 
  respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
  ```

- **Task-Specific Prompt Templates**:
  - REC task: `"{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."`
  - OVD task: Includes detailed reasoning process requirements
  - Other tasks: Customized according to requirements

- **Generated Output**:
  ```
  <think>reasoning steps and analysis process...</think>
  <answer>final answer (may be text, numbers, JSON-formatted bounding boxes, etc.)</answer>
  ```

## 2. Training Pipeline and Configuration Key Points

### 2.1 Training Entry and Mode
- Entry: `torchrun --nproc_per_node=X src/open_r1/grpo_jsonl.py --config ...`
- Key configuration parameters:
  ```yaml
  # Model configuration
  model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  freeze_vision_modules: false  # Whether to freeze vision modules
  
  # Data configuration
  data_file_paths: "path1.jsonl:path2.jsonl"  # Multiple data files separated by ":"
  image_folders: "path1/:path2/"              # Corresponding image folders
  reward_method: "default:mcq"                # Reward method for each dataset
  task_type: "rec"                            # Task type
  is_reward_customized_from_vlm_module: false # Whether to use VLM module custom reward
  
  # GRPO configuration
  use_vllm: true                              # Use vLLM acceleration
  vllm_gpu_memory_utilization: 0.7            # GPU memory utilization
  num_generations: 7                          # Generate 7 responses per prompt
  
  # Training hyperparameters
  learning_rate: 1.0e-06
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_prompt_length: 1024
  max_completion_length: 1024
  
  # Reward functions
  reward_funcs: [accuracy, format, length, repetition]
  
  # Vision processing parameters (Qwen2-VL)
  max_pixels: 12845056
  min_pixels: 3136
  
  # Vision processing parameters (InternVL)
  max_anyres_num: 12
  
  # DeepSpeed configuration
  deepspeed: path/to/zero3.json
  ```

### 2.2 Multi-Task Reward Mechanism

#### 2.2.1 REC Task (Referring Expression Comprehension)
- **Accuracy Reward**: Calculate IoU between predicted bounding box and ground truth
  - Extract bounding box coordinates `[x1, y1, x2, y2]` from model output
  - Adjust bounding box coordinates based on actual image size (resize operation)
  - Calculate IoU value as reward (range 0-1)
- **Format Reward**: Check if output contains `<think>...<answer>...[x1, y1, x2, y2]...</answer>` format

#### 2.2.2 OVD Task (Open-Vocabulary Object Detection)
- **Accuracy Reward**: Calculate mAP or mAP@50
  - Parse JSON-formatted bounding box list
  - Use COCO evaluation tools to calculate mAP
  - Support variant with length penalty (odLength_reward)
- **Weighted Sum Reward** (weighted_sum):
  - Position accuracy (alpha=0.7): Average IoU
  - Label accuracy (beta=0.0): Category matching rate
  - Completeness (gamma=0.3): Consider missed detections and false positives
  - Final score = (alpha × position score + beta × label score + gamma × completeness score) / (alpha + beta + gamma)

#### 2.2.3 Math Task (Multimodal Math Reasoning)
- **Accuracy Reward**: Symbolic verification
  - Use `math_verify` library for symbolic parsing and verification
  - Support numerical exact matching as fallback

#### 2.2.4 General Rewards
- **Length Reward** (cosine_reward):
  - Cosine-based length control
  - Correct answers: shorter length → higher reward (1.0 → 0.5)
  - Incorrect answers: shorter length → greater penalty (0.0 → -0.5)
  - Formula: `reward = max_value - (max_value - min_value) × (1 - cos(len × π / max_len)) / 2`

- **Repetition Penalty** (repetition_reward):
  - Use n-gram statistics to detect repetitive content
  - For JSON data: n-gram_size=1 (detect repeated bounding boxes)
  - For text data: n-gram_size=6 (detect repeated phrases)
  - Penalty formula: `reward = (1 - unique_ngrams / total_ngrams) × max_penalty`
  - max_penalty = -1.0

### 2.3 Modular Extension Mechanism

#### 2.3.1 Adding New Models
1. Create new module class under `vlm_modules/`, inherit from `VLMBaseModule`
2. Implement all abstract methods (model loading, input processing, etc.)
3. Register new model in `get_vlm_module()` function
4. Optional: Implement task-specific reward functions

#### 2.3.2 Adding New Tasks
1. Add task-specific prompt template in `get_question_template()`
2. Implement task-specific reward function
3. Register reward function in `reward_funcs_registry`
4. Specify `task_type` and `reward_method` in training script

### 2.4 Efficient Training Strategies

#### 2.4.1 Vision Module Freezing
- Set `freeze_vision_modules=true`
- Only train language model part, significantly reduce memory usage and training time
- Suitable for scenarios where visual features are already good enough

#### 2.4.2 LoRA Fine-tuning
- Configure LoRA parameters via `peft_config`
- Only train a small number of adapter parameters
- Significantly reduce memory requirements, support larger batch sizes

#### 2.4.3 Multi-Node Training
- Use `torchrun` to launch multi-node training
- Configure `--nnodes`, `--node_rank`, `--master_addr`, `--master_port`
- Support cross-node gradient synchronization and data parallelism

#### 2.4.4 vLLM Acceleration
- Set `use_vllm=true`
- Use vLLM engine to accelerate inference generation phase
- Significantly improve generation throughput, shorten training time
