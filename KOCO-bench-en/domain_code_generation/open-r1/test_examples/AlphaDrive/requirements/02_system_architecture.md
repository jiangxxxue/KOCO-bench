# AlphaDrive System Architecture and Module Design

## 1. Code Structure and Main Components

### 1.1 Directory Organization
```
AlphaDrive/
├── src/
│   ├── r1-v/                           # Training framework based on open-r1
│   │   ├── src/open_r1/
│   │   │   ├── grpo.py                 # GRPO training entry point
│   │   │   ├── sft.py                  # Supervised fine-tuning entry point
│   │   │   └── trainer/
│   │   │       └── grpo_trainer.py     # GRPO trainer core implementation
│   │   └── configs/                    # Training configuration files
│   ├── distill_r1/                     # R1 reasoning data generation tools
│   │   ├── query_r1.py                 # Query R1 to generate reasoning trajectories
│   │   ├── filter_r1.py                # Filter valid reasoning trajectories
│   │   ├── prompt.py                   # R1 system prompts
│   │   └── create_hf_dataset.py        # Create HuggingFace dataset
│   ├── eval/                           # Evaluation tools
│   └── scripts/                        # Training scripts
├── data_tools/
│   ├── prompt_hub.py                   # Prompt template library
│   └── example.json                    # Data examples
├── eval_tools/
│   ├── qwen2vl_plan_cmd_eval_grpo.py   # GRPO model evaluation script
│   └── qwen2vl_plan_cmd_eval_sft.py    # SFT model evaluation script
└── train_tools/
    ├── run_train_grpo.sh               # GRPO training launch script
    └── run_train_sft.sh                # SFT training launch script
```

### 1.2 Core Modules
- Training Module
  - `Qwen2VLGRPOTrainer`: Extends TRL's GRPOTrainer, supports multimodal input for vision-language models
  - Supports distributed training strategies like FSDP, DeepSpeed ZeRO-2/ZeRO-3
  - Integrates vLLM backend to accelerate generation process
  
- Reward Module
  - `plan_format_reward`: Format validation reward function
  - `plan_speed_reward`: Speed planning reward function
  - `plan_path_reward`: Path planning reward function
  
- Data Processing Module
  - Multimodal data loading: Handles combined image + text input
  - Prompt construction: Dynamically generates prompts based on scenario information
  - Response parsing: Extracts content from `<think>` and `<answer>` tags

## 2. Training Pipeline and Configuration Essentials

### 2.1 Training Entry Points and Modes
- SFT Entry Point: `src/r1-v/src/open_r1/sft.py`
  - Supervised fine-tuning on distilled reasoning data
  - Learns to generate structured reasoning processes and decisions
  
- GRPO Entry Point: `src/r1-v/src/open_r1/grpo.py`
  - Optimizes decision quality through reinforcement learning
  - Supports multi-reward function fusion

### 2.2 Multimodal Input Processing
AlphaDrive processes three types of input information:
1. Visual Input: Driving scene images
   - Uses Qwen2-VL's image processor
   - Supports dynamic resolution (controlled by `max_pixels` and `min_pixels`)
   
2. Text Input: Structured prompts
   - Current speed: `{speed}m/s`
   - Navigation instructions: e.g., "turn left", "go straight", etc.
   - Task description: Requires model to output reasoning process and decisions
   
3. Output Format:
   ```
   <think>
   [Reasoning process: analyze scene, consider constraints, derive decisions]
   </think>
   <answer>
   [Speed decision], [Path decision]
   </answer>
   ```

### 2.3 Reward Function Design

#### 2.3.1 Format Reward
- Function: Validates if output conforms to expected format
- Implementation: Uses regex to match `<think>.*?</think>\s*<answer>.*?</answer>` pattern
- Reward value: Returns 1.0 if format compliant, 0.0 otherwise

#### 2.3.2 Speed Planning Reward
- Decision space: {KEEP, ACCELERATE, DECELERATE, STOP}
- Complexity weights:
  - ACCELERATE: 0.9
  - DECELERATE: 1.0
  - STOP: 1.0
  - KEEP: 0.8
- Computation formula:
  ```
  R_speed = F1_score × complexity_factor + diversity_factor
  
  Where:
  F1_score = 2 × (precision × recall) / (precision + recall)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  complexity_factor = Σ(weight_i) / |predicted_actions|
  diversity_factor = ±0.4 (based on whether first occurrence in batch)
  ```

#### 2.3.3 Path Planning Reward
- Decision space: {STRAIGHT, LEFT_TURN, RIGHT_TURN, LEFT_CHANGE, RIGHT_CHANGE}
- Complexity weights:
  - LEFT_TURN: 1.0
  - RIGHT_TURN: 1.0
  - LEFT_CHANGE: 1.0
  - RIGHT_CHANGE: 1.0
  - STRAIGHT: 0.8
- Computation formula: Same structure as speed reward

### 2.4 GRPO Training Flow

#### 2.4.1 Generation Phase
1. Construct multimodal input:
   - Use `AutoProcessor` to process images and text
   - Set `padding_side="left"` to support batch generation
   
2. Generate multiple responses:
   - Generate G responses per prompt (default G=2)
   - Use sampling strategy (temperature=1.0) to increase diversity

#### 2.4.2 Reward Computation Phase
1. Parse generated responses, extract decision content
2. Compute three reward functions in parallel
3. Sum to obtain total reward: `R_total = R_format + R_speed + R_path`

#### 2.4.3 Advantage Estimation Phase
1. Compute statistics grouped by prompt:
   ```
   mean_R = mean(R_total[group])
   std_R = std(R_total[group])
   ```
   
2. Normalize advantages:
   ```
   advantages = (R_total - mean_R) / (std_R + 1e-4)
   ```

#### 2.4.4 Policy Update Phase
1. Compute log probabilities:
   - Policy model: `logp_θ = log P_θ(response | prompt)`
   - Reference model: `logp_ref = log P_ref(response | prompt)`
   
2. Compute KL divergence:
   ```
   KL = exp(logp_ref - logp_θ) - (logp_ref - logp_θ) - 1
   ```
   
3. Compute loss:
   ```
   loss = -mean(exp(logp_θ - logp_θ.detach()) × advantages - β × KL)
   ```
   
4. Gradient update: Update policy model parameters using Adam optimizer

### 2.5 Key Configuration Parameters
- Model configuration:
  - `model_name_or_path`: Base VLM model (e.g., Qwen2-VL-2B-Instruct)
  - `attn_implementation`: flash_attention_2 (accelerates attention computation)
  - `max_pixels`: 401408 (maximum image pixels)
  
- Training configuration:
  - `per_device_train_batch_size`: 1 (batch size per device)
  - `gradient_accumulation_steps`: 2 (gradient accumulation steps)
  - `num_generations`: 2 (number of responses generated per prompt)
  - `max_prompt_length`: 1024 (maximum prompt length)
  - `max_completion_length`: automatic (maximum generation length)
  
- Reward configuration:
  - `reward_funcs`: ["plan_speed_reward", "plan_path_reward", "plan_format_reward"]
  - `beta`: KL penalty coefficient (controls deviation from reference model)

## 3. Evaluation System

### 3.1 Evaluation Metrics
1. Overall planning accuracy:
   ```
   Accuracy = (samples with correct speed AND path) / total samples
   ```

2. Classification F1 scores:
   - Compute F1 score for each decision category (9 categories)
   - Comprehensively evaluate precision and recall

3. Fine-grained accuracy:
   - Statistics for each speed-path combination accuracy
   - Total of 4×5=20 combinations

### 3.2 Evaluation Flow
1. Load trained model and evaluation dataset
2. For each sample:
   - Construct multimodal input
   - Generate decision (sample 2 responses, take first)
   - Parse `<answer>` tag content
   - Compare with ground truth
3. Compute metrics and save results

## 4. Data Generation Flow

### 4.1 Reasoning Trajectory Generation
1. Construct scene description: Combine image, speed, navigation information
2. Query DeepSeek-R1:
   - Use R1 system prompts
   - Generate responses containing `<think>` reasoning process
3. Parse and store generated trajectories

### 4.2 Data Filtering
1. Validate answer correctness:
   - Extract `<answer>` tag content
   - Compare with ground truth
2. Filter samples with format errors or incorrect answers
3. Retain high-quality reasoning trajectories

### 4.3 Dataset Construction
1. Organize filtered data
2. Convert to HuggingFace dataset format
3. Upload to Hub for training use

