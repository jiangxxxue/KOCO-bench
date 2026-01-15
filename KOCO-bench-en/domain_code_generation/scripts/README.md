# Code Generation and Testing Workflow Documentation

## Overview

This project implements a complete code generation and testing evaluation workflow, including:
1. Parsing algorithm method documentation
2. Constructing LLM prompts
3. Generating code using models
4. Replacing code and executing tests
5. Evaluating results and calculating metrics

## Directory Structure

```
scripts/
├── README.md                           # This document
├── parse_algorithm_methods.py          # Algorithm method parser
├── prompts_construction.py             # Prompt constructor
├── execution_evaluation_pure.py        # Pure mode execution evaluator
├── run_parse_algorithm_methods.sh      # Script for parsing algorithm methods
├── run_prompts_construction.sh         # Script for constructing prompts
├── run_execution_evaluation_pure.sh    # Script for execution evaluation
├── run_batch_execution_evaluation_pure.sh  # Script for batch execution evaluation
├── download_model.py                   # Model download utility
├── run_download_model.sh               # Model download script
│
├── sft/                                # SFT (Supervised Fine-Tuning) related
│   ├── build_training_dataset.sh       # Build training dataset
│   ├── finetune_dataset_builder.py     # Dataset builder
│   ├── finetuning.py                   # Model fine-tuning script
│   └── run_finetuning.sh               # Script for running fine-tuning
│
├── lora/                               # LoRA training and inference related
│   ├── README.md                       # LoRA documentation
│   ├── LORA_SERVER_README.md           # LoRA server usage guide
│   ├── finetuning_lora.py              # LoRA fine-tuning script
│   ├── run_finetuning_lora.sh          # Run LoRA fine-tuning
│   ├── merge_lora.py                   # Merge LoRA weights
│   ├── inference_server_lora.py        # LoRA inference server
│   ├── inference_client_lora.py        # LoRA inference client
│   ├── start_inference_server_lora.sh  # Start LoRA server
│   ├── stop_inference_server_lora.sh   # Stop LoRA server
│   ├── run_batch_code_generation_with_lora_server.sh   # Batch code generation
│   └── run_single_code_generation_with_lora_server.sh  # Single code generation
│
├── inference/                          # Inference server related
│   ├── INFERENCE_SERVER_README.md      # Inference server usage guide
│   ├── inference_server.py             # Inference server
│   ├── inference_client.py             # Inference client
│   ├── start_inference_server.sh       # Start inference server
│   ├── stop_inference_server.sh        # Stop inference server
│   └── run_batch_code_generation_with_server.sh  # Batch code generation
│
├── apicall/                            # OpenRouter API call related
│   ├── generate_completions_openrouter.py  # OpenRouter code generator
│   └── run_openrouter.sh                   # Run OpenRouter API generation
│
├── LLM_eval_openrouter.sh             # One-click script for complete evaluation workflow
├── run_knowledge_conflict_test.sh     # Knowledge conflict testing script
├── QUICK_START_AGGREGATE.md           # Quick guide for aggregating metrics
├── aggregate_metrics.py               # Aggregate metrics for single model
├── batch_aggregate_metrics.py         # Batch aggregate metrics for multiple models
└── aggregate_cross_framework.py       # Aggregate metrics across frameworks
```

## Core Script Functions

### 1. Data Preparation Phase

#### `parse_algorithm_methods.py` - Algorithm Method Parser
- **Function**: Extract function information from Markdown documents
- **Input**: Markdown files containing function descriptions
- **Output**: Structured JSONL files containing function names, summaries, parameters, descriptions, etc.
- **Usage**:
  ```bash
  bash scripts/run_parse_algorithm_methods.sh
  ```

#### `prompts_construction.py` - Prompt Constructor
- **Function**: Create conversational format prompts suitable for LLMs for each function
- **Input**: Parsed function information JSONL files
- **Output**: JSONL files with added prompt fields
- **Usage**:
  ```bash
  bash scripts/run_prompts_construction.sh
  ```

### 2. Model Training Phase

#### SFT Training
- **Location**: `scripts/sft/`
- **Function**: Supervised Fine-Tuning
- **Usage**:
  ```bash
  # Build training dataset
  bash scripts/sft/build_training_dataset.sh
  
  # Run SFT training
  bash scripts/sft/run_finetuning.sh
  ```

#### LoRA Training
- **Location**: `scripts/lora/`
- **Function**: Parameter-efficient fine-tuning using LoRA
- **Usage**:
  ```bash
  # Run LoRA training
  bash scripts/lora/run_finetuning_lora.sh
  
  # Merge LoRA weights (optional)
  python scripts/lora/merge_lora.py --base_model_path <base_model> \
                                     --lora_path <lora_adapter> \
                                     --output_path <merged_model>
  ```
- **Detailed Documentation**: See `scripts/lora/README.md` and `scripts/lora/LORA_SERVER_README.md`

### 3. Code Generation Phase

#### Method 1: Using Inference Server (for local models)
- **Location**: `scripts/inference/`
- **Function**: Start local inference server for batch code generation
- **Usage**:
  ```bash
  # Start inference server
  bash scripts/inference/start_inference_server.sh
  
  # Batch code generation
  bash scripts/inference/run_batch_code_generation_with_server.sh
  
  # Stop server
  bash scripts/inference/stop_inference_server.sh
  ```
- **Detailed Documentation**: See `scripts/inference/INFERENCE_SERVER_README.md`

#### Method 2: Using LoRA Inference Server (for LoRA models)
- **Location**: `scripts/lora/`
- **Function**: Start LoRA inference server for code generation
- **Usage**:
  ```bash
  # Start LoRA server
  bash scripts/lora/start_inference_server_lora.sh
  
  # Batch code generation
  bash scripts/lora/run_batch_code_generation_with_lora_server.sh
  
  # Or single generation
  bash scripts/lora/run_single_code_generation_with_lora_server.sh \
      --framework verl --test_example prime
  
  # Stop server
  bash scripts/lora/stop_inference_server_lora.sh
  ```
- **Detailed Documentation**: See `scripts/lora/LORA_SERVER_README.md`

#### Method 3: Using OpenRouter API (recommended for cloud models)
- **Location**: `scripts/apicall/`
- **Function**: Call OpenRouter API for code generation using various LLMs
- **Supported Models**: Qwen, DeepSeek, Claude, GPT, Gemini, etc.
- **Usage**:
  ```bash
  # Set your OpenRouter API key
  export OPENROUTER_API_KEY='sk-or-v1-xxx'
  
  # Run with default model
  bash scripts/apicall/run_openrouter.sh
  
  # Or specify model and framework
  bash scripts/apicall/run_openrouter.sh \
      --model qwen/qwen2.5-coder-32b-instruct \
      --framework verl \
      --test-example prime
  ```

### 4. Test Evaluation Phase

#### `execution_evaluation_pure.py` - Pure Mode Execution Evaluator
- **Function**: Replace generated code into original files, execute tests and evaluate results
- **Input**: JSONL files containing generated code
- **Output**: JSONL files containing test results and metrics files
- **Features**: Pure text replacement mode, directly modifies file contents
- **Usage**:
  ```bash
  # Single evaluation
  bash scripts/run_execution_evaluation_pure.sh
  
  # Batch evaluation
  bash scripts/run_batch_execution_evaluation_pure.sh
  ```

### 5. Metrics Aggregation Phase

#### Aggregate Metrics for Single Model
- **Function**: Calculate comprehensive metrics for a model across multiple test examples
- **Usage**:
  ```bash
  cd scripts
  
  python aggregate_metrics.py \
    --model_dir data/verl/qwen2.5-coder-32b-instruct \
    --test_examples prime ARES LUFFY PURE \
    --framework verl
  ```

#### Batch Aggregate Metrics for Multiple Models
- **Function**: Compare performance across multiple models
- **Usage**:
  ```bash
  cd scripts
  
  python batch_aggregate_metrics.py \
    --base_dir data/verl \
    --model_names model1 model2 model3 \
    --test_examples prime ARES LUFFY PURE \
    --output_csv comparison_results.csv
  ```

#### Aggregate Metrics Across Frameworks
- **Function**: Compare model performance across different frameworks
- **Usage**:
  ```bash
  cd scripts
  
  python aggregate_cross_framework.py \
    --model_name qwen2.5-coder-32b-instruct \
    --frameworks verl tensorrt_model_optimizer \
    --output_csv cross_framework_comparison.csv
  ```

**Detailed Guide**: See [QUICK_START_AGGREGATE.md](QUICK_START_AGGREGATE.md) for more examples and tips

## One-Click Experiment Scripts

### Complete Evaluation Workflow with OpenRouter
- **Script**: `LLM_eval_openrouter.sh`
- **Function**: Automatically runs the complete pipeline from data preparation to metrics aggregation
- **Usage**:
  ```bash
  # Configure in the script:
  # 1. OPENROUTER_API_KEY
  # 2. DEFAULT_MODEL
  # 3. DEFAULT_FRAMEWORK
  # 4. PROJECT_DIR
  
  # Run complete pipeline
  bash scripts/LLM_eval_openrouter.sh
  ```
- **Pipeline Steps**:
  1. Parse algorithm methods
  2. Construct prompts
  3. Generate code via OpenRouter API
  4. Execute batch evaluation
  5. Aggregate metrics

### Knowledge Conflict Testing
- **Script**: `run_knowledge_conflict_test.sh`
- **Function**: Test knowledge conflict by training on Framework A, then Framework B, and evaluating on Framework A
- **Usage**:
  ```bash
  # Configure in the script:
  # 1. Base model path
  # 2. FIRST_FRAMEWORK (e.g., verl)
  # 3. SECOND_FRAMEWORK (e.g., open-r1)
  # 4. TEST_FRAMEWORK (usually same as first)
  
  # Run knowledge conflict test
  bash scripts/run_knowledge_conflict_test.sh
  ```
- **Experimental Steps**:
  1. Prepare training data for Framework A
  2. First SFT training (Base → Framework A)
  3. Prepare training data for Framework B
  4. Second SFT training (Framework A → Framework B)
  5. Prepare evaluation data for test framework
  6. Start inference server
  7. Batch code generation
  8. Batch evaluation
  9. Aggregate metrics

## Complete Workflow Examples

### Workflow 1: Basic Data Preparation and Prompt Construction

```bash
# Step 1: Parse algorithm methods
bash scripts/run_parse_algorithm_methods.sh

# Step 2: Construct prompts
bash scripts/run_prompts_construction.sh
```

### Workflow 2: Model Training (Optional)

**Option A: SFT Training**
```bash
# Build training dataset
cd scripts/sft
bash build_training_dataset.sh

# Run SFT training
bash run_finetuning.sh
```

**Option B: LoRA Training**
```bash
# Run LoRA training
bash scripts/lora/run_finetuning_lora.sh
```

### Workflow 3: Code Generation and Evaluation

**Option A: Using SFT Model + Inference Server**
```bash
# 1. Start inference server
bash scripts/inference/start_inference_server.sh

# 2. Batch code generation
bash scripts/inference/run_batch_code_generation_with_server.sh

# 3. Stop server
bash scripts/inference/stop_inference_server.sh

# 4. Execute test evaluation
bash scripts/run_batch_execution_evaluation_pure.sh
```

**Option B: Using LoRA Model + LoRA Server**
```bash
# 1. Start LoRA server
bash scripts/lora/start_inference_server_lora.sh

# 2. Batch code generation
bash scripts/lora/run_batch_code_generation_with_lora_server.sh

# 3. Stop server
bash scripts/lora/stop_inference_server_lora.sh

# 4. Execute test evaluation
bash scripts/run_batch_execution_evaluation_pure.sh
```

**Option C: Using OpenRouter API**
```bash
# 1. Set API key and call OpenRouter to generate code
export OPENROUTER_API_KEY='sk-or-v1-xxx'
bash scripts/apicall/run_openrouter.sh --framework verl

# 2. Execute test evaluation
bash scripts/run_batch_execution_evaluation_pure.sh

# 3. Aggregate metrics
cd scripts
python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-7b-instruct \
  --test_examples prime ARES LUFFY PURE \
  --framework verl
```

**Option D: One-Click Complete Pipeline**
```bash
# Edit LLM_eval_openrouter.sh to configure:
# - OPENROUTER_API_KEY
# - DEFAULT_MODEL
# - DEFAULT_FRAMEWORK

# Run everything in one command
bash scripts/LLM_eval_openrouter.sh
```

## Configuration Parameters

### Environment Variable Configuration

Most scripts support configuration through environment variables:

#### Data Paths
```bash
export DATA_DIR="../data"              # Data directory
export MODEL_PATH="../models/your_model"  # Model path
export OUTPUT_DIR="../data/outputs"    # Output directory
```

#### Model Parameters
```bash
export NUM_COMPLETIONS=4               # Number of candidates generated per function
export TEMPERATURE=0.7                 # Sampling temperature
export TOP_P=0.9                       # top-p sampling parameter
export MAX_TOKENS=2048                 # Maximum number of generated tokens
```

#### Server Parameters
```bash
export SERVER_HOST="localhost"         # Server host
export SERVER_PORT=8000                # Server port
export GPU_ID=0                        # GPU ID to use
```

## Output Files

### Data Preparation Phase Output
- `{framework}_algorithm_methods.jsonl` - Parsed function data
- `{framework}_algorithm_methods_prompts.jsonl` - Data with added prompts

### Code Generation Phase Output
- `{framework}/{test_example}/completions_{model_name}.jsonl` - Generated code completions

### Evaluation Phase Output
- `{framework}/{test_example}/completions_{model_name}_execution_results.jsonl` - Test execution results
- `{framework}/{test_example}/completions_{model_name}_metrics.json` - Evaluation metrics

### Metrics Description
- `total_functions`: Total number of functions
- `total_tests`: Total number of tests
- `total_passed`: Number of tests passed
- `overall_success_rate`: Overall success rate
- `pass@k`: Pass@k metric (k=1,2,3,4)
- `avg_pass_ratio`: Average pass ratio

#### OpenRouter API Parameters
```bash
export OPENROUTER_API_KEY="sk-or-v1-xxx"  # OpenRouter API key
export MODEL_NAME="qwen/qwen2.5-coder-32b-instruct"  # Model to use
```

## Usage Examples

### Example 1: Using OpenRouter API for Code Generation

```bash
# 1. Prepare data
bash scripts/run_parse_algorithm_methods.sh
bash scripts/run_prompts_construction.sh

# 2. Generate code using OpenRouter API
export OPENROUTER_API_KEY='sk-or-v1-xxx'
bash scripts/apicall/run_openrouter.sh \
    --model qwen/qwen2.5-coder-32b-instruct \
    --framework verl

# 3. Evaluate results
bash scripts/run_batch_execution_evaluation_pure.sh

# 4. Aggregate metrics
cd scripts
python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-32b-instruct \
  --test_examples prime ARES LUFFY PURE \
  --framework verl
```

### Example 1b: One-Click Complete Pipeline

```bash
# Simply run the one-click script (after configuration)
bash scripts/LLM_eval_openrouter.sh
```

### Example 2: Training Custom Model and Generating Code

```bash
# 1. Prepare data
bash scripts/run_parse_algorithm_methods.sh
bash scripts/run_prompts_construction.sh

# 2. Train LoRA model
bash scripts/lora/run_finetuning_lora.sh

# 3. Generate code using LoRA model
bash scripts/lora/start_inference_server_lora.sh
bash scripts/lora/run_batch_code_generation_with_lora_server.sh
bash scripts/lora/stop_inference_server_lora.sh

# 4. Evaluate results
bash scripts/run_batch_execution_evaluation_pure.sh
```

### Example 3: For Specific Framework and Test Case

```bash
# Generate code for verl framework's prime test using LoRA server
export FRAMEWORK="verl"
export TEST_EXAMPLE="prime"

bash scripts/lora/start_inference_server_lora.sh
bash scripts/lora/run_single_code_generation_with_lora_server.sh
bash scripts/lora/stop_inference_server_lora.sh

# Evaluate single test
bash scripts/run_execution_evaluation_pure.sh
```

### Example 4: Batch Model Comparison

```bash
# Compare multiple models on the same framework
cd scripts

python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names \
    qwen2.5-coder-7b-instruct \
    qwen2.5-coder-7b-sft \
    qwen2.5-coder-32b-instruct \
    qwen2.5-coder-7b-lora \
  --test_examples prime ARES LUFFY PURE \
  --output_csv verl_model_comparison.csv

# View results
cat verl_model_comparison.csv
```

### Example 5: Cross-Framework Comparison

```bash
# Compare same model across different frameworks
cd scripts

python aggregate_cross_framework.py \
  --model_name qwen2.5-coder-32b-instruct \
  --frameworks verl tensorrt_model_optimizer open-r1 \
  --output_csv cross_framework_results.csv
```

### Example 6: Knowledge Conflict Testing

```bash
# Test how sequential training on different frameworks affects performance
bash scripts/run_knowledge_conflict_test.sh

# This will:
# 1. Train on Framework A (e.g., verl)
# 2. Continue training on Framework B (e.g., open-r1)
# 3. Evaluate on Framework A to measure knowledge conflict
```

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check if model path is correct
   - Ensure sufficient GPU memory
   - Check log files in `scripts/logs/` directory

2. **Server Startup Failed**
   - Check if port is occupied: `lsof -i :8000`
   - Check server logs: `tail -f scripts/logs/inference_server*.log`
   - Ensure GPU is available: `nvidia-smi`

3. **Test Execution Failed**
   - Check if source code path is correct
   - Ensure test files exist and are executable
   - Check evaluation logs for specific errors

4. **Missing Dependencies**
   - Install required Python packages: `pip install -r requirements.txt`
   - Ensure Python version >= 3.8

### Debug Mode

```bash
# Enable verbose output
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEBUG=1

# View logs
tail -f scripts/logs/inference_server.log
tail -f scripts/logs/inference_server_lora.log
```

## Advanced Features

### Customizing Prompt Templates
Modify the `create_user_prompt` function in `prompts_construction.py` to customize prompt formats.

### Adding New Evaluation Metrics
Add new metric calculations in the `calculate_metrics` function in `execution_evaluation_pure.py`.

### Supporting New Frameworks
1. Create subdirectory for new framework in data directory
2. Add corresponding algorithm method documentation
3. Update framework list in `run_parse_algorithm_methods.sh`

## Performance Optimization Recommendations

1. **GPU Memory Optimization**
   - Use smaller batch sizes
   - Enable model quantization
   - Use LoRA instead of full fine-tuning

2. **Inference Speed Optimization**
   - Use VLLM backend (default for inference server)
   - Adjust `max_tokens` and `num_completions` parameters
   - Use multi-GPU parallel inference

3. **Storage Optimization**
   - Regularly clean `scripts/logs/` directory
   - Compress old output files
   - Use incremental data saving

## Important Notes

1. Ensure sufficient GPU memory to run models (at least 16GB recommended)
2. Test execution requires correct Python environment and dependencies
3. Server logs and PID files are saved in `scripts/logs/` directory
4. Recommended to run small-scale data to validate workflow in test environment first
5. Regularly backup important output files and model weights
6. LoRA models require loading both base model and adapter, ensure both paths are correct

## Related Documentation

- [Quick Start: Aggregating Metrics](QUICK_START_AGGREGATE.md) - Guide for metrics aggregation and comparison
- [LoRA Usage Guide](lora/README.md)
- [LoRA Server Documentation](lora/LORA_SERVER_README.md)
- [Inference Server Documentation](inference/INFERENCE_SERVER_README.md)

## Quick Reference

### Most Common Commands

```bash
# 1. One-click complete evaluation (recommended for beginners)
bash scripts/LLM_eval_openrouter.sh

# 2. Aggregate metrics for a single model
cd scripts
python aggregate_metrics.py \
  --model_dir data/verl/model_name \
  --test_examples prime ARES LUFFY PURE \
  --framework verl

# 3. Compare multiple models
cd scripts
python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names model1 model2 model3 \
  --test_examples prime ARES LUFFY PURE \
  --output_csv comparison.csv

# 4. Knowledge conflict test
bash scripts/run_knowledge_conflict_test.sh
```

### Supported Models (OpenRouter)

- Qwen: `qwen/qwen2.5-coder-7b-instruct`, `qwen/qwen2.5-coder-32b-instruct`
- DeepSeek: `deepseek/deepseek-chat`
- Claude: `anthropic/claude-3.5-sonnet`, `anthropic/claude-sonnet-4.5`
- GPT: `openai/gpt-4`, `openai/gpt-4-turbo`
- Gemini: `google/gemini-2.5-pro`
- And more models supported by OpenRouter

## Technical Support

For issues, please check:
1. README documentation in related directories
2. Log files in `scripts/logs/` directory
3. Usage instruction comments at the beginning of each script file
