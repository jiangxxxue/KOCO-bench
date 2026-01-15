# KOCO-BENCH Knowledge Understanding Evaluation Scripts

This directory contains scripts for evaluating language models on the KOCO-BENCH knowledge understanding tasks using OpenRouter API.

## 📁 Files

- **`evaluation_openrouter.py`**: Python script that evaluates multiple-choice questions using OpenRouter API
- **`run_evaluation_openrouter.sh`**: Bash script for one-command evaluation of all datasets
- **`evaluation_old.py`**: Legacy script for local model evaluation (deprecated)
- **`README.md`**: This file

## 🚀 Quick Start

### 1. Set up API Key

First, get your API key from [OpenRouter](https://openrouter.ai/keys), then set it as an environment variable:

```bash
export OPENROUTER_API_KEY='sk-or-v1-your-api-key-here'
```

Or modify the `OPENROUTER_API_KEY` variable in `run_evaluation_openrouter.sh`.

### 2. Run Evaluation

#### Option A: Evaluate All Datasets (Recommended)

```bash
cd /KOCO-bench/KOCO-bench-en/domain_knowledge_understanding/scripts

# Use default model (qwen/qwen2.5-coder-7b-instruct)
bash run_evaluation_openrouter.sh

# Or specify a different model
MODEL="anthropic/claude-sonnet-4.5" bash run_evaluation_openrouter.sh
```

#### Option B: Evaluate a Single Dataset

```bash
# Evaluate only one dataset (e.g., ascend-transformer-boost)
DATASET="ascend-transformer-boost" bash run_evaluation_openrouter.sh

# With custom model
MODEL="qwen/qwen-2.5-coder-32b-instruct" DATASET="robocasa" bash run_evaluation_openrouter.sh
```

#### Option C: Use Python Script Directly

```bash
python3 evaluation_openrouter.py \
    --model "qwen/qwen2.5-coder-7b-instruct" \
    --input ../problems/problems_ascend-transformer-boost_EN.json \
    --output ../results/qwen2.5-coder-7b-instruct/results_ascend-transformer-boost.json \
    --temperature 0.0 \
    --max_tokens 4096 \
    --delay 0.5
```

## 📊 Output

Results are saved in the `../results/<model_name>/` directory:

```
results/
├── qwen2.5-coder-7b-instruct/
│   ├── results_ascend-transformer-boost.json
│   ├── results_cann-hccl.json
│   ├── results_cosmos-rl.json
│   ├── results_robocasa.json
│   ├── results_trackerLab.json
│   ├── results_triton-ascend.json
│   ├── results_VSLAM-LAB.json
│   └── summary.json  # Overall statistics
└── claude-sonnet-4.5/
    └── ...
```

Each result file contains:
- **summary**: Overall statistics (total, correct, incorrect, accuracy)
- **results**: Detailed results for each problem (predicted answer, ground truth, correctness)

## 🎯 Available Models

The script supports any model available on OpenRouter. Popular choices include:

### Open Source Models
- `meta-llama/llama-3.1-8b-instruct`
- `qwen/qwen2.5-coder-7b-instruct`
- `qwen/qwen-2.5-coder-32b-instruct`
- `deepseek/deepseek-chat-v3.1`

### Closed Source Models
- `anthropic/claude-sonnet-4.5`
- `openai/gpt-5-mini`
- `openai/o4-mini`
- `moonshotai/kimi-k2-0905`

See [OpenRouter Models](https://openrouter.ai/models) for the full list.

## ⚙️ Configuration

You can customize evaluation parameters in `run_evaluation_openrouter.sh`:

```bash
# API Configuration
OPENROUTER_API_KEY='your-api-key'
DEFAULT_MODEL="qwen/qwen2.5-coder-7b-instruct"

# Generation Parameters
TEMPERATURE=0.0        # 0.0 for deterministic output
MAX_TOKENS=4096        # Maximum response length
TOP_P=1.0              # Nucleus sampling parameter
DELAY=0.5              # Delay between API calls (seconds)
```

Or pass them directly to the Python script:

```bash
python3 evaluation_openrouter.py \
    --model "your-model" \
    --input "input.json" \
    --output "output.json" \
    --temperature 0.0 \
    --max_tokens 4096 \
    --top_p 1.0 \
    --delay 0.5
```

## 📝 Problem Format

Input JSON files should follow this format:

```json
{
  "meta": {
    "repository": "repo-name",
    "total_problems": 10,
    ...
  },
  "problems": [
    {
      "id": 1,
      "dataset": "dataset-name",
      "instruction": "choose ONE option",
      "stem": "Question text\n\n- (A) Option A\n- (B) Option B\n...",
      "explanation": "Explanation (optional)",
      "gt": ["A"]
    },
    {
      "id": 2,
      "instruction": "choose MULTIPLE options",
      "stem": "...",
      "gt": ["A", "C", "D"]
    }
  ]
}
```

## 🔍 How It Works

1. **Load Problems**: Parse the JSON file to extract problems, options, and ground truth answers
2. **Prepare Prompts**: Format each problem with instructions for the model
3. **Call API**: Send prompts to OpenRouter API and receive responses
4. **Parse Answers**: Extract answer letters from model responses (looking for `\boxed{A}` format)
5. **Evaluate**: Compare predicted answers with ground truth
6. **Save Results**: Write detailed results and statistics to JSON files

## 🐛 Troubleshooting

### API Key Issues

```
❌ Error: OPENROUTER_API_KEY is not set
```

**Solution**: Set the environment variable:
```bash
export OPENROUTER_API_KEY='sk-or-v1-your-key'
```

### Rate Limiting

If you encounter rate limiting errors, increase the `--delay` parameter:

```bash
python3 evaluation_openrouter.py --delay 1.0 ...
```

### Empty Responses

If the model returns empty responses, try:
- Increasing `--max_tokens`
- Adjusting `--temperature` (try 0.1 or 0.3)
- Using a different model

## 📈 Batch Evaluation

To evaluate multiple models in batch:

```bash
#!/bin/bash

MODELS=(
    "qwen/qwen2.5-coder-7b-instruct"
    "deepseek/deepseek-chat-v3.1"
    "anthropic/claude-sonnet-4.5"
)

for model in "${MODELS[@]}"; do
    echo "Evaluating $model..."
    MODEL="$model" bash run_evaluation_openrouter.sh
done
```

## 📚 Datasets

Current datasets in `../problems/`:

1. **ascend-transformer-boost**: Ascend NPU optimization framework
2. **cann-hccl**: CANN collective communication library
3. **cosmos-rl**: Reinforcement learning framework
4. **robocasa**: Robot manipulation benchmark
5. **trackerLab**: Visual tracking framework
6. **triton-ascend**: Triton compiler for Ascend
7. **VSLAM-LAB**: Visual SLAM laboratory

## 🔗 Related Files

- **Problems**: `../problems/problems_*_EN.json`
- **Repositories**: `../repositories/`
- **Results**: `../results/`




