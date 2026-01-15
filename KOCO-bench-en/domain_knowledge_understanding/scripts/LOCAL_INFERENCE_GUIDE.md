# KOCO-BENCH Knowledge Understanding - Local Inference Guide

This guide explains how to use local LLM models for the Knowledge Understanding evaluation task.

## Overview

The Knowledge Understanding part of KOCO-BENCH evaluates LLMs' ability to answer multiple-choice questions after reading project code. This directory contains scripts for running evaluations with local LLM models via an inference server.

## Architecture

The local inference system consists of:

1. **inference_server.py** - HTTP API server that loads the model once and serves inference requests
2. **evaluation_local.py** - Evaluation client that sends questions to the server
3. **start_inference_server.sh** - Script to start the inference server in background
4. **stop_inference_server.sh** - Script to stop the inference server
5. **run_evaluation_local.sh** - Batch evaluation script for all problem sets

## Quick Start

### Step 1: Prepare Your Environment

```bash
# Activate your conda environment with PyTorch
conda activate your-env

# Install required packages
pip install torch transformers fastapi uvicorn requests
```

### Step 2: Start the Inference Server

```bash
cd /KOCO-bench/KOCO-bench-en/domain_knowledge_understanding/scripts

# Start with default settings (model at ../../models/qwen2.5-coder-7b-instruct)
bash start_inference_server.sh

# Or specify custom model path
MODEL_PATH=/path/to/your/model bash start_inference_server.sh

# Or specify custom port (if 8000 is occupied)
SERVER_PORT=8001 bash start_inference_server.sh
```

The server will:
- Load the model (this takes a few minutes)
- Start listening on http://localhost:8000
- Print status updates
- Run in the background

**Check server status:**
```bash
curl http://localhost:8000/health
```

### Step 3: Run Evaluation

#### Option A: Evaluate All Datasets (Recommended)

```bash
bash run_evaluation_local.sh
```

This will:
- Evaluate all problem sets in `../problems/`
- Save results to `../results/{model_name}/`
- Generate a summary with overall accuracy

#### Option B: Evaluate Single Dataset

```bash
# Evaluate specific dataset
DATASET=cosmos-rl bash run_evaluation_local.sh
```

Available datasets:
- `ascend-transformer-boost`
- `cosmos-rl`
- `robocasa`
- `trackerLab`
- `triton-ascend`
- `VSLAM-LAB`

#### Option C: Manual Evaluation

```bash
python3 evaluation_local.py \
    --server_url http://localhost:8000 \
    --input ../problems/problems_cosmos-rl_EN.json \
    --output ../results/my-model/results_cosmos-rl.json \
    --temperature 0.0 \
    --max_tokens 4096
```

### Step 4: Stop the Server

```bash
bash stop_inference_server.sh
```

## Configuration

### Environment Variables

#### For `start_inference_server.sh`:
- `MODEL_PATH` - Path to your local model (default: `../../models/qwen2.5-coder-7b-instruct`)
- `SERVER_PORT` - Server port (default: `8000`)
- `SERVER_HOST` - Server host (default: `0.0.0.0`)
- `MAX_CONTEXT_LEN` - Max context length (default: `8192`)
- `CUDA_VISIBLE_DEVICES` - GPU device (default: `0`)
- `LOG_FILE` - Log file path (default: `../logs/inference_server.log`)
- `PID_FILE` - PID file path (default: `../logs/inference_server.pid`)

#### For `run_evaluation_local.sh`:
- `SERVER_URL` - Inference server URL (default: `http://localhost:8000`)
- `DATASET` - Specific dataset to evaluate (default: empty = all datasets)
- `PROJECT_DIR` - Project root directory (modify in script)

### Evaluation Parameters

You can modify these in `run_evaluation_local.sh`:
- `TEMPERATURE` - Sampling temperature (default: `0.0` for deterministic)
- `MAX_TOKENS` - Maximum tokens to generate (default: `4096`)
- `TOP_P` - Top-p sampling (default: `1.0`)
- `DELAY` - Delay between requests in seconds (default: `0.5`)

## File Structure

```
domain_knowledge_understanding/
├── scripts/
│   ├── inference_server.py           # Inference server (NEW)
│   ├── evaluation_local.py           # Local evaluation client (NEW)
│   ├── start_inference_server.sh     # Start server script (NEW)
│   ├── stop_inference_server.sh      # Stop server script (NEW)
│   ├── run_evaluation_local.sh       # Batch evaluation script (NEW)
│   ├── evaluation_openrouter.py      # OpenRouter API evaluation (existing)
│   └── run_evaluation_openrouter.sh  # OpenRouter batch script (existing)
├── problems/
│   ├── problems_cosmos-rl_EN.json
│   ├── problems_robocasa_EN.json
│   └── ...
├── results/
│   ├── {model_name}/
│   │   ├── results_cosmos-rl.json
│   │   ├── results_robocasa.json
│   │   └── summary.json
│   └── ...
└── logs/
    ├── inference_server.log
    └── inference_server.pid
```

## Output Format

### Individual Results (`results_{dataset}.json`)

```json
{
  "summary": {
    "model": "qwen2.5-coder-7b-instruct",
    "server_url": "http://localhost:8000",
    "total": 50,
    "correct": 35,
    "incorrect": 15,
    "accuracy_percent": 70.0
  },
  "results": [
    {
      "id": 1,
      "dataset": "cosmos-rl",
      "instruction": "choose ONE option",
      "stem": "Question text with options...",
      "explanation": "Optional explanation",
      "pred_raw": "Model's raw response...",
      "pred_letters": ["B"],
      "gt_letters": ["B"],
      "is_correct": true
    },
    ...
  ]
}
```

### Summary Statistics (`summary.json`)

```json
{
  "model": "qwen2.5-coder-7b-instruct",
  "server_url": "http://localhost:8000",
  "num_datasets": 6,
  "total_problems": 300,
  "total_correct": 210,
  "total_incorrect": 90,
  "overall_accuracy_percent": 70.0
}
```

## Troubleshooting

### Server won't start

1. **Check if port is occupied:**
   ```bash
   lsof -i :8000
   # If occupied, use a different port:
   SERVER_PORT=8001 bash start_inference_server.sh
   ```

2. **Check CUDA availability:**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Check logs:**
   ```bash
   tail -f ../logs/inference_server.log
   ```

### Out of Memory (OOM) errors

1. **Use smaller batch size** (already set to 1 for knowledge understanding)

2. **Reduce context length:**
   ```bash
   MAX_CONTEXT_LEN=4096 bash start_inference_server.sh
   ```

3. **Use smaller model** or enable quantization

### Connection errors

1. **Verify server is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check firewall/proxy settings:**
   ```bash
   export NO_PROXY="localhost,127.0.0.1"
   ```

### Slow generation

- The knowledge understanding task requires reading large code repositories, so generation may be slower than code generation tasks
- Adjust `DELAY` parameter in `run_evaluation_local.sh` if needed
- Consider using a more powerful GPU

## Comparison with OpenRouter API

| Feature | Local Inference | OpenRouter API |
|---------|----------------|----------------|
| Setup | Complex (model download, GPU required) | Simple (just API key) |
| Cost | Free (after initial setup) | Pay per token |
| Speed | Depends on hardware | Generally fast |
| Privacy | Fully private | Data sent to third party |
| Models | Your choice | Limited selection |
| Customization | Full control | Limited |

**Scripts:**
- Local: `run_evaluation_local.sh` (this guide)
- API: `run_evaluation_openrouter.sh` (existing)

## Advanced Usage

### Running on Multiple GPUs

```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 bash start_inference_server.sh
```

### Custom Model Configuration

Edit `inference_server.py` to modify:
- Model loading parameters (dtype, device_map, etc.)
- Generation parameters (temperature, top_p, etc.)
- Context window size

### Parallel Evaluation

You can run multiple evaluation clients against the same server:

```bash
# Terminal 1
DATASET=cosmos-rl bash run_evaluation_local.sh

# Terminal 2 (with different dataset)
DATASET=robocasa bash run_evaluation_local.sh
```

## Performance Tips

1. **Use flash-attention** if available:
   ```bash
   pip install flash-attn
   ```

2. **Optimize model loading** with quantization:
   - Modify `inference_server.py` to use `load_in_8bit=True` or `load_in_4bit=True`

3. **Increase context length** if model supports it:
   ```bash
   MAX_CONTEXT_LEN=16384 bash start_inference_server.sh
   ```

## Support

For issues or questions:
1. Check logs: `tail -f ../logs/inference_server.log`
2. Review existing evaluation results in `../results/`
3. Compare with OpenRouter API results for validation

## References

- Code Generation inference scripts (similar architecture)
- OpenRouter evaluation scripts (compatible output format)
- HuggingFace Transformers documentation

