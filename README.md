# KOCO-BENCH: Benchmarking Domain Specialization for Large Language Models in Software Development

<div align="center">
</div>

[![Arxiv](https://img.shields.io/badge/Arxiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2601.13240)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/xueniki/KOCO-bench)


## 📋 Overview

**KOCO-bench** is a novel benchmark designed to evaluate domain specialization methods for Large Language Models (LLMs) in real-world software development scenarios. Unlike existing benchmarks that focus on assessing *what* knowledge LLMs possess, KOCO-bench evaluates *how* LLMs acquire and apply new domain knowledge.

### Key Highlights

- 🌐 **6 Emerging Domains**: Covering diverse areas of modern software development
- 🔧 **11 Software Frameworks**: Real-world frameworks with active development
- 📦 **25 Projects**: Comprehensive projects with production-level code
- 📚 **Curated Knowledge Corpora**: Explicit domain knowledge sources for specialization methods
- 🎯 **Multi-Granularity Evaluation**: From function-level to project-level code generation
- ✅ **Rigorous Test Suites**: Automated evaluation with comprehensive test coverage
- 💡 **Knowledge Understanding**: Multiple-choice Q&A for domain comprehension

## 🔍 Problem Statement

Large language models excel at general programming tasks but struggle with domain-specific software development. KOCO-bench addresses this gap by providing:

1. **Explicit Knowledge Corpora**: Structured domain knowledge (APIs, rules, constraints, etc.) for developing and testing specialization methods
2. **Realistic Evaluation**: Tasks require acquiring and applying domain knowledge, mimicking real-world development workflows
3. **Challenging Scenarios**: Current state-of-the-art LLMs show limited performance, highlighting the need for better domain specialization methods

## 🎯 Evaluation Tasks

### Task 1: Domain Code Generation

Multi-granularity code generation tasks with automated evaluation:

- **Function-Level**: Generate individual functions based on specifications
- **Module-Level**: Implement multiple related functions
- **Project-Level**: Complete end-to-end project implementations

Each task includes:
- 📝 Detailed requirements and specifications from algorithm documentation
- 🧪 Comprehensive test suites for automatic evaluation
- 📊 Multiple evaluation metrics (pass@k, avg_pass_ratio, etc.)

### Task 2: Domain Knowledge Understanding

Multiple-choice Q&A tasks to assess domain comprehension across **6 frameworks**:

| Dataset | Description |
|---------|-------------|
| **ascend-transformer-boost** | Ascend NPU transformer optimization |
| **cosmos-rl** |  Cosmos RL framework concepts |
| **robocasa** | Robot manipulation and simulation |
| **trackerLab** |   Visual object tracking |
| **triton-ascend** |  Triton compiler for Ascend |
| **VSLAM-LAB** | Visual SLAM algorithms |

Each question assesses:
- ✅ API usage comprehension
- ✅ Framework design patterns
- ✅ Domain-specific constraints
- ✅ Best practices and conventions
- ✅ Architectural decisions

## 🔧 Build Environment

We provide pre-built Docker images for all frameworks. Simply pull and tag them:

```bash
# ~80.6 GB
docker pull drunkpiano2005/koco-verl-openr1:1.0
docker tag drunkpiano2005/koco-verl-openr1:1.0 kocobench/verl-openr1:v0.4

# ~42.4 GB
docker pull drunkpiano2005/koco-tensorrt:1.0
docker tag drunkpiano2005/koco-tensorrt:1.0 tensorrt:latest

# ~5 GB
docker pull drunkpiano2005/koco-raganything-smolagents:1.0
docker tag drunkpiano2005/koco-raganything-smolagents:1.0 raganything-smolagents:test
```

Then configure your API key:

```bash
cd KOCO-bench-en/domain_code_generation/scripts
cp .env.example .env
# Edit .env and fill in your OPENROUTER_API_KEY (required)
# Optionally add OPENAI_API_KEY for the knowledge_search embedding tool
```

### OpenHands Agent Environment

The OpenHands agent evaluation requires its own Python environment (Python ≥ 3.12). Set it up with [uv](https://docs.astral.sh/uv/):

```bash
cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands
uv sync              # creates .venv and installs all dependencies
uv run pytest tools/knowledge_search/test_knowledge_search.py -v   # verify (optional)
```

> **Without uv**: `python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"` also works.


## 🚀 Quick Start

### Task 1: Domain Code Generation Evaluation

#### Option 1: Single LLM Evaluation

Evaluate a single LLM's code generation ability via the CLI pipeline (prompt construction → API call → Docker execution → metrics):

```bash
cd KOCO-bench-en/domain_code_generation/scripts

# Full pipeline (generate + evaluate + aggregate)
python cli.py run --framework verl --model deepseek/deepseek-v3.2

# Or run steps separately:
python cli.py generate  --framework verl --model deepseek/deepseek-v3.2   # steps 1-3
python cli.py score     --framework verl --model deepseek/deepseek-v3.2   # steps 4-5
```

#### Option 2: OpenHands Agent Evaluation

Evaluate an OpenHands agent that explores repositories and implements functions autonomously:

```bash
cd KOCO-bench-en/domain_code_generation/scripts/koco_openhands
uv sync   # first time only — installs openhands-sdk, regex, etc.

# Full pipeline (agent infer + evaluate)
uv run python cli.py run --framework verl --model deepseek/deepseek-v3.2

# Or run steps separately:
uv run python cli.py infer --framework verl --model deepseek/deepseek-v3.2   # agent inference
uv run python cli.py eval  --framework verl --model deepseek/deepseek-v3.2   # evaluation
```

#### Option 3: Training Custom Models

```bash
cd KOCO-bench-en/domain_code_generation/scripts

# SFT Training
bash sft/run_finetuning.sh

# LoRA Training (parameter-efficient)
bash lora/run_finetuning_lora.sh

# Inference with trained model
bash inference/start_inference_server.sh
bash inference/run_batch_code_generation_with_server.sh
```

### Task 2: Domain Knowledge Understanding Evaluation

#### Option 1: One-Click Evaluation (All Datasets)

```bash
cd KOCO-bench-en/domain_knowledge_understanding/scripts

# Set API key
export OPENROUTER_API_KEY='sk-or-v1-xxx'

# Evaluate all datasets with default model
bash run_evaluation_openrouter.sh

# Or specify a different model
MODEL="qwen/qwen-2.5-coder-32b-instruct" bash run_evaluation_openrouter.sh
```

#### Option 2: Evaluate Single Dataset

```bash
cd KOCO-bench-en/domain_knowledge_understanding/scripts

# Evaluate only one dataset
DATASET="cosmos-rl" bash run_evaluation_openrouter.sh

# With custom model
MODEL="anthropic/claude-sonnet-4.5" DATASET="robocasa" bash run_evaluation_openrouter.sh
```

#### Option 3: Using Python Script Directly

```bash
cd KOCO-bench-en/domain_knowledge_understanding/scripts

python3 evaluation_openrouter.py \
    --model "qwen/qwen2.5-coder-7b-instruct" \
    --input ../problems/problems_cosmos-rl_EN.json \
    --output ../results/qwen2.5-coder-7b-instruct/results_cosmos-rl.json \
    --temperature 0.0 \
    --max_tokens 4096
```

#### Option 4: Local Model Evaluation

```bash
cd KOCO-bench-en/domain_knowledge_understanding/scripts

# Start inference server
bash start_inference_server.sh

# Run evaluation
bash run_evaluation_local.sh

# Stop server
bash stop_inference_server.sh
```


## 📖 Documentation

### Code Generation Documentation
- **[Code Generation Scripts README](KOCO-bench-en/domain_code_generation/scripts/README.md)**: Comprehensive guide for code generation evaluation workflow
- **[Quick Start: Aggregating Metrics](KOCO-bench-en/domain_code_generation/scripts/QUICK_START_AGGREGATE.md)**: Guide for metrics aggregation and comparison
- **[LoRA Training Guide](KOCO-bench-en/domain_code_generation/scripts/lora/README.md)**: LoRA fine-tuning documentation
- **[Inference Server Guide](KOCO-bench-en/domain_code_generation/scripts/inference/INFERENCE_SERVER_README.md)**: Local model serving
- **[Agent Guide](KOCO-bench-en/domain_code_generation/scripts/agent/README.md)**: Agent documentation

### Knowledge Understanding Documentation
- **[Knowledge Understanding Scripts README](KOCO-bench-en/domain_knowledge_understanding/scripts/README.md)**: Guide for running MCQ evaluation
- **[Local Inference Guide](KOCO-bench-en/domain_knowledge_understanding/scripts/LOCAL_INFERENCE_GUIDE.md)**: Using local models for knowledge understanding



## 🙏 Acknowledgments

We thank the open-source communities of all frameworks included in KOCO-bench for their excellent work and contributions to the field.

