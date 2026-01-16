# Claude Code Domain Dataset - Prompt Generator

Generate prompt files with keywords for Claude Code domain-specific code generation.

## Quick Start

```bash
cd source
./generate_all.sh
```

This generates all `*_with_prompts_v2.jsonl` files in the `output/` directory.

## What This Does

Generates prompt files for **5 frameworks** and **25 test examples**:

- **open-r1** (5): AlphaDrive, CANOE, VLM-R1, VisualQuality-R1, ml-diffucoder
- **verl** (7): ARES, DAPO, LUFFY, PACS, PURE, critic-rl, prime
- **raganything** (5): BookWorm, Chat-ANYTHING, law-unleashed-rag, obsidianGraphRAG, rag4chat
- **smolagents** (4): DeepSearchAgents, ToolBrain, examples, smolcc
- **tensorrt_model_optimizer** (4): FlagScale, Nemo, TensorRT-Incubator, byteperf

**Output**: 25 files, ~133 functions, ~628KB

## Prerequisites

- Python 3.8+
- Dataset directory with framework data (see structure below)

## Directory Structure

```
source/
├── generate_all.sh             # One-click generation script
├── verify_prompts.py           # Verification script
├── README.md                   # This file
│
├── dataset/                    # Input: Framework datasets
│   ├── open-r1/
│   │   ├── knowledge_corpus/
│   │   │   └── metadata.json
│   │   └── test_examples/
│   │       └── {test_example}/
│   │           ├── code/
│   │           └── requirements/
│   │               └── 03_algorithm_and_core_methods.md
│   ├── verl/
│   ├── raganything/
│   ├── smolagents/
│   └── tensorrt_model_optimizer/
│
├── prompt/                     # Generation scripts
│   ├── parse_algorithm_methods.py
│   ├── prompts_construction.py
│   └── README.md
│
├── output/                     # Generated files (created by script)
│   └── *_with_prompts_v2.jsonl
│
└── claude_code/                # Claude Code integration
    ├── incremental_runner.py
    ├── repo_level_code_runner.py
    └── covert_output.py
```

## Output Format

Each generated file contains JSONL format:

```json
{
  "function_name": "compute_score",
  "overview": "...",
  "function_signature": "def compute_score(...):",
  "prompt": [
    {
      "role": "system",
      "content": "You are a Python programmer developing with the verl framework. ...\n\nThe framework knowledge base is located at: ./verl/knowledge_corpus. ..."
    },
    {
      "role": "user",
      "content": "Please implement a function named `compute_score`..."
    }
  ],
  "keyword": [
    "/absolute/path/to/verl/knowledge_corpus",
    "/absolute/path/to/verl/test_examples/ARES/code"
  ]
}
```

## Usage

### 1. Generate Prompt Files

```bash
./generate_all.sh
```

### 2. Verify Generated Files

```bash
python3 verify_prompts.py
```

### 3. Use with Claude Code

```bash
cd claude_code
python3 incremental_runner.py \
  --data-dir ../output \
  --output-dir ../generated_results
```

## Manual Generation (Optional)

If you want to generate files manually for a specific framework:

```bash
cd prompt

# Step 1: Parse markdown
python3 parse_algorithm_methods.py \
  --input ../dataset/verl/test_examples/ARES/requirements/03_algorithm_and_core_methods.md \
  --output ../data/verl/algorithm_methods_data_ARES.jsonl \
  --code-base ../dataset/verl/test_examples/ARES/code

# Step 2: Generate prompts with keywords
python3 prompts_construction.py \
  --input ../data/verl/algorithm_methods_data_ARES.jsonl \
  --metadata ../dataset/verl/knowledge_corpus/metadata.json \
  --output ../output/ARES_with_prompts_v2.jsonl \
  --base-dir $(pwd)/../dataset \
  --framework verl \
  --test-example ARES
```




