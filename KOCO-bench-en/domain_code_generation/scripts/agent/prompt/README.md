# Prompt Generation Scripts

Scripts for generating complete prompt files with keyword fields for Claude Code usage.

## Overview

This directory contains scripts to generate `*_with_prompts_v2.jsonl` files from parsed algorithm methods data. The generated files include:
- **prompt** field: System and user prompts with repository information
- **keyword** field: Absolute paths to knowledge corpus and test example code

## Files

- `parse_algorithm_methods.py` - Parse markdown documentation to extract function information
- `prompts_construction.py` - Add prompt and keyword fields to data
- `run_parse_algorithm_methods.sh` - Shell script to run parsing
- `run_prompts_construction.sh` - Shell script to run prompt generation

## Prerequisites

1. **Directory Structure**:
   ```
   domain_knowledge_dataset/
   ├── {framework}/
   │   ├── knowledge_corpus/
   │   │   └── metadata.json
   │   └── test_examples/
   │       └── {test_example}/
   │           └── code/
   └── source/
       └── prompt/
           ├── prompts_construction.py
           └── run_prompts_construction.sh
   ```

2. **Metadata File** (`metadata.json`):
   ```json
   {
     "framework_name": "RAGAnything",
     "framework_description": "RAGAnything is a comprehensive multimodal RAG system..."
   }
   ```

3. **Input Data**: Parsed algorithm methods data file
   - Format: `algorithm_methods_data_{test_example}.jsonl`
   - Location: `source/data/{framework}/`

## Usage

### Method 1: Using Shell Script (Recommended)

#### Process Single Test Example

```bash
# Set environment variables
export BASE_DIR="/absolute/path/to/domain_knowledge_dataset"
export FRAMEWORK="raganything"
export TEST_EXAMPLE="rag4chat"

# Run the script
./run_prompts_construction.sh
```

#### Process All Test Examples for a Framework

```bash
# Set environment variables
export BASE_DIR="/absolute/path/to/domain_knowledge_dataset"
export FRAMEWORK="raganything"

# Run without TEST_EXAMPLE to process all
./run_prompts_construction.sh
```

### Method 2: Using Python Script Directly

```bash
python3 prompts_construction.py \
  --input ./data/raganything/algorithm_methods_data_rag4chat.jsonl \
  --metadata /path/to/raganything/knowledge_corpus/metadata.json \
  --output ./data/raganything/rag4chat_with_prompts_v2.jsonl \
  --base-dir /absolute/path/to/domain_knowledge_dataset \
  --framework raganything \
  --test-example rag4chat
```

## Parameters

### prompts_construction.py

| Parameter | Short | Required | Description |
|-----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Input JSONL file path |
| `--metadata` | `-m` | Yes | Metadata JSON file path |
| `--output` | `-o` | No | Output file path (default: `{input}_with_prompts_v2.jsonl`) |
| `--base-dir` | `-b` | Yes | Base directory for building absolute paths |
| `--framework` | `-f` | Yes | Framework name (e.g., raganything, verl) |
| `--test-example` | `-t` | Yes | Test example name (e.g., rag4chat, ARES) |

### run_prompts_construction.sh

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FRAMEWORK` | raganything | Framework name |
| `TEST_EXAMPLE` | (empty) | Test example name (empty = process all) |
| `BASE_DIR` | (auto-detected) | Absolute path to domain_knowledge_dataset |

## Output Format

The generated `*_with_prompts_v2.jsonl` file contains:

```json
{
  "function_name": "compute_reward",
  "overview": "...",
  "function_signature": "def compute_reward(...):",
  "input_parameters": "...",
  "detailed_description": "...",
  "output": "...",
  "func_implementation": "",
  "test_code_path": "",
  "implementation_location": "...",
  "prompt": [
    {
      "role": "system",
      "content": "You are a Python programmer developing with the RAGAnything framework. ...\n\nThe framework knowledge base is located at: ./raganything/knowledge_corpus. ...\n\nThe development repository for this specific project is located at: ./raganything/test_examples/rag4chat. ..."
    },
    {
      "role": "user",
      "content": "Please implement a function named `compute_reward`.\n\n**Function Overview:**\n..."
    }
  ],
  "keyword": [
    "/absolute/path/to/domain_knowledge_dataset/raganything/knowledge_corpus",
    "/absolute/path/to/domain_knowledge_dataset/raganything/test_examples/rag4chat/code"
  ]
}
```

## Key Features

### 1. Enhanced System Prompt

The system prompt includes:
- Framework name and description
- Relative paths to knowledge corpus and test example code
- Explanation of the dual-repository setup

### 2. Keyword Field with Absolute Paths

The keyword field contains absolute paths used by `incremental_runner.py` for:
- Copying repositories to temporary directories
- Providing Claude Code access to framework knowledge and project code

### 3. Path Validation

The script validates:
- Input file exists
- Metadata file exists
- Base directory exists
- Knowledge corpus directory exists (warning if missing)
- Test example code directory exists (warning if missing)

## Examples

### Example 1: RAGAnything Framework

```bash
export BASE_DIR="/home/user/domain_knowledge_dataset"
export FRAMEWORK="raganything"
export TEST_EXAMPLE="rag4chat"

./run_prompts_construction.sh
```

**Output**: `./data/raganything/rag4chat_with_prompts_v2.jsonl`

**Keyword paths**:
- `/home/user/domain_knowledge_dataset/raganything/knowledge_corpus`
- `/home/user/domain_knowledge_dataset/raganything/test_examples/rag4chat/code`

### Example 2: VERL Framework

```bash
export BASE_DIR="/home/user/domain_knowledge_dataset"
export FRAMEWORK="verl"
export TEST_EXAMPLE="ARES"

./run_prompts_construction.sh
```

**Output**: `./data/verl/ARES_with_prompts_v2.jsonl`

**Keyword paths**:
- `/home/user/domain_knowledge_dataset/verl/knowledge_corpus`
- `/home/user/domain_knowledge_dataset/verl/test_examples/ARES/code`

### Example 3: Process All Test Examples

```bash
export BASE_DIR="/home/user/domain_knowledge_dataset"
export FRAMEWORK="raganything"

# Process all test examples in raganything framework
./run_prompts_construction.sh
```

## Troubleshooting

### Error: "Input file does not exist"

**Cause**: The algorithm methods data file hasn't been generated yet.

**Solution**: Run the parsing script first:
```bash
FRAMEWORK=raganything TEST_EXAMPLE=rag4chat ./run_parse_algorithm_methods.sh
```

### Error: "Base directory does not exist"

**Cause**: The BASE_DIR environment variable is not set or points to an invalid path.

**Solution**: Set the correct absolute path:
```bash
export BASE_DIR="/absolute/path/to/domain_knowledge_dataset"
```

### Warning: "Knowledge corpus directory does not exist"

**Cause**: The framework's knowledge_corpus directory is missing.

**Impact**: The keyword path will be generated but won't be valid for repository copying.

**Solution**: Ensure the directory structure is correct:
```
domain_knowledge_dataset/
└── {framework}/
    └── knowledge_corpus/
```

### Warning: "Test example code directory does not exist"

**Cause**: The test example's code directory is missing.

**Impact**: The keyword path will be generated but won't be valid for repository copying.

**Solution**: Ensure the directory structure is correct:
```
domain_knowledge_dataset/
└── {framework}/
    └── test_examples/
        └── {test_example}/
            └── code/
```

## Integration with Claude Code

The generated `*_with_prompts_v2.jsonl` files are used by:

1. **incremental_runner.py**: Reads the keyword field to copy repositories
2. **repo_level_code_runner.py**: Uses the prompt field to call Claude Code

**Workflow**:
```
1. Generate prompts with keywords
   ↓
2. Run incremental_runner.py
   ↓ (reads keyword field)
3. Copy repositories to temp directory
   ↓ (uses prompt field)
4. Execute Claude Code
   ↓
5. Extract implementation results
```

## Notes

- **Absolute vs Relative Paths**:
  - `keyword` field: Absolute paths (for actual file operations)
  - System prompt: Relative paths (for Claude's reference)

- **Path Consistency**: Ensure BASE_DIR points to the same location used by incremental_runner.py

- **Metadata Fallback**: If metadata.json is missing, the script uses default framework descriptions

## See Also

- [REPO_COPY_MECHANISM.md](../claude_code/REPO_COPY_MECHANISM.md) - How repository copying works
- [FIELD_COMPARISON.md](../claude_code/FIELD_COMPARISON.md) - Field comparison and data flow
- [DATA_FLOW.md](../claude_code/DATA_FLOW.md) - Complete data processing flow
