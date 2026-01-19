#!/bin/bash
# Build framework training dataset - supports multiple frameworks/repos
#
# Usage:
#   Build default framework:
#     ./scripts/build_training_dataset.sh
#
#   Build other framework:
#     FRAMEWORK=your_framework ./scripts/build_training_dataset.sh
#
#   Specify specific repo name:
#     FRAMEWORK=your_framework REPO_NAME=custom-repo ./scripts/build_training_dataset.sh
#
#   Set maximum file size (bytes):
#     MAX_FILE_SIZE=2097152 ./scripts/build_training_dataset.sh

set -e

# ========================================
# Configuration Variables
# ========================================

# Framework name
FRAMEWORK="${FRAMEWORK:-your_framework}"

# Repo name (directory name in knowledge corpus)
REPO_NAME="${REPO_NAME:-${FRAMEWORK}-main}"

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# Source directory: knowledge corpus
SOURCE_DIR="${PROJECT_ROOT}/${FRAMEWORK}/knowledge_corpus/${REPO_NAME}"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/scripts/data/${FRAMEWORK}"
OUTPUT_FILE="${OUTPUT_DIR}/${FRAMEWORK}_training_dataset.jsonl"

# Parameters
MAX_FILE_SIZE="${MAX_FILE_SIZE:-1048576}"  # 1MB
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/your/tokenizer}"

# ========================================
# Execute Build
# ========================================

echo "========================================================"
echo "Building Training Dataset"
echo "========================================================"
echo "Framework: ${FRAMEWORK}"
echo "Repo: ${REPO_NAME}"
echo "Source Directory: ${SOURCE_DIR}"
echo "Output File: ${OUTPUT_FILE}"
echo "Max File Size: ${MAX_FILE_SIZE} bytes"
echo "Tokenizer Model: ${TOKENIZER_PATH}"
echo "========================================================"
echo ""

# Check source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory does not exist: $SOURCE_DIR"
    echo ""
    echo "Hint: Please ensure the following path exists:"
    echo "  ${PROJECT_ROOT}/${FRAMEWORK}/knowledge_corpus/${REPO_NAME}"
    echo ""
    echo "Or use environment variables to specify other framework/repo:"
    echo "  FRAMEWORK=your_framework REPO_NAME=your_repo ./scripts/build_training_dataset.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run dataset builder
cd "${SCRIPT_DIR}/sft"
python3 finetune_dataset_builder.py \
    --source-dir "$SOURCE_DIR" \
    --output-file "$OUTPUT_FILE" \
    --format jsonl \
    --max-file-size "$MAX_FILE_SIZE" \
    --tokenizer-path "$TOKENIZER_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "✅ Dataset build completed!"
    echo "========================================================"
    echo "Data file: ${OUTPUT_FILE}"
    echo "Statistics file: ${OUTPUT_FILE%.jsonl}.stats.json"
    echo ""
    
    # Display statistics
    if [ -f "${OUTPUT_FILE%.jsonl}.stats.json" ]; then
        echo "📊 Dataset Statistics:"
        cat "${OUTPUT_FILE%.jsonl}.stats.json" | python3 -c "
import json, sys
stats = json.load(sys.stdin)
print(f\"  Total files: {stats['total_files']}")
print(f\"  Processed: {stats['processed_files']}")
print(f\"  Skipped: {stats['skipped_files']}")
print(f\"  Total characters: {stats['total_size_chars']:,}")
print(f\"  Total lines: {stats['total_lines']:,}")
if 'total_tokens' in stats and stats['total_tokens'] > 0:
    print(f\"  Total tokens: {stats['total_tokens']:,}")
    print(f\"  Average tokens per file: {stats.get('average_tokens_per_file', 0):.1f}")
print(f\"  File type distribution:\")
for ftype, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
    print(f\"    {ftype}: {count}\")
"
    fi
    
    echo ""
    echo "========================================================"
else
    echo ""
    echo "❌ Dataset build failed"
    exit 1
fi

