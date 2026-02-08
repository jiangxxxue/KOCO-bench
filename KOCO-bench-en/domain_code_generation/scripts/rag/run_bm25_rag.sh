#!/bin/bash
# BM25 RAG Prompt Construction Script
# Supports multiple frameworks and test examples

# ========================================
# Configuration Variables
# ========================================

# Framework name
export FRAMEWORK="${FRAMEWORK:-example-framework}"

# Script directory
RAG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root directory
PROJECT_ROOT="$(cd "${RAG_SCRIPT_DIR}/.." && pwd)"

# Input data directory (source of parsed .jsonl files)
export INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/data}"

# Output directory for RAG results
RAG_OUTPUT_DIR="${RAG_OUTPUT_DIR:-${PROJECT_ROOT}/rag-output}"

# Knowledge base directories
# Can specify multiple knowledge bases with -k flag
KB_INSTANCE="${KB_INSTANCE:-${PROJECT_ROOT}/knowledge_base/instance}"
KB_FRAMEWORK="${KB_FRAMEWORK:-${PROJECT_ROOT}/knowledge_base/framework}"

# Metadata file path
METADATA_FILE="${METADATA_FILE:-${PROJECT_ROOT}/metadata.json}"

# Top-K retrieval number
TOP_K="${TOP_K:-5}"

# ========================================
# Function: Process single test example
# ========================================
process_single_example() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "BM25 RAG Prompt Construction"
    echo "========================================================"
    echo "Framework: ${framework}"
    echo "Test Example: ${test_example}"
    echo "========================================================"
    
    # Path construction
    local input_data_file="${INPUT_DIR}/${framework}/algorithm_methods_data_${test_example}.jsonl"
    
    # Output data file
    local rag_framework_dir="${RAG_OUTPUT_DIR}/${framework}"
    local output_data_file="${rag_framework_dir}/algorithm_methods_data_${test_example}.jsonl"
    
    # Check if input data file exists
    if [ ! -f "$input_data_file" ]; then
        echo "⚠️  Skipping: Input file does not exist: $input_data_file"
        return 1
    fi
    
    # Ensure output directory exists
    mkdir -p "${rag_framework_dir}"
    
    # Build --knowledge-base arguments
    local kb_args="-k ${KB_INSTANCE}"
    if [ -n "$KB_FRAMEWORK" ] && [ -d "$KB_FRAMEWORK" ]; then
        kb_args="$kb_args -k ${KB_FRAMEWORK}"
    fi
    
    # Check if metadata file exists
    if [ ! -f "$METADATA_FILE" ]; then
        echo "⚠️  Warning: Metadata file does not exist: $METADATA_FILE"
        echo "Continuing without metadata..."
        METADATA_FILE=""
    fi
    
    # Run BM25 RAG script
    cd "${RAG_SCRIPT_DIR}"
    python3 bm25_rag.py \
        --input "$input_data_file" \
        ${METADATA_FILE:+--metadata "$METADATA_FILE"} \
        $kb_args \
        --framework "$framework" \
        --top-k "${TOP_K}" \
        --output "$output_data_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ BM25 RAG prompt construction completed!"
        return 0
    else
        echo "❌ Construction failed"
        return 2
    fi
}

# ========================================
# Main Logic
# ========================================

if [ -n "$TEST_EXAMPLE" ]; then
    # Process single test example
    process_single_example "$FRAMEWORK" "$TEST_EXAMPLE"
    exit $?
else
    # Find all data files to process
    INPUT_FRAMEWORK_DIR="${INPUT_DIR}/${FRAMEWORK}"
    if [ ! -d "$INPUT_FRAMEWORK_DIR" ]; then
        echo "❌ Error: Data directory does not exist: $INPUT_FRAMEWORK_DIR"
        exit 1
    fi
    
    data_files=($(ls "${INPUT_FRAMEWORK_DIR}"/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v "\.output$" | grep -v "\.result$"))
    
    if [ ${#data_files[@]} -eq 0 ]; then
        echo "❌ Error: No data files found in $INPUT_FRAMEWORK_DIR"
        exit 1
    fi
    
    echo "Found ${#data_files[@]} data files to process"
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    for data_file in "${data_files[@]}"; do
        filename=$(basename "$data_file")
        example=$(echo "$filename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
        process_single_example "$FRAMEWORK" "$example"
        res=$?
        if [ $res -eq 0 ]; then ((SUCCESS_COUNT++)); elif [ $res -eq 2 ]; then ((FAIL_COUNT++)); fi
    done
    
    echo ""
    echo "Summary: Success $SUCCESS_COUNT, Failed $FAIL_COUNT"
    echo "Output results located at: ${RAG_OUTPUT_DIR}/${FRAMEWORK}"
    [ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
fi


