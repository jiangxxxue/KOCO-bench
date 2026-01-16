#!/bin/bash
# Batch generate prompt files for all frameworks in source/dataset

set -e

SOURCE_DIR="/source"
DATASET_DIR="$SOURCE_DIR/dataset"
DATA_DIR="$SOURCE_DIR/data"
OUTPUT_DIR="$SOURCE_DIR/output"
PROMPT_SCRIPT_DIR="$SOURCE_DIR/prompt"

# Create directories
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

echo "========================================================"
echo "Batch Prompt Generation for Source Dataset"
echo "========================================================"
echo ""
echo "Source directory: $SOURCE_DIR"
echo "Dataset directory: $DATASET_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Counter
SUCCESS_COUNT=0
FAIL_COUNT=0

# Function to process a single test example
process_test_example() {
    local framework=$1
    local test_example=$2

    echo "----------------------------------------"
    echo "Processing: $framework / $test_example"
    echo "----------------------------------------"

    # Paths
    local md_file="$DATASET_DIR/$framework/test_examples/$test_example/requirements/03_algorithm_and_core_methods.md"
    local code_base="$DATASET_DIR/$framework/test_examples/$test_example/code"
    local metadata="$DATASET_DIR/$framework/knowledge_corpus/metadata.json"
    local data_file="$DATA_DIR/$framework/algorithm_methods_data_$test_example.jsonl"
    local output_file="$OUTPUT_DIR/${test_example}_with_prompts_v2.jsonl"

    # Check if markdown file exists
    if [ ! -f "$md_file" ]; then
        echo "⚠️  Skip: Markdown file not found"
        return 1
    fi

    # Create framework data directory
    mkdir -p "$DATA_DIR/$framework"

    # Step 1: Parse markdown
    echo "Step 1: Parsing markdown..."
    cd "$PROMPT_SCRIPT_DIR"
    python3 parse_algorithm_methods.py \
        --input "$md_file" \
        --output "$data_file" \
        --code-base "$code_base" > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "❌ Failed to parse markdown"
        return 2
    fi

    # Step 2: Generate prompts with keywords
    echo "Step 2: Generating prompts with keywords..."
    python3 prompts_construction.py \
        --input "$data_file" \
        --metadata "$metadata" \
        --output "$output_file" \
        --base-dir "$DATASET_DIR" \
        --framework "$framework" \
        --test-example "$test_example" > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate prompts"
        return 2
    fi

    # Verify
    local record_count=$(wc -l < "$output_file")
    echo "✓ Generated: $output_file ($record_count records)"

    return 0
}

# Find all test examples
echo "Scanning for test examples..."
echo ""

# Process verl framework
if [ -d "$DATASET_DIR/verl/test_examples" ]; then
    for test_dir in "$DATASET_DIR/verl/test_examples"/*; do
        if [ -d "$test_dir" ]; then
            test_example=$(basename "$test_dir")
            process_test_example "verl" "$test_example"
            if [ $? -eq 0 ]; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        fi
    done
fi

# Process raganything framework
if [ -d "$DATASET_DIR/raganything/test_examples" ]; then
    for test_dir in "$DATASET_DIR/raganything/test_examples"/*; do
        if [ -d "$test_dir" ]; then
            test_example=$(basename "$test_dir")
            process_test_example "raganything" "$test_example"
            if [ $? -eq 0 ]; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        fi
    done
fi

# Process smolagents framework
if [ -d "$DATASET_DIR/smolagents/test_examples" ]; then
    for test_dir in "$DATASET_DIR/smolagents/test_examples"/*; do
        if [ -d "$test_dir" ]; then
            test_example=$(basename "$test_dir")
            process_test_example "smolagents" "$test_example"
            if [ $? -eq 0 ]; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        fi
    done
fi

# Process tensorrt_model_optimizer framework
if [ -d "$DATASET_DIR/tensorrt_model_optimizer/test_examples" ]; then
    for test_dir in "$DATASET_DIR/tensorrt_model_optimizer/test_examples"/*; do
        if [ -d "$test_dir" ]; then
            test_example=$(basename "$test_dir")
            process_test_example "tensorrt_model_optimizer" "$test_example"
            if [ $? -eq 0 ]; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        fi
    done
fi

# Summary
echo ""
echo "========================================================"
echo "Batch Generation Complete"
echo "========================================================"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""
echo "Generated files location:"
echo "  $OUTPUT_DIR"
echo ""
echo "List generated files:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  No files generated"
echo ""
