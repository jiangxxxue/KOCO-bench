#!/bin/bash

# Configure before running:
# 1. OPENROUTER_API_KEY - set via: export OPENROUTER_API_KEY='your-key'
#    Get key: https://openrouter.ai/keys
# 2. DEFAULT_MODEL - OpenRouter model ID (e.g. qwen/qwen-2.5-coder-32b-instruct)
# 3. DEFAULT_FRAMEWORK - framework name under test_examples
# 4. PROJECT_DIR - project root (default: parent of scripts/ parent)

# Default config (override via environment)
DEFAULT_MODEL="${DEFAULT_MODEL:-qwen/qwen-2.5-coder-32b-instruct}"
DEFAULT_FRAMEWORK="${DEFAULT_FRAMEWORK:-your_framework}"
NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"

# Project root: set PROJECT_DIR or use directory containing domain_code_generation
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Parse algorithm methods
# Test example name (empty = process all)
FRAMEWORK="${FRAMEWORK:-$DEFAULT_FRAMEWORK}"
TEST_EXAMPLE="${TEST_EXAMPLE:-}"

SCRIPT_DIR="${PROJECT_DIR}/scripts"

# ========================================
# Process single test example
# ========================================
process_single_example() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "Parse algorithm core methods"
    echo "========================================================"
    echo "Framework: ${framework}"
    echo "Test example: ${test_example}"
    echo "========================================================"
    
    local input_file="${PROJECT_DIR}/${framework}/test_examples/${test_example}/requirements/03_algorithm_and_core_methods.md"
    local code_base="${PROJECT_DIR}/${framework}/test_examples/${test_example}/code"
    local test_base="${PROJECT_DIR}/${framework}/test_examples/${test_example}/code/tests"

    local output_dir="${PROJECT_DIR}/scripts/data/${framework}"
    local output_file="${output_dir}/algorithm_methods_data_${test_example}.jsonl"
    
    echo "Input file: ${input_file}"
    echo "Code base: ${code_base}"
    echo "Output file: ${output_file}"
    echo ""
    
    if [ ! -f "$input_file" ]; then
        echo "⚠️  Skipping: input file does not exist"
        return 1
    fi
    
    mkdir -p "$output_dir"
    
    cd "${SCRIPT_DIR}"
    python3 parse_algorithm_methods.py \
        --input "$input_file" \
        --output "$output_file" \
        --code-base "$code_base" \
        --test-base "$test_base"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Parse completed!"
        echo "Output file: $output_file"
        local num_functions=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        echo "Extracted functions: $num_functions"
        return 0
    else
        echo ""
        echo "❌ Parse failed"
        return 1
    fi
}

# ========================================
# Step 1: Parse algorithm methods
# ========================================
run_parse_algorithm_methods() {
    echo ""
    echo "###########################################################"
    echo "# Step 1: Parse algorithm core methods"
    echo "###########################################################"
    echo ""

    if [ -n "$TEST_EXAMPLE" ]; then
        echo "Mode: single test example"
        echo ""
        process_single_example "$FRAMEWORK" "$TEST_EXAMPLE"
        return $?
    else
        echo "========================================================"
        echo "Mode: process all test examples for framework ${FRAMEWORK}"
        echo "========================================================"
        echo ""
        
        TEST_EXAMPLES_DIR="${PROJECT_DIR}/${FRAMEWORK}/test_examples"
        
        if [ ! -d "$TEST_EXAMPLES_DIR" ]; then
            echo "❌ Error: framework directory does not exist: $TEST_EXAMPLES_DIR"
            return 1
        fi
        
        test_examples=($(ls -d "$TEST_EXAMPLES_DIR"/*/ 2>/dev/null | xargs -n 1 basename))
        
        if [ ${#test_examples[@]} -eq 0 ]; then
            echo "❌ Error: no test examples found"
            return 1
        fi
        
        echo "Found ${#test_examples[@]} test examples: ${test_examples[*]}"
        echo ""
        
        SUCCESS_COUNT=0
        FAIL_COUNT=0
        SKIP_COUNT=0
        
        for example in "${test_examples[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "Processing: ${example}"
            echo "----------------------------------------"
            
            process_single_example "$FRAMEWORK" "$example"
            result=$?
            
            if [ $result -eq 0 ]; then
                ((SUCCESS_COUNT++))
            elif [ $result -eq 1 ]; then
                ((SKIP_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
            
            echo ""
        done
        
        echo "========================================================"
        echo "Batch parse completed"
        echo "========================================================"
        echo "Framework: ${FRAMEWORK}"
        echo "Success: ${SUCCESS_COUNT}"
        echo "Skipped: ${SKIP_COUNT}"
        echo "Failed: ${FAIL_COUNT}"
        echo "========================================================"
        
        [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
    fi
}


############################################################
# Run prompts construction
############################################################
process_single_example_prompts() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "Build prompts"
    echo "========================================================"
    echo "Framework: ${framework}"
    echo "Test example: ${test_example}"
    echo "========================================================"
    
    local metadata_file="${PROJECT_DIR}/${framework}/knowledge_corpus/metadata.json"
    local data_dir="${PROJECT_DIR}/scripts/data/${framework}"
    local data_file="${data_dir}/algorithm_methods_data_${test_example}.jsonl"
    
    echo "Metadata file: ${metadata_file}"
    echo "Data file: ${data_file}"
    echo ""
    
    if [ ! -f "$data_file" ]; then
        echo "⚠️  Skipping: data file does not exist"
        echo "Run first: FRAMEWORK=${framework} TEST_EXAMPLE=${test_example} ./scripts/run_parse_algorithm_methods.sh"
        return 1
    fi
    
    if [ ! -f "$metadata_file" ]; then
        echo "⚠️  Warning: metadata file does not exist, using default framework description"
    fi
    
    cd "${SCRIPT_DIR}"
    python3 prompts_construction.py \
        --input "$data_file" \
        --metadata "$metadata_file" \
        --output "$data_file"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Prompt construction completed!"
        return 0
    else
        echo ""
        echo "❌ Build failed"
        return 1
    fi
}

# ========================================
# Step 2: Build prompts
# ========================================
run_prompts_construction() {
    echo ""
    echo "###########################################################"
    echo "# Step 2: Build prompts"
    echo "###########################################################"
    echo ""

    if [ -n "$TEST_EXAMPLE" ]; then
        echo "Mode: single test example"
        echo ""
        process_single_example_prompts "$FRAMEWORK" "$TEST_EXAMPLE"
        return $?
    else
        echo "========================================================"
        echo "Mode: process all test examples for framework ${FRAMEWORK}"
        echo "========================================================"
        echo ""
        
        DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}"
        
        if [ ! -d "$DATA_DIR" ]; then
            echo "❌ Error: data directory does not exist: $DATA_DIR"
            echo "Run Step 1 first: parse algorithm methods"
            return 1
        fi
        
        data_files=($(ls "$DATA_DIR"/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v "\.output$" | grep -v "\.result$"))
        
        if [ ${#data_files[@]} -eq 0 ]; then
            echo "❌ Error: no data files found"
            return 1
        fi
        
        echo "Found ${#data_files[@]} data files"
        echo ""
        
        SUCCESS_COUNT=0
        FAIL_COUNT=0
        SKIP_COUNT=0
        
        for data_file in "${data_files[@]}"; do
            filename=$(basename "$data_file")
            example=$(echo "$filename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
            
            echo ""
            echo "----------------------------------------"
            echo "Processing: ${example}"
            echo "----------------------------------------"
            
            process_single_example_prompts "$FRAMEWORK" "$example"
            result=$?
            
            if [ $result -eq 0 ]; then
                ((SUCCESS_COUNT++))
            elif [ $result -eq 1 ]; then
                ((SKIP_COUNT++))
            else
                ((FAIL_COUNT++))
            fi
        done
        
        echo ""
        echo "========================================================"
        echo "Batch build completed"
        echo "========================================================"
        echo "Framework: ${FRAMEWORK}"
        echo "Success: ${SUCCESS_COUNT}"
        echo "Skipped: ${SKIP_COUNT}"
        echo "Failed: ${FAIL_COUNT}"
        echo "========================================================"
        
        [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
    fi
}



# ========================================
# Step 3: OpenRouter API code generation
# ========================================
run_openrouter_api() {
    echo ""
    echo "###########################################################"
    echo "# Step 3: OpenRouter API code generation"
    echo "###########################################################"
    echo ""

    MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
    
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "❌ Error: OPENROUTER_API_KEY is not set"
        echo ""
        echo "Set your API key first:"
        echo "  export OPENROUTER_API_KEY='your-key'"
        echo ""
        echo "Get API key: https://openrouter.ai/keys"
        return 1
    fi
    
    MODEL_DIR_NAME=$(basename "${MODEL_NAME}")
    
    DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}"
    MODEL_OUTPUT_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    
    mkdir -p "${MODEL_OUTPUT_DIR}"
    
    echo "========================================================"
    echo "🤖 OpenRouter API code generation"
    echo "========================================================"
    echo "Model: ${MODEL_NAME}"
    echo "Framework: ${FRAMEWORK}"
    echo "Data directory: ${DATA_DIR}"
    echo "Output directory: ${MODEL_OUTPUT_DIR}"
    echo "Directory name: ${MODEL_DIR_NAME}"
    echo "========================================================"
    echo ""
    
    if [ -n "$TEST_EXAMPLE" ]; then
        echo "Processing single test instance: ${TEST_EXAMPLE}"
        echo ""
        
        INPUT_FILE="${DATA_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.jsonl"
        OUTPUT_FILE="${MODEL_OUTPUT_DIR}/algorithm_methods_data_${TEST_EXAMPLE}_output.jsonl"
        LOG_FILE="${MODEL_OUTPUT_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.log"
        if [ ! -f "$INPUT_FILE" ]; then
            echo "❌ Error: file does not exist: $INPUT_FILE"
            return 1
        fi
        
        python3 ${PROJECT_DIR}/scripts/apicall/generate_completions_openrouter.py \
            --model "${MODEL_NAME}" \
            --input_file "${INPUT_FILE}" \
            --output_file "${OUTPUT_FILE}" \
            --num_completions ${NUM_COMPLETIONS} \
            --max_tokens 30000 \
            --temperature 0.0 \
            --top_p 1.0 \
            --delay 0.5 \
            --debug \
            2>&1 | tee ${LOG_FILE}    
        
        return $?
    else
        echo "Processing all test instances..."
        echo ""
        
        TEST_FILES=($(ls ${DATA_DIR}/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v output))
        
        if [ ${#TEST_FILES[@]} -eq 0 ]; then
            echo "❌ Error: no test files found"
            echo "Directory: ${DATA_DIR}"
            return 1
        fi
        
        echo "Found ${#TEST_FILES[@]} files"
        echo ""
        
        SUCCESS=0
        FAIL=0
        
        for input_file in "${TEST_FILES[@]}"; do
            filename=$(basename "$input_file" .jsonl)
            output_file="${MODEL_OUTPUT_DIR}/${filename}_output.jsonl"
            LOG_FILE="${MODEL_OUTPUT_DIR}/${filename}.log"
            echo "Processing: $(basename $input_file)"
            
            if python3 ${PROJECT_DIR}/scripts/apicall/generate_completions_openrouter.py \
                --model "${MODEL_NAME}" \
                --input_file "${input_file}" \
                --output_file "${output_file}" \
                --num_completions ${NUM_COMPLETIONS} \
                --max_tokens 30000 \
                --temperature 0.0 \
                --top_p 1.0 \
                --delay 0.5 \
                --debug \
                2>&1 | tee ${LOG_FILE}; then
                ((SUCCESS++))
                echo "✅ Done"
            else
                ((FAIL++))
                echo "❌ Failed"
            fi
            
            echo ""
        done
        
        echo "========================================================"
        echo "📊 Processing completed"
        echo "========================================================"
        echo "Total: ${#TEST_FILES[@]}"
        echo "✅ Success: ${SUCCESS}"
        echo "❌ Failed: ${FAIL}"
        echo "Output: ${MODEL_OUTPUT_DIR}"
        echo "========================================================"
        
        if [ $FAIL -gt 0 ]; then
            return 1
        fi
    fi
    
    echo ""
    echo "✅ OpenRouter API code generation completed!"
    return 0
}

# ========================================
# Step 4: Batch execution evaluation (pure mode)
# ========================================
run_batch_execution_evaluation() {
    echo ""
    echo "###########################################################"
    echo "# Step 4: Batch execution evaluation (pure mode)"
    echo "###########################################################"
    echo ""

    MODEL_DIR_NAME=$(basename "${DEFAULT_MODEL}")
    DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    EVAL_SCRIPT="${PROJECT_DIR}/scripts/run_execution_evaluation_pure.sh"
    
    echo "========================================================"
    echo "🔬 Batch execution evaluation (pure mode)"
    echo "========================================================"
    echo "Framework: ${FRAMEWORK}"
    echo "Model: ${MODEL_DIR_NAME}"
    echo "Data directory: ${DATA_DIR}"
    echo "Mode: pure mode - simulates manual workflow"
    echo "========================================================"
    echo ""
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ Error: data directory does not exist: $DATA_DIR"
        echo "Available model directories:"
        ls -d "${PROJECT_DIR}/scripts/data/${FRAMEWORK}"/*/ 2>/dev/null || echo "  (none)"
        return 1
    fi
    
    echo "🔍 Scanning test instances..."
    echo ""
    
    TEST_EXAMPLES_EVAL=()
    while IFS= read -r file; do
        filename=$(basename "$file")
        if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
            test_example="${BASH_REMATCH[1]}"
            TEST_EXAMPLES_EVAL+=("$test_example")
            echo "  ✓ Found: $test_example"
        fi
    done < <(find "$DATA_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)
    
    if [ ${#TEST_EXAMPLES_EVAL[@]} -eq 0 ]; then
        echo ""
        echo "❌ Error: no test instances found"
        echo "Ensure algorithm_methods_data_*_output.jsonl files exist in directory"
        echo ""
        echo "Current directory contents:"
        ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "  (empty)"
        return 1
    fi
    
    echo ""
    echo "📊 Found ${#TEST_EXAMPLES_EVAL[@]} test instances"
    echo "========================================================"
    echo ""
    
    TOTAL=${#TEST_EXAMPLES_EVAL[@]}
    SUCCESS=0
    FAILED=0
    FAILED_TESTS=()
    
    START_TIME=$(date +%s)
    
    for i in "${!TEST_EXAMPLES_EVAL[@]}"; do
        test_example="${TEST_EXAMPLES_EVAL[$i]}"
        index=$((i + 1))
        
        echo ""
        echo "========================================================"
        echo "🔬 [$index/$TOTAL] Pure mode evaluation: $test_example"
        echo "========================================================"
        echo ""
        
        FRAMEWORK="$FRAMEWORK" MODEL_NAME="$MODEL_DIR_NAME" TEST_EXAMPLE="$test_example" bash "$EVAL_SCRIPT"
        
        if [ $? -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
            echo ""
            echo "✅ [$index/$TOTAL] $test_example - evaluation succeeded"
        else
            FAILED=$((FAILED + 1))
            FAILED_TESTS+=("$test_example")
            echo ""
            echo "❌ [$index/$TOTAL] $test_example - evaluation failed"
        fi
        
        echo "========================================================"
        
        if [ $index -lt $TOTAL ]; then
            echo ""
            sleep 1
        fi
    done
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo ""
    echo "========================================================"
    echo "📈 Batch evaluation completed (pure mode)"
    echo "========================================================"
    echo "Framework: ${FRAMEWORK}"
    echo "Model: ${MODEL_DIR_NAME}"
    echo "Data directory: ${DATA_DIR}"
    echo "Evaluation mode: pure mode"
    echo ""
    echo "Total: $TOTAL test instances"
    echo "Success: $SUCCESS"
    echo "Failed: $FAILED"
    echo "Duration: ${DURATION}s"
    echo ""
    
    if [ $FAILED -gt 0 ]; then
        echo "Failed test instances:"
        for test_name in "${FAILED_TESTS[@]}"; do
            echo "  ❌ $test_name"
        done
        echo ""
    fi
    
    echo "========================================================"
    echo "📊 Metrics summary for all test instances (pure mode)"
    echo "========================================================"
    echo ""
    
    for test_example in "${TEST_EXAMPLES_EVAL[@]}"; do
        output_file="${DATA_DIR}/algorithm_methods_data_${test_example}_result.jsonl"
        metrics_file="${output_file//_result.jsonl/_result.metrics.json}"
        
        if [ -f "$metrics_file" ]; then
            echo "[${test_example}]"
            cat "$metrics_file" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    for key, value in data.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.4f}')
        else:
            print(f'  {key}: {value}')
except:
    pass
" 2>/dev/null
            echo ""
        else
            echo "[${test_example}]"
            echo "  ⚠️  Metrics file not found: $(basename "$metrics_file")"
            echo ""
        fi
    done
    
    echo "========================================================"
    echo ""
    
    if [ $FAILED -gt 0 ]; then
        echo "⚠️  Some tests failed, check the failed list above"
        return 1
    else
        echo "✅ All tests completed successfully!"
        export SUCCESSFUL_TEST_EXAMPLES="${TEST_EXAMPLES_EVAL[*]}"
        return 0
    fi
}

# ========================================
# Step 5: Aggregate evaluation metrics
# ========================================
run_aggregate_metrics() {
    echo ""
    echo "###########################################################"
    echo "# Step 5: Aggregate evaluation metrics"
    echo "###########################################################"
    echo ""

    MODEL_DIR_NAME=$(basename "${DEFAULT_MODEL}")
    MODEL_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Error: model directory does not exist: ${MODEL_DIR}"
        return 1
    fi
    
    if [ -z "$SUCCESSFUL_TEST_EXAMPLES" ]; then
        echo "🔍 Scanning test instances..."
        TEST_EXAMPLES_LIST=()
        while IFS= read -r file; do
            filename=$(basename "$file")
            if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
                test_example="${BASH_REMATCH[1]}"
                TEST_EXAMPLES_LIST+=("$test_example")
            fi
        done < <(find "$MODEL_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)
        TEST_EXAMPLES_STR="${TEST_EXAMPLES_LIST[*]}"
    else
        TEST_EXAMPLES_STR="$SUCCESSFUL_TEST_EXAMPLES"
    fi
    
    if [ -z "$TEST_EXAMPLES_STR" ]; then
        echo "❌ Error: no test instances found"
        return 1
    fi
    
    AGGREGATE_SCRIPT="${PROJECT_DIR}/scripts/aggregate_metrics.py"
    if [ ! -f "$AGGREGATE_SCRIPT" ]; then
        echo "❌ Error: aggregate_metrics.py not found"
        return 1
    fi
    
    echo "========================================================"
    echo "📊 Aggregate evaluation metrics"
    echo "========================================================"
    echo "Model directory: ${MODEL_DIR}"
    echo "Test instances: ${TEST_EXAMPLES_STR}"
    echo "Framework: ${FRAMEWORK}"
    echo "========================================================"
    echo ""
    
    cd "${PROJECT_DIR}/scripts"
    
    python3 aggregate_metrics.py \
        --model_dir "${MODEL_DIR}" \
        --test_examples ${TEST_EXAMPLES_STR} \
        --framework "${FRAMEWORK}"
    
    AGGREGATE_RESULT=$?
    
    if [ $AGGREGATE_RESULT -eq 0 ]; then
        echo ""
        echo "✅ Aggregation completed!"
        return 0
    else
        echo ""
        echo "❌ Aggregation failed (exit code: $AGGREGATE_RESULT)"
        return 1
    fi
}

############################################################
# Main: run five steps in sequence
############################################################

echo "============================================================"
echo "🚀 Run full pipeline"
echo "============================================================"
echo "Framework: ${FRAMEWORK}"
echo "Model: ${DEFAULT_MODEL}"
if [ -n "$TEST_EXAMPLE" ]; then
    echo "Test example: ${TEST_EXAMPLE}"
else
    echo "Test example: all"
fi
echo "============================================================"
echo ""

# Step 1: Parse algorithm methods
run_parse_algorithm_methods
STEP1_RESULT=$?
if [ $STEP1_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Step 1 failed, stopping"
    exit $STEP1_RESULT
fi

# Step 2: Build prompts
run_prompts_construction
STEP2_RESULT=$?
if [ $STEP2_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Step 2 failed, stopping"
    exit $STEP2_RESULT
fi

# Step 3: OpenRouter API call
run_openrouter_api
STEP3_RESULT=$?
if [ $STEP3_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Step 3 failed, stopping"
    exit $STEP3_RESULT
fi

# Step 4: Batch execution evaluation (uncomment to run)
# run_batch_execution_evaluation
# STEP4_RESULT=$?
# if [ $STEP4_RESULT -ne 0 ]; then
#     echo ""
#     echo "❌ Step 4 failed, stopping"
#     exit $STEP4_RESULT
# fi

# Step 5: Aggregate metrics (uncomment to run)
# run_aggregate_metrics
# STEP5_RESULT=$?
# if [ $STEP5_RESULT -ne 0 ]; then
#     echo ""
#     echo "❌ Step 5 failed"
#     exit $STEP5_RESULT
# fi

echo ""
echo "============================================================"
echo "🎉 Pipeline completed!"
echo "============================================================"
echo "Framework: ${FRAMEWORK}"
echo "Model: ${DEFAULT_MODEL}"
echo "Steps completed:"
echo "  ✅ 1. Parse algorithm core methods"
echo "  ✅ 2. Build prompts"
echo "  ✅ 3. OpenRouter API code generation"
echo "  ✅ 4. Batch execution evaluation (optional)"
echo "  ✅ 5. Aggregate evaluation metrics (optional)"
echo "============================================================"
exit 0

