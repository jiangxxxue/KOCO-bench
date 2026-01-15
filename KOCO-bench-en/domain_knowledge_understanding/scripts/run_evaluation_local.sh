#!/bin/bash

#########需要手动修改的地方###########
### 1. MODEL_PATH (if different from default)
### 2. SERVER_PORT (if default 8000 is occupied)
### 3. PROJECT_DIR
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
# Default configuration
DEFAULT_SERVER_URL="http://localhost:8000"
DEFAULT_SERVER_PORT=8000

# Project root directory - MODIFY THIS
PROJECT_DIR="/KOCO-bench/KOCO-bench-en/domain_knowledge_understanding"

# Script directory
SCRIPT_DIR="${PROJECT_DIR}/scripts"

# Problems directory
PROBLEMS_DIR="${PROJECT_DIR}/problems"

# Results output directory
RESULTS_DIR="${PROJECT_DIR}/results"

# Evaluation parameters
TEMPERATURE=0.0
MAX_TOKENS=8192
TOP_P=1.0
DELAY=0.5

# ========================================
# Colors
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# Function: Get all problem files
# ========================================
get_problem_files() {
    # Find all problems_*_EN.json files
    find "${PROBLEMS_DIR}" -name "problems_*_EN.json" -type f | sort
}

# ========================================
# Function: Extract dataset name from file path
# ========================================
extract_dataset_name() {
    local filepath=$1
    local filename=$(basename "$filepath")
    # Extract xxx from problems_xxx_EN.json
    local dataset=$(echo "$filename" | sed 's/problems_\(.*\)_EN\.json/\1/')
    echo "$dataset"
}

# ========================================
# Function: Check if server is running
# ========================================
check_server_health() {
    local server_url=$1
    
    if curl -s "${server_url}/health" > /dev/null 2>&1; then
        return 0  # Healthy
    else
        return 1  # Not healthy
    fi
}

# ========================================
# Function: Get model name from server
# ========================================
get_model_name_from_server() {
    local server_url=$1
    
    local model_name=$(curl -s "${server_url}/health" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('model', 'local-model'))" 2>/dev/null || echo "local-model")
    echo "$model_name"
}

# ========================================
# Function: Process single problem file
# ========================================
process_single_problem() {
    local server_url=$1
    local problem_file=$2
    
    local dataset=$(extract_dataset_name "$problem_file")
    
    echo "========================================================"
    echo "Evaluating problem set"
    echo "========================================================"
    echo "Server: ${server_url}"
    echo "Dataset: ${dataset}"
    echo "Problem file: ${problem_file}"
    echo "========================================================"
    echo ""
    
    # Check if problem file exists
    if [ ! -f "$problem_file" ]; then
        echo -e "${RED}❌ Error: Problem file does not exist${NC}"
        return 1
    fi
    
    # Get model name from server
    local model_name=$(get_model_name_from_server "$server_url")
    
    # Create result directory
    local result_dir="${RESULTS_DIR}/${model_name}"
    mkdir -p "$result_dir"
    
    # Output file path
    local output_file="${result_dir}/results_${dataset}.json"
    
    echo "Model: ${model_name}"
    echo "Output file: ${output_file}"
    echo ""
    
    # Run evaluation
    cd "${SCRIPT_DIR}"
    python3 evaluation_local.py \
        --server_url "${server_url}" \
        --input "${problem_file}" \
        --output "${output_file}" \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --top_p ${TOP_P} \
        --delay ${DELAY}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Evaluation complete!${NC}"
        echo "Result file: ${output_file}"
        return 0
    else
        echo ""
        echo -e "${RED}❌ Evaluation failed${NC}"
        return 1
    fi
}

# ========================================
# Function: Batch process all problem files
# ========================================
process_all_problems() {
    local server_url=$1
    
    echo ""
    echo "###########################################################"
    echo "# Batch evaluation of all problem sets"
    echo "###########################################################"
    echo ""
    echo "Server: ${server_url}"
    echo "========================================================"
    echo ""
    
    # Get all problem files
    local problem_files=($(get_problem_files))
    
    if [ ${#problem_files[@]} -eq 0 ]; then
        echo -e "${RED}❌ Error: No problem files found${NC}"
        echo "Directory: ${PROBLEMS_DIR}"
        return 1
    fi
    
    echo "Found ${#problem_files[@]} problem files:"
    for pf in "${problem_files[@]}"; do
        local dataset=$(extract_dataset_name "$pf")
        echo "  - ${dataset}"
    done
    echo ""
    
    # Statistics
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    # Start time
    START_TIME=$(date +%s)
    
    # Process each file
    for i in "${!problem_files[@]}"; do
        local problem_file="${problem_files[$i]}"
        local dataset=$(extract_dataset_name "$problem_file")
        local index=$((i + 1))
        
        echo ""
        echo "----------------------------------------"
        echo "[${index}/${#problem_files[@]}] Processing: ${dataset}"
        echo "----------------------------------------"
        
        process_single_problem "$server_url" "$problem_file"
        result=$?
        
        if [ $result -eq 0 ]; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
        
        echo ""
    done
    
    # End time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # Display summary
    echo ""
    echo "========================================================"
    echo "Batch evaluation complete"
    echo "========================================================"
    echo "Server: ${server_url}"
    echo "Total: ${#problem_files[@]}"
    echo -e "${GREEN}✅ Success: ${SUCCESS_COUNT}${NC}"
    echo -e "${RED}❌ Failed: ${FAIL_COUNT}${NC}"
    echo "Duration: ${DURATION} seconds"
    echo "========================================================"
    
    # Return failure if any failed
    [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
}

# ========================================
# Function: Generate summary statistics
# ========================================
generate_summary() {
    local server_url=$1
    local model_name=$(get_model_name_from_server "$server_url")
    local result_dir="${RESULTS_DIR}/${model_name}"
    
    echo ""
    echo "###########################################################"
    echo "# Generate summary statistics"
    echo "###########################################################"
    echo ""
    
    if [ ! -d "$result_dir" ]; then
        echo -e "${RED}❌ Error: Result directory does not exist: ${result_dir}${NC}"
        return 1
    fi
    
    # Find all result files
    local result_files=($(find "$result_dir" -name "results_*.json" -type f | sort))
    
    if [ ${#result_files[@]} -eq 0 ]; then
        echo -e "${RED}❌ Error: No result files found${NC}"
        return 1
    fi
    
    echo "Found ${#result_files[@]} result files"
    echo ""
    
    # Statistics
    TOTAL_PROBLEMS=0
    TOTAL_CORRECT=0
    
    echo "========================================================"
    echo "📊 Detailed results per dataset"
    echo "========================================================"
    echo ""
    
    for result_file in "${result_files[@]}"; do
        local filename=$(basename "$result_file")
        local dataset=$(echo "$filename" | sed 's/results_\(.*\)\.json/\1/')
        
        # Extract statistics
        local stats=$(python3 -c "
import json
import sys
try:
    with open('${result_file}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        summary = data.get('summary', {})
        total = summary.get('total', 0)
        correct = summary.get('correct', 0)
        accuracy = summary.get('accuracy_percent', 0.0)
        print(f'{total},{correct},{accuracy:.2f}')
except Exception as e:
    print('0,0,0.00', file=sys.stderr)
    sys.exit(1)
")
        
        if [ $? -eq 0 ]; then
            IFS=',' read -r total correct accuracy <<< "$stats"
            TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + total))
            TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
            
            echo "【${dataset}】"
            echo "  Total: ${total}"
            echo "  Correct: ${correct}"
            echo "  Accuracy: ${accuracy}%"
            echo ""
        else
            echo "【${dataset}】"
            echo "  ⚠️  Cannot read results"
            echo ""
        fi
    done
    
    # Overall statistics
    if [ $TOTAL_PROBLEMS -gt 0 ]; then
        OVERALL_ACCURACY=$(python3 -c "print(f'{$TOTAL_CORRECT / $TOTAL_PROBLEMS * 100:.2f}')")
    else
        OVERALL_ACCURACY="0.00"
    fi
    
    echo "========================================================"
    echo "📈 Overall statistics"
    echo "========================================================"
    echo "Model: ${model_name}"
    echo "Number of datasets: ${#result_files[@]}"
    echo "Total problems: ${TOTAL_PROBLEMS}"
    echo -e "${GREEN}✅ Correct: ${TOTAL_CORRECT}${NC}"
    echo -e "${RED}❌ Incorrect: $((TOTAL_PROBLEMS - TOTAL_CORRECT))${NC}"
    echo "🎯 Overall accuracy: ${OVERALL_ACCURACY}%"
    echo "========================================================"
    
    # Save summary to file
    local summary_file="${result_dir}/summary.json"
    python3 -c "
import json
summary = {
    'model': '${model_name}',
    'server_url': '${server_url}',
    'num_datasets': ${#result_files[@]},
    'total_problems': ${TOTAL_PROBLEMS},
    'total_correct': ${TOTAL_CORRECT},
    'total_incorrect': ${TOTAL_PROBLEMS} - ${TOTAL_CORRECT},
    'overall_accuracy_percent': ${OVERALL_ACCURACY}
}
with open('${summary_file}', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
"
    
    echo ""
    echo "💾 Summary saved to: ${summary_file}"
    
    return 0
}

# ========================================
# Main execution logic
# ========================================

# Parse command line arguments
SERVER_URL="${SERVER_URL:-$DEFAULT_SERVER_URL}"
DATASET="${DATASET:-}"

echo "============================================================"
echo "🚀 KOCO-BENCH Knowledge Understanding Evaluation (Local)"
echo "============================================================"
echo "Server: ${SERVER_URL}"
if [ -n "$DATASET" ]; then
    echo "Dataset: ${DATASET}"
else
    echo "Dataset: All"
fi
echo "============================================================"
echo ""

# Check if server is running
echo "Checking inference server..."
if ! check_server_health "$SERVER_URL"; then
    echo -e "${RED}❌ Error: Cannot connect to inference server: ${SERVER_URL}${NC}"
    echo ""
    echo "Please start the inference server first:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  bash start_inference_server.sh"
    echo ""
    echo "Or set custom server URL:"
    echo "  export SERVER_URL=http://your-server:8000"
    exit 1
fi

echo -e "${GREEN}✅ Server is healthy${NC}"
MODEL_NAME=$(get_model_name_from_server "$SERVER_URL")
echo "Model: ${MODEL_NAME}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Execute evaluation
if [ -n "$DATASET" ]; then
    # Single dataset
    PROBLEM_FILE="${PROBLEMS_DIR}/problems_${DATASET}_EN.json"
    
    if [ ! -f "$PROBLEM_FILE" ]; then
        echo -e "${RED}❌ Error: Problem file does not exist: ${PROBLEM_FILE}${NC}"
        exit 1
    fi
    
    process_single_problem "$SERVER_URL" "$PROBLEM_FILE"
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Evaluation completed successfully${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}❌ Evaluation failed${NC}"
        exit 1
    fi
else
    # All datasets
    process_all_problems "$SERVER_URL"
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        # Generate summary statistics
        generate_summary "$SERVER_URL"
        
        echo ""
        echo "============================================================"
        echo "🎉 All evaluations complete!"
        echo "============================================================"
        exit 0
    else
        echo ""
        echo "⚠️  Some evaluations failed"
        exit 1
    fi
fi

