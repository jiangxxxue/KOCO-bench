#!/bin/bash
# Batch code generation script (using inference server)
# Automatically iterate through all input files and generate code for each file via inference server

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# Configuration
# ========================================
FRAMEWORK="${FRAMEWORK:-your_framework}"
MODEL_NAME="${MODEL_NAME:-your_model}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"

# Generation parameters
NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-1}"  # Batch size

# Behavior control
SKIP_EXISTING="${SKIP_EXISTING:-false}"  # Default: overwrite existing files

# ========================================
# Color Output
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# Environment Check
# ========================================
echo -e "${BLUE}🔍 Checking environment...${NC}"

# Check Python environment
if ! python -c "import requests; print('✅ requests')" 2>/dev/null; then
    echo -e "${RED}❌ Error: Cannot import requests${NC}"
    echo "Please install requests: pip install requests"
    exit 1
fi

# Check script file
if [ ! -f "inference_client.py" ]; then
    echo -e "${RED}❌ Error: inference_client.py not found${NC}"
    exit 1
fi

# ========================================
# Check server health status
# ========================================

echo -e "${BLUE}🔍 Checking inference server...${NC}"

if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Cannot connect to inference server: ${SERVER_URL}${NC}"
    echo ""
    echo "Please start inference server first:"
    echo "  bash scripts/inference/start_inference_server.sh"
    echo ""
    echo "Or set custom server address:"
    echo "  export SERVER_URL=http://your-server:8000"
    exit 1
fi

# Get server information
server_info=$(curl -s "${SERVER_URL}/health" 2>/dev/null)
server_model=$(echo "$server_info" | python -c "import sys, json; print(json.load(sys.stdin).get('model', 'unknown'))" 2>/dev/null || echo "unknown")

echo -e "${GREEN}✅ Server connection successful${NC}"
echo "  Address: ${SERVER_URL}"
echo "  Model: ${server_model}"

# ========================================
# Find input files
# ========================================

DATA_DIR="../data/${FRAMEWORK}"
MODEL_OUTPUT_DIR="${DATA_DIR}/${MODEL_NAME}"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Data directory does not exist: ${DATA_DIR}${NC}"
    exit 1
fi

# Create model output directory
mkdir -p "${MODEL_OUTPUT_DIR}"

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 Batch Code Generation (using inference server)${NC}"
echo "========================================================"
echo "Framework: ${FRAMEWORK}"
echo "Model name: ${MODEL_NAME}"
echo "Server: ${SERVER_URL}"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${MODEL_OUTPUT_DIR}"
echo "========================================================"
echo ""

# Find all input files (exclude already generated output files)
mapfile -t input_files < <(find "$DATA_DIR" -maxdepth 1 -name "algorithm_methods_data_*.jsonl" \
    ! -name "*_output.jsonl" \
    ! -name "*.result*" \
    ! -name "*.metrics*" \
    -type f | sort)

if [ ${#input_files[@]} -eq 0 ]; then
    echo -e "${RED}❌ Error: No input files found${NC}"
    echo "Directory: ${DATA_DIR}"
    echo "Pattern: algorithm_methods_data_*.jsonl"
    exit 1
fi

echo -e "${GREEN}Found ${#input_files[@]} test instances:${NC}"
for file in "${input_files[@]}"; do
    basename=$(basename "$file")
    example=$(echo "$basename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
    echo "  ✓ $example"
done
echo ""
echo "Starting batch processing..."
echo ""

# ========================================
# Batch processing
# ========================================

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

declare -a SUCCESS_LIST
declare -a FAIL_LIST
declare -a SKIP_LIST

total=${#input_files[@]}
current=0

for file in "${input_files[@]}"; do
    current=$((current + 1))
    basename=$(basename "$file")
    example=$(echo "$basename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
    
    echo "========================================"
    echo -e "${BLUE}[${current}/${total}] Processing: ${example}${NC}"
    echo "========================================"
    echo "Input file: $file"
    
    # Check if file exists and is not empty
    if [ ! -s "$file" ]; then
        echo -e "${YELLOW}⚠️  Skipping: file is empty or does not exist${NC}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        SKIP_LIST+=("$example")
        echo ""
        continue
    fi
    
    # Check if already generated
    expected_output="${MODEL_OUTPUT_DIR}/${basename%.jsonl}_output.jsonl"
    if [ -f "$expected_output" ]; then
        echo -e "${YELLOW}⚠️  Output file already exists: $(basename $expected_output)${NC}"
        
        # Decide behavior based on configuration
        if [ "$SKIP_EXISTING" = "true" ]; then
            echo "Auto-skipping ${example}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            SKIP_LIST+=("$example")
            echo ""
            continue
        else
            # Overwrite existing file
            echo "Will overwrite existing file"
        fi
    fi
    
    # Execute generation (disable set -e to catch errors)
    set +e
    
    python inference_client.py \
        --server_url "$SERVER_URL" \
        --input_file "$file" \
        --model_name "$MODEL_NAME" \
        --num_completions $NUM_COMPLETIONS \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "/tmp/gen_${example}.log"
    
    exit_code=${PIPESTATUS[0]}
    
    set -e
    
    # Check results
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ${example} generation successful${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        SUCCESS_LIST+=("$example")
    else
        echo -e "${RED}❌ ${example} generation failed (exit code: $exit_code)${NC}"
        echo "Log saved to: /tmp/gen_${example}.log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_LIST+=("$example")
    fi
    
    echo ""
done

# ========================================
# Output summary
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}📊 Batch generation completed${NC}"
echo "========================================================"
echo "Total: ${total}"
echo -e "${GREEN}✅ Success: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}❌ Failed: ${FAIL_COUNT}${NC}"
echo -e "${YELLOW}⊗ Skipped: ${SKIP_COUNT}${NC}"
echo "========================================================"

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${GREEN}Successful test instances:${NC}"
    for item in "${SUCCESS_LIST[@]}"; do
        echo "  ✓ $item"
    done
fi

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed test instances:${NC}"
    for item in "${FAIL_LIST[@]}"; do
        echo "  ✗ $item"
    done
fi

if [ ${SKIP_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Skipped test instances:${NC}"
    for item in "${SKIP_LIST[@]}"; do
        echo "  ⊗ $item"
    done
fi

echo ""
echo "Output directory: ${DATA_DIR}/${MODEL_NAME}/"

# Display generated files
if [ -d "${DATA_DIR}/${MODEL_NAME}" ]; then
    echo ""
    echo "Generated files:"
    ls -lh "${DATA_DIR}/${MODEL_NAME}/"*_output.jsonl 2>/dev/null || echo "  (none)"
fi

echo ""

# Set exit code based on results
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠️  ${FAIL_COUNT} test instances failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 All code generation completed!${NC}"
exit 0

