#!/bin/bash
# Single instance code generation script (using LoRA inference server)
# Generate code for a specified single test instance

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# Configuration
# ========================================

FRAMEWORK="${FRAMEWORK:-your_framework}"
TEST_EXAMPLE="${TEST_EXAMPLE:-prime}"  # Default test instance
MODEL_NAME="${MODEL_NAME:-your_framework-lora}"
SERVER_URL="${SERVER_URL:-http://localhost:8001}"  # LoRA server default port 8001

# Generation parameters
NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-1}"  # Batch size

# Behavior control
SKIP_EXISTING="${SKIP_EXISTING:-false}"  # Default: overwrite existing files

# ========================================
# Parse command line arguments
# ========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --test-example)
            TEST_EXAMPLE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --num-completions)
            NUM_COMPLETIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --framework FRAMEWORK         Framework name (default: your_framework)"
            echo "  --test-example EXAMPLE        Test instance name (default: prime)"
            echo "  --model-name MODEL            Model name (default: your_framework-lora)"
            echo "  --server-url URL              Server address (default: http://localhost:8001)"
            echo "  --num-completions N           Number of completions (default: 1)"
            echo "  --temperature T               Temperature parameter (default: 0.7)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  FRAMEWORK          Framework name"
            echo "  TEST_EXAMPLE       Test instance name"
            echo "  MODEL_NAME         Model name"
            echo "  SERVER_URL         Server address"
            echo "  NUM_COMPLETIONS    Number of completions"
            echo "  MAX_TOKENS         Maximum generation length"
            echo "  TEMPERATURE        Temperature parameter"
            echo "  TOP_P              Top-p sampling"
            echo "  BATCH_SIZE         Batch size"
            echo "  SKIP_EXISTING      Whether to skip existing files (true/false)"
            echo ""
            echo "Examples:"
            echo "  # Using command line arguments"
            echo "  $0 --framework your_framework --test-example prime"
            echo ""
            echo "  # Using environment variables"
            echo "  FRAMEWORK=your_framework TEST_EXAMPLE=prime $0"
            echo ""
            echo "  # Generate multiple completions"
            echo "  $0 --framework your_framework --test-example prime --num-completions 4 --temperature 0.8"
            exit 0
            ;;
        *)
            echo "❌ Unknown parameter: $1"
            echo "Use --help to see help information"
            exit 1
            ;;
    esac
done

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
if [ ! -f "inference_client_lora.py" ]; then
    echo -e "${RED}❌ Error: inference_client_lora.py not found${NC}"
    exit 1
fi

# ========================================
# Check server health status
# ========================================

echo -e "${BLUE}🔍 Checking LoRA inference server...${NC}"

if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Cannot connect to inference server: ${SERVER_URL}${NC}"
    echo ""
    echo "Please start LoRA inference server first:"
    echo "  bash scripts/lora/start_inference_server_lora.sh"
    echo ""
    echo "Or set custom server address:"
    echo "  export SERVER_URL=http://your-server:8001"
    exit 1
fi

# Get server information
server_info=$(curl -s "${SERVER_URL}/health" 2>/dev/null)
server_base_model=$(echo "$server_info" | python -c "import sys, json; print(json.load(sys.stdin).get('base_model', 'unknown'))" 2>/dev/null || echo "unknown")
server_lora=$(echo "$server_info" | python -c "import sys, json; print(json.load(sys.stdin).get('lora_adapter', 'unknown'))" 2>/dev/null || echo "unknown")

echo -e "${GREEN}✅ Server connection successful${NC}"
echo "  Address: ${SERVER_URL}"
echo "  Base model: ${server_base_model}"
echo "  LoRA adapter: ${server_lora}"

# ========================================
# Find input file
# ========================================

DATA_DIR="../data/${FRAMEWORK}"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Data directory does not exist: ${DATA_DIR}${NC}"
    echo "Please run data preparation scripts first:"
    echo "  FRAMEWORK=${FRAMEWORK} bash scripts/run_parse_algorithm_methods.sh"
    echo "  FRAMEWORK=${FRAMEWORK} bash scripts/run_prompts_construction.sh"
    exit 1
fi

# Build input file path
INPUT_FILE="${DATA_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.jsonl"

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Error: Input file does not exist: ${INPUT_FILE}${NC}"
    echo ""
    echo "Available test instances:"
    ls -1 "${DATA_DIR}"/algorithm_methods_data_*.jsonl 2>/dev/null | \
        sed 's/.*algorithm_methods_data_\(.*\)\.jsonl/  - \1/' || \
        echo "  (No test instances found)"
    echo ""
    echo "Please run data preparation scripts first:"
    echo "  FRAMEWORK=${FRAMEWORK} TEST_EXAMPLE=${TEST_EXAMPLE} bash scripts/run_parse_algorithm_methods.sh"
    echo "  FRAMEWORK=${FRAMEWORK} TEST_EXAMPLE=${TEST_EXAMPLE} bash scripts/run_prompts_construction.sh"
    exit 1
fi

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 Single Instance Code Generation (using LoRA inference server)${NC}"
echo "========================================================"
echo "Framework: ${FRAMEWORK}"
echo "Test instance: ${TEST_EXAMPLE}"
echo "Model name: ${MODEL_NAME}"
echo "Server: ${SERVER_URL}"
echo "Input file: ${INPUT_FILE}"
echo "Completions: ${NUM_COMPLETIONS}"
echo "Temperature: ${TEMPERATURE}"
echo "========================================================"
echo ""

# Check if file is empty
if [ ! -s "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Error: Input file is empty${NC}"
    exit 1
fi

# Check if already generated
EXPECTED_OUTPUT="${DATA_DIR}/${MODEL_NAME}/algorithm_methods_data_${TEST_EXAMPLE}_output.jsonl"
if [ -f "$EXPECTED_OUTPUT" ]; then
    echo -e "${YELLOW}⚠️  Output file already exists: ${EXPECTED_OUTPUT}${NC}"
    
    if [ "$SKIP_EXISTING" = "true" ]; then
        echo "Skipping generation (SKIP_EXISTING=true)"
        echo ""
        echo "To regenerate, set: SKIP_EXISTING=false"
        exit 0
    else
        echo "Will overwrite existing file"
        echo ""
    fi
fi

# ========================================
# Execute generation
# ========================================

echo -e "${BLUE}🤖 Starting code generation...${NC}"
echo ""

# Execute generation
set +e

python inference_client_lora.py \
    --server_url "$SERVER_URL" \
    --input_file "$INPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --num_completions $NUM_COMPLETIONS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "/tmp/gen_lora_${TEST_EXAMPLE}.log"

exit_code=${PIPESTATUS[0]}

set -e

# ========================================
# Check results
# ========================================

echo ""
echo "========================================================"

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ Code generation successful!${NC}"
    echo "========================================================"
    echo ""
    echo "Output file: ${EXPECTED_OUTPUT}"
    
    # Display statistics
    if [ -f "$EXPECTED_OUTPUT" ]; then
        num_records=$(wc -l < "$EXPECTED_OUTPUT" 2>/dev/null || echo "0")
        echo "Generated records: ${num_records}"
    fi
    
    echo ""
    echo "🎉 Completed!"
    exit 0
else
    echo -e "${RED}❌ Code generation failed (exit code: ${exit_code})${NC}"
    echo "========================================================"
    echo ""
    echo "Log file: /tmp/gen_lora_${TEST_EXAMPLE}.log"
    echo ""
    echo "Please check:"
    echo "  1. Is server running normally: curl ${SERVER_URL}/health"
    echo "  2. Is input file format correct"
    echo "  3. Server logs: tail -f ../logs/inference_server_lora.log"
    exit 1
fi

