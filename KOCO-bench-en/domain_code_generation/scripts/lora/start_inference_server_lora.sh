#!/bin/bash
# Script to start LoRA inference server
# Run inference service in background, loading base model + LoRA adapter

set -e

cd "$(dirname "$0")"

# ========================================
# Configuration
# ========================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Base model path (must be specified)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/your/base/model}"

# LoRA adapter path
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-../models/your_framework-lora}"

# Server configuration
SERVER_PORT="${SERVER_PORT:-8001}"  # Use different port to avoid conflict with SFT server
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# Log files
LOG_FILE="${LOG_FILE:-../logs/inference_server_lora.log}"
PID_FILE="${PID_FILE:-../logs/inference_server_lora.pid}"

# ========================================
# Color Output
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# Create log directories
# ========================================
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$PID_FILE")"

# ========================================
# Check if already running
# ========================================

check_server_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Running
        else
            # PID file exists but process doesn't, clean up PID file
            rm -f "$PID_FILE"
            return 1  # Not running
        fi
    else
        return 1  # Not running
    fi
}

# ========================================
# Health check
# ========================================

check_server_health() {
    local max_retries=600  # Maximum wait 1200 seconds (20 minutes, LoRA loading is slow)
    local retry_delay=2
    
    for i in $(seq 1 $max_retries); do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            return 0  # Healthy
        fi
        sleep $retry_delay
    done
    
    return 1  # Unhealthy
}

# ========================================
# Environment check
# ========================================

echo -e "${BLUE}🔍 Checking environment...${NC}"

if ! python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null; then
    echo -e "${RED}❌ Error: Cannot import PyTorch${NC}"
    echo "Please activate the correct conda environment"
    exit 1
fi

if ! python -c "import fastapi; print('✅ FastAPI')" 2>/dev/null; then
    echo -e "${RED}❌ Error: Cannot import FastAPI${NC}"
    echo "Please install FastAPI: pip install fastapi uvicorn"
    exit 1
fi

if ! python -c "import peft; print('✅ PEFT')" 2>/dev/null; then
    echo -e "${RED}❌ Error: Cannot import PEFT${NC}"
    echo "Please install PEFT: pip install peft"
    exit 1
fi

if [ ! -f "inference_server_lora.py" ]; then
    echo -e "${RED}❌ Error: inference_server_lora.py not found${NC}"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo -e "${RED}❌ Error: Base model path does not exist: ${BASE_MODEL_PATH}${NC}"
    echo "Please set the correct BASE_MODEL_PATH environment variable"
    exit 1
fi

if [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo -e "${RED}❌ Error: LoRA adapter path does not exist: ${LORA_ADAPTER_PATH}${NC}"
    echo "Please set the correct LORA_ADAPTER_PATH environment variable"
    exit 1
fi

# ========================================
# Check existing server
# ========================================

if check_server_running; then
    pid=$(cat "$PID_FILE")
    echo -e "${YELLOW}⚠️  LoRA inference server is already running (PID: ${pid})${NC}"
    echo ""
    echo "Server information:"
    echo "  Address: http://localhost:${SERVER_PORT}"
    echo "  Log: ${LOG_FILE}"
    echo "  PID file: ${PID_FILE}"
    echo ""
    echo "To restart the server, please stop it first:"
    echo "  kill ${pid}"
    echo "  or run: bash scripts/lora/stop_inference_server_lora.sh"
    exit 0
fi

# ========================================
# Start server
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 Starting LoRA inference server${NC}"
echo "========================================================"
echo "Base model: ${BASE_MODEL_PATH}"
echo "LoRA adapter: ${LORA_ADAPTER_PATH}"
echo "Server address: http://${SERVER_HOST}:${SERVER_PORT}"
echo "Max context length: ${MAX_CONTEXT_LEN}"
echo "Data type: ${TORCH_DTYPE}"
echo "Log file: ${LOG_FILE}"
echo "========================================================"
echo ""

echo -e "${BLUE}Starting server (background mode)...${NC}"
echo "This may take a few minutes to load the base model and LoRA adapter..."
echo ""

# Start server (background mode)
nohup python inference_server_lora.py \
    --base_model "$BASE_MODEL_PATH" \
    --lora_adapter "$LORA_ADAPTER_PATH" \
    --port "$SERVER_PORT" \
    --host "$SERVER_HOST" \
    --max_context_len "$MAX_CONTEXT_LEN" \
    --torch_dtype "$TORCH_DTYPE" \
    > "$LOG_FILE" 2>&1 &

# Save PID
server_pid=$!
echo $server_pid > "$PID_FILE"

echo -e "${GREEN}✓ Server started (PID: ${server_pid})${NC}"
echo ""

# ========================================
# Wait for server to be ready
# ========================================

echo -e "${BLUE}Waiting for server to be ready...${NC}"
echo "You can view logs with:"
echo "  tail -f ${LOG_FILE}"
echo ""

if check_server_health; then
    echo -e "${GREEN}✅ LoRA server started successfully!${NC}"
    echo ""
    echo "Server information:"
    echo "  Health check: http://localhost:${SERVER_PORT}/health"
    echo "  Generate endpoint: http://localhost:${SERVER_PORT}/generate"
    echo "  Log file: ${LOG_FILE}"
    echo "  PID file: ${PID_FILE}"
    echo ""
    echo "Test health check:"
    echo "  curl http://localhost:${SERVER_PORT}/health"
    echo ""
    echo "Stop server:"
    echo "  kill ${server_pid}"
    echo "  or run: bash scripts/lora/stop_inference_server_lora.sh"
    echo ""
else
    echo -e "${RED}❌ Server startup failed or timed out${NC}"
    echo ""
    echo "Please check log file: ${LOG_FILE}"
    echo ""
    echo "Last 20 lines of log:"
    echo "========================================"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "Log file is empty or does not exist"
    echo "========================================"
    
    # Cleanup
    if ps -p "$server_pid" > /dev/null 2>&1; then
        echo ""
        echo "Stopping failed server process..."
        kill "$server_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    
    exit 1
fi

echo -e "${GREEN}🎉 LoRA inference server is ready!${NC}"
exit 0

