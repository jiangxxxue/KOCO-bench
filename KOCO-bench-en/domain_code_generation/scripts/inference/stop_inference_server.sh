#!/bin/bash
# Script to stop inference server

set -e

cd "$(dirname "$0")"

# ========================================
# Configuration
# ========================================

PID_FILE="${PID_FILE:-../logs/inference_server.pid}"
LOG_FILE="${LOG_FILE:-../logs/inference_server.log}"

# ========================================
# Color Output
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# Check if server is running
# ========================================

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  Inference server is not running (PID file does not exist)${NC}"
    exit 0
fi

pid=$(cat "$PID_FILE")

if ! ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Inference server is not running (process ${pid} does not exist)${NC}"
    echo "Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 0
fi

# ========================================
# Stop server
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}🛑 Stopping inference server${NC}"
echo "========================================================"
echo "PID: ${pid}"
echo "Log: ${LOG_FILE}"
echo "========================================================"
echo ""

echo -e "${BLUE}Stopping server...${NC}"

# Send SIGTERM signal
kill "$pid" 2>/dev/null || {
    echo -e "${RED}❌ Failed to stop process ${pid}${NC}"
    exit 1
}

# Wait for process to terminate
max_wait=10
for i in $(seq 1 $max_wait); do
    if ! ps -p "$pid" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Check if successfully stopped
if ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Process did not respond to SIGTERM, sending SIGKILL...${NC}"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
fi

# Clean up PID file
rm -f "$PID_FILE"

echo -e "${GREEN}✅ Inference server stopped${NC}"
echo ""

exit 0

