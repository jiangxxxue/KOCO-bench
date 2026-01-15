#!/bin/bash
# Stop inference server script

set -e

cd "$(dirname "$0")"

# Configuration
PID_FILE="${PID_FILE:-../logs/inference_server.pid}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping inference server...${NC}"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  No PID file found: ${PID_FILE}${NC}"
    echo "Server may not be running."
    exit 0
fi

# Read PID
pid=$(cat "$PID_FILE")

# Check if process exists
if ! ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Process ${pid} is not running${NC}"
    echo "Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 0
fi

# Stop the process
echo "Stopping process ${pid}..."
kill "$pid" 2>/dev/null || true

# Wait for process to stop
echo "Waiting for process to stop..."
for i in {1..10}; do
    if ! ps -p "$pid" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Check if still running
if ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Process still running, force killing...${NC}"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
fi

# Clean up PID file
rm -f "$PID_FILE"

echo ""
echo -e "${GREEN}✅ Server stopped successfully${NC}"
exit 0

