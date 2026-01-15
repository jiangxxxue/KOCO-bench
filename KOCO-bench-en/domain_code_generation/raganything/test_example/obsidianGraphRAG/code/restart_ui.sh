#!/bin/bash
# Restart Obsidian RAG UI Server
# This script stops any running instances and starts fresh

echo ""
echo "======================================================"
echo "Restarting Obsidian RAG UI Server"
echo "======================================================"
echo ""

# Step 1: Stop any running Python processes
echo "[1/3] Stopping existing server processes..."

if pgrep -f "run_ui.py" > /dev/null; then
    echo "   Found running server processes"
    pkill -f "run_ui.py"
    pkill -f "uvicorn"
    echo "   ✓ Stopped running servers"
    sleep 2
else
    echo "   ✓ No running processes found"
fi

# Step 2: Clean up any port locks
echo ""
echo "[2/3] Checking port 8000..."

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "   Port 8000 is in use, attempting to free..."
    kill -9 $(lsof -t -i:8000) 2>/dev/null
    echo "   ✓ Port 8000 freed"
    sleep 2
else
    echo "   ✓ Port 8000 is available"
fi

# Step 3: Start the server
echo ""
echo "[3/3] Starting UI server..."
echo ""

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "turing0.1" ]; then
    echo "WARNING: Not in conda environment 'turing0.1'"
    echo "Please run: conda activate turing0.1"
    echo ""
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Start the server
python run_ui.py


