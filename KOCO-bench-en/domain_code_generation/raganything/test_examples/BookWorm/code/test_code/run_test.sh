#!/bin/bash

# Set UTF-8 encoding for proper display
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# RAGAnything Test Auto-Run Script - Bash Version
# Focused on running all tests in my_test_process_with_rag.py

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
CODE_ROOT="$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${CODE_ROOT}:${CODE_ROOT}/code"

echo "=========================================="
echo "RAGAnything Test Auto-Run"
echo "Running all tests in my_test_process_with_rag.py"
echo "=========================================="

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "[Error] Python environment not found"
    exit 1
fi

echo "Python environment: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Project root: $PROJECT_ROOT"
echo "Test directory: $SCRIPT_DIR"
echo "=========================================="
echo ""

# Change to test directory
cd "$SCRIPT_DIR"

# Check if test file exists
test_file="$SCRIPT_DIR/my_test_process_with_rag.py"
if [ ! -f "$test_file" ]; then
    echo "[Error] Test file not found: $test_file"
    exit 1
fi

echo "Running RAGAnything test suite..."
echo "Test file: my_test_process_with_rag.py"
echo "Tests included:"
echo "  - test_process_with_rag_unit: Unit test (fully mocked)"
echo "  - test_rag_integration_with_mock_server: Integration test (using mock server)"
echo ""

# Run pytest tests
echo "Starting pytest..."
echo ""

python -m pytest "$test_file" -v --tb=short --color=yes
exit_code=$?

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo "[Success] All tests passed!"
    echo "[Passed] Unit test: test_process_with_rag_unit"
    echo "[Passed] Integration test: test_rag_integration_with_mock_server"
    echo "[Passed] Test architecture: Using pytest framework, supports async tests"
    echo "[Passed] Mock server: Automatically starts and stops, no manual management needed"
    exit 0
else
    echo "[Warning] Some tests failed, please check the failed tests above"
    echo "Exit code: $exit_code"
    exit $exit_code
fi

