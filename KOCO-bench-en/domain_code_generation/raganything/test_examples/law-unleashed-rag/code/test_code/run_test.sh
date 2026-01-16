#!/bin/bash

# Set UTF-8 encoding for proper display
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# RAGAnything Test Auto-Run Script - Bash Version
# Focused on running all tests in law-unleashed-rag testcode directory

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
CODE_ROOT="$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${CODE_ROOT}"

echo "=========================================="
echo "Law-Unleashed-RAG Test Auto-Run"
echo "Running all tests in testcode directory"
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

# Define test files and descriptions
declare -a TEST_FILES=(
    "my_test_initialize_rag_storage.py:RAG Storage Initialization Test"
    "my_test_create_rag_instance.py:RAG Instance Creation Test"
)

# Check if all test files exist
all_files_exist=true
for test_item in "${TEST_FILES[@]}"; do
    IFS=':' read -r test_file test_desc <<< "$test_item"
    test_file_path="$SCRIPT_DIR/$test_file"
    if [ ! -f "$test_file_path" ]; then
        echo "[Error] Test file not found: $test_file_path"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    exit 1
fi

echo "Running Law-Unleashed-RAG test suite..."
echo "Test files included:"
for test_item in "${TEST_FILES[@]}"; do
    IFS=':' read -r test_file test_desc <<< "$test_item"
    echo "  - $test_file: $test_desc"
done
echo ""

# Run pytest tests
echo "Starting pytest..."
echo ""

declare -a EXIT_CODES=()
declare -a TEST_NAMES=()
declare -a TEST_DESCS=()

for test_item in "${TEST_FILES[@]}"; do
    IFS=':' read -r test_file test_desc <<< "$test_item"
    TEST_NAMES+=("$test_file")
    TEST_DESCS+=("$test_desc")
    
    echo "Running: $test_file"
    echo "Description: $test_desc"
    
    python -m pytest "$SCRIPT_DIR/$test_file" -v --tb=short --color=yes
    exit_code=$?
    EXIT_CODES+=($exit_code)
    
    if [ $exit_code -eq 0 ]; then
        echo "[Passed] $test_file test passed"
    else
        echo "[Failed] $test_file test failed"
    fi
    echo ""
done

echo "=========================================="
echo "Test Results Summary"
echo "=========================================="

all_passed=true
for i in "${!TEST_NAMES[@]}"; do
    test_name="${TEST_NAMES[$i]}"
    test_desc="${TEST_DESCS[$i]}"
    exit_code="${EXIT_CODES[$i]}"
    
    if [ $exit_code -eq 0 ]; then
        echo "[Success] $test_name: $test_desc"
    else
        echo "[Failed] $test_name: $test_desc"
        all_passed=false
    fi
done

echo ""

if [ "$all_passed" = true ]; then
    echo "[Success] All tests passed!"
    echo "[Passed] Total tests: ${#TEST_FILES[@]}"
    echo "[Passed] All test files executed successfully"
    exit 0
else
    echo "[Warning] Some tests failed, please check the failed tests above"
    failed_count=0
    for exit_code in "${EXIT_CODES[@]}"; do
        if [ $exit_code -ne 0 ]; then
            ((failed_count++))
        fi
    done
    echo "Failed tests: $failed_count / ${#TEST_FILES[@]}"
    exit 1
fi

