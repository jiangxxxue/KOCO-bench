#!/bin/bash

# PRIME Core Test Runner Script - Focused on core algorithm and configuration regression testing
# These tests don't require complex distributed environments, focusing on verifying core functionality consistency with ground-truth code

# Configure Python environment
export PATH="/mnt/data/jiangxue/miniconda3/envs/verl_sglang/bin:$PATH"

# Get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Set PYTHONPATH to include necessary module paths
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/code/recipe"

echo "=========================================="
echo "PRIME Core Test Execution"
echo "Focused on core algorithm and configuration regression testing"
echo "=========================================="
echo "Python environment: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Change to project root directory
cd "$PROJECT_ROOT"

echo "Running PRIME core algorithm complete tests (7 independent modules)..."
echo "Test objective: Verify all 7 core algorithm functions described in documentation, one dedicated test file per function"
echo ""

# Define test files and descriptions
declare -a TEST_FILES=(
    "test_forward_micro_batch.py:Implicit Process Reward Calculation (_forward_micro_batch)"
    "test_ce_dpo_loss.py:Cross-Entropy DPO Loss (compute_ce_dpo_loss_rm)"
    "test_detach_dpo_loss.py:Detached DPO Loss (compute_detach_dpo_loss_rm)"
    "test_dpo_accuracy.py:DPO Pairwise Comparison Accuracy (compute_dpo_accuracy)"
    "test_dpo_abs_accuracy.py:DPO Absolute Accuracy (compute_dpo_abs_accuracy)"
    "test_rloo_advantage_return.py:RLOO Advantage Estimation (compute_rloo_advantage_return)"
    "test_filter_and_downsample.py:Sample Filtering and Downsampling (filter_and_downsample)"
)

# Run all tests and record results
declare -a EXIT_CODES=()
declare -a TEST_NAMES=()

for test_item in "${TEST_FILES[@]}"; do
    IFS=':' read -r test_file test_desc <<< "$test_item"
    TEST_NAMES+=("$test_desc")
    
    echo "Running test: $test_desc"
    echo "File: $test_file"
    python code/tests/$test_file
    exit_code=$?
    EXIT_CODES+=($exit_code)
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_desc: Passed"
    else
        echo "❌ $test_desc: Failed"
    fi
    echo ""
done

echo "=========================================="
echo "Core Test Results Summary"
echo "=========================================="

# Summarize all test results
all_passed=true
for i in "${!TEST_NAMES[@]}"; do
    test_name="${TEST_NAMES[$i]}"
    exit_code="${EXIT_CODES[$i]}"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name"
    else
        echo "❌ $test_name"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "🎉 All core tests passed!"
    echo "✅ Complete coverage of 7 core algorithm functions described in documentation"
    echo "✅ Each functionality has its own dedicated test file"
    echo "✅ Removed redundant tests unrelated to core algorithms, keeping tests focused and efficient"
    echo "✅ Test architecture: One py file tests one functionality, easy to maintain and understand"
    exit 0
else
    echo "⚠️  Some core tests failed, please check the failed tests above"
    exit 1
fi
