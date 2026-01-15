#!/usr/bin/env bash
# PURE core functionality test script
# Run all unit tests for PURE core algorithms

set -xeuo pipefail

echo "=== PURE Core Functionality Tests Starting ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_CODE_DIR="${SCRIPT_DIR}"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# Initialize test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function: Run a single test file
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    echo "--- Running test: ${test_name} ---"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 -m unittest "${test_file}" -v; then
        echo "✓ ${test_name} test passed"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ ${test_name} test failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# Function: Run a single test file (tolerant mode)
run_test_tolerant() {
    local test_file="$1"
    local test_name="$2"
    
    echo "--- Running test (tolerant mode): ${test_name} ---"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 -m unittest "${test_file}" -v 2>&1; then
        echo "✓ ${test_name} test passed or partially passed"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ ${test_name} test encountered issues, but continuing execution"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# Ensure we're in the correct directory
cd "${TEST_CODE_DIR}"

# Verify data paths
echo "=== Verifying Data Paths ==="
if [ -f "/workspace/data/math/train.parquet" ]; then
    echo "✓ Training data file exists"
else
    echo "✗ Training data file does not exist: /workspace/data/math/train.parquet"
fi

if [ -d "/workspace/data/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" ]; then
    echo "✓ Test model is cached"
else
    echo "✗ Test model is not cached, some integration tests may fail"
fi

echo "=== Stage 1: compute_return Test ==="
echo "This test verifies token-level return sequence calculation, including sum and min methods"
echo "Test file: test_compute_return.py - directly imports verl.trainer.ppo.core_algos.compute_return function"
run_test "test_compute_return" "Return Sequence Calculation"

echo "=== Stage 2: ProcessRewardModelWorker._forward_micro_batch Test ==="
echo "This test verifies token-level score calculation and min-form credit assignment of process reward model"
echo "Test file: test_forward_micro_batch.py - imports ProcessRewardModelWorker class and tests _forward_micro_batch method"
run_test_tolerant "test_forward_micro_batch" "Process Reward Model Forward Inference"

echo "=== Stage 3: ProcessRewardModelWorker.compute_rm_score Test ==="
echo "This test verifies batch PRM inference, data preprocessing and aggregation functionality"
echo "Test file: test_compute_rm_score.py - imports ProcessRewardModelWorker class and tests compute_rm_score method"
run_test_tolerant "test_compute_rm_score" "Batch Process Reward Calculation"

echo "=== Test Results Summary ==="
echo "Total tests: ${TOTAL_TESTS}"
echo "Passed tests: ${PASSED_TESTS}"
echo "Failed tests: ${FAILED_TESTS}"

if [ ${FAILED_TESTS} -eq 0 ]; then
    echo "🎉 All core functionality tests passed!"
    echo ""
    echo "PURE core algorithm functionality verification completed - rewritten version tests:"
    echo "✓ compute_return function - directly import and test real function"
    echo "✓ ProcessRewardModelWorker._forward_micro_batch - test PRM forward inference and min-form credit assignment"  
    echo "✓ ProcessRewardModelWorker.compute_rm_score - test batch processing and data aggregation"
    echo "✓ Min-form credit assignment temperature weighting mechanism verification"
    echo "✓ PRM binary classification probability difference calculation verification"
    echo "✓ Batch processing and micro-batch data flow verification"
    echo ""
    echo "Test improvement notes:"
    echo "- All tests now directly import and use real functions from the PURE project"
    echo "- Removed re-implemented logic to ensure testing actual code"
    echo "- Maintained comprehensive input/output examples and edge case testing"
    echo ""
    echo "Suggested next steps:"
    echo "1. Run integration tests: ./run_pure_integration.sh" 
    echo "2. Verify LLM-generated functions on actual data"
    echo "3. Compare with original PURE implementation"
    exit 0
else
    echo "❌ ${FAILED_TESTS} test(s) failed"
    echo ""
    echo "Possible reasons for test failures:"
    echo "1. Dependencies not properly installed (torch, verl, etc.)"
    echo "2. Test data or model paths do not exist"
    echo "3. LLM-generated function implementation does not match expectations"
    echo ""
    echo "Debugging suggestions:"
    echo "1. Check Python environment and dependency packages"
    echo "2. Run failed test files individually for debugging"
    echo "3. Compare LLM-generated functions with original implementation"
    exit 1
fi
