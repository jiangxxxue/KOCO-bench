#!/bin/bash

# Run all test files for ml-diffucoder

echo "========================================="
echo "Running ml-diffucoder Test Suite"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Counter for test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test file
run_test() {
    local test_file=$1
    echo "Running $test_file..."
    echo "-----------------------------------------"
    
    if python3 "$test_file"; then
        echo "✓ $test_file PASSED"
        ((PASSED_TESTS++))
    else
        echo "✗ $test_file FAILED"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
    echo ""
}

# Run each test file
run_test "test_forward_process.py"
run_test "test_selective_log_softmax.py"
run_test "test_get_per_token_logps.py"
run_test "test_generate_and_score_completions.py"
run_test "test_code_reward.py"
run_test "test_code_format_reward.py"

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed."
    exit 1
fi
