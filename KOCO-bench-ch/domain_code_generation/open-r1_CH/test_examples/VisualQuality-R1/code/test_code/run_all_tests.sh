#!/bin/bash

# Run all test files for VisualQuality-R1

echo "========================================="
echo "Running VisualQuality-R1 Test Suite"
echo "========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the test directory
cd "$SCRIPT_DIR"

# Array to store test results
declare -a test_results

# Function to run a test and record result
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .py)
    
    echo "Running $test_name..."
    echo "-----------------------------------"
    
    if python3 "$test_file"; then
        echo "✓ $test_name PASSED"
        test_results+=("PASS: $test_name")
    else
        echo "✗ $test_name FAILED"
        test_results+=("FAIL: $test_name")
    fi
    
    echo ""
}

# Run all test files
run_test "test_fidelity_reward.py"
run_test "test_accuracy_reward.py"
run_test "test_format_reward.py"
run_test "test_generate_and_score_completions.py"
run_test "test_compute_loss.py"
run_test "test_repeat_random_sampler.py"

# Print summary
echo "========================================="
echo "Test Summary"
echo "========================================="
for result in "${test_results[@]}"; do
    echo "$result"
done
echo ""

# Count passes and failures
pass_count=$(printf '%s\n' "${test_results[@]}" | grep -c "PASS")
fail_count=$(printf '%s\n' "${test_results[@]}" | grep -c "FAIL")

echo "Total: ${#test_results[@]} tests"
echo "Passed: $pass_count"
echo "Failed: $fail_count"
echo ""

# Exit with appropriate code
if [ $fail_count -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "Some tests failed! ✗"
    exit 1
fi
