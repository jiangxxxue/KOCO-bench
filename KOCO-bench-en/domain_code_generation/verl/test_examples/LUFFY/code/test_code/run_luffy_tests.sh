#!/bin/bash

# LUFFY test runner script
# Used to run all LUFFY core algorithm tests

echo "=========================================="
echo "Running LUFFY Core Algorithm Tests"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/../luffy/verl:${PYTHONPATH}"

# Test result statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a single test file
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .py)
    
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"
    
    if python3 "$test_file" -v; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    echo ""
}

# Run all tests
echo "Starting test execution..."
echo ""

# Test 1: compute_grpo_outcome_advantage_split
run_test "test_compute_grpo_outcome_advantage_split.py"

# Test 2: compute_token_on_off_policy_loss
run_test "test_compute_token_on_off_policy_loss.py"

# Test 3: compute_sft_pure_loss
run_test "test_compute_sft_pure_loss.py"

# Print test summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
else
    echo -e "Failed:       $FAILED_TESTS"
fi
echo "=========================================="

# Return appropriate exit code
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi
