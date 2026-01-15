#!/bin/bash
# Script to quickly run all tests

echo "=========================================="
echo "Running TensorRT-Incubator Unit Tests"
echo "=========================================="
echo ""

# Change to test directory
cd "$(dirname "$0")"

echo "Test directory: $(pwd)"
echo ""

# Count test files
TEST_FILES=$(ls test_*.py 2>/dev/null | wc -l)
echo "Found $TEST_FILES test files"
echo ""

# Run all tests
echo "Starting tests..."
echo "=========================================="
python3 -m unittest discover -s . -p "test_*.py" -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Some tests failed"
    echo "=========================================="
    exit 1
fi

