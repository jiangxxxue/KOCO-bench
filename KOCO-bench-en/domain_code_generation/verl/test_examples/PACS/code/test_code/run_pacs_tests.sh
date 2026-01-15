#!/bin/bash

# PACS Core Functions Test Runner
# Run all PACS core function tests

echo "=========================================="
echo "Running PACS Core Functions Tests"
echo "=========================================="
echo ""

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../src"

# Run tests and collect results
FAILED=0

echo "Test 1: compute_reward"
echo "----------------------------------------"
python test_compute_reward.py -v
if [ $? -ne 0 ]; then
    FAILED=$((FAILED + 1))
fi
echo ""

echo "Test 2: compute_weight"
echo "----------------------------------------"
python test_compute_weight.py -v
if [ $? -ne 0 ]; then
    FAILED=$((FAILED + 1))
fi
echo ""

echo "Test 3: compute_pacs_loss"
echo "----------------------------------------"
python test_compute_pacs_loss.py -v
if [ $? -ne 0 ]; then
    FAILED=$((FAILED + 1))
fi
echo ""

echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ $FAILED test suite(s) failed"
    exit 1
fi
