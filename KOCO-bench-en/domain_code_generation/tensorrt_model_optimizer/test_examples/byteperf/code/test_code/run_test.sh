#!/bin/bash
# ByteMLPerf test runner script

echo "ByteMLPerf Test Suite"
echo "==================="

# Change to script directory
cd "$(dirname "$0")"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed"
    exit 1
fi

# Check if test files exist
if [ ! -f "test_loss_balancer.py" ]; then
    echo "❌ Test file test_loss_balancer.py does not exist"
    exit 1
fi

if [ ! -f "test_mcore_distillation_adjust.py" ]; then
    echo "❌ Test file test_mcore_distillation_adjust.py does not exist"
    exit 1
fi

# Run tests
echo "Running loss balancer tests..."
python test_loss_balancer.py

echo "Running MCore adaptation tests..."
python test_mcore_distillation_adjust.py

echo "All tests completed!"
