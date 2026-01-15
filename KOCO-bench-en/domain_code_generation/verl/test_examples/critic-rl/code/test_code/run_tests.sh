#!/bin/bash

# CTRL Core Functionality Test Runner Script
# Run all tests and display detailed statistics

set -e

echo "========================================"
echo "CTRL Core Functionality Tests"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run test statistics script
python run_tests_with_stats.py

echo ""
echo "========================================"
echo "Tests Completed"
echo "========================================"

