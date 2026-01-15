#!/bin/bash

echo "=================================="
echo "VLM-R1 Core Function Tests"
echo "=================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

python3 run_tests_with_stats.py

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Tests completed with exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
