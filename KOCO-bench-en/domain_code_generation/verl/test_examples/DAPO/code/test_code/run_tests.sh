#!/usr/bin/env bash
# DAPO Core Function Test Startup Script
# Use Python test runner to provide detailed test statistics

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# Change to test directory
cd "${SCRIPT_DIR}"

# Run Python test runner
python3 run_tests_with_stats.py "$@"

