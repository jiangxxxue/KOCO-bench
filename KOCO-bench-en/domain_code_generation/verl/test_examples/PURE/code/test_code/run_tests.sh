#!/usr/bin/env bash
# DAPO core functionality test launcher script
# Use Python test runner to provide detailed test statistics

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# Switch to test directory
cd "${SCRIPT_DIR}"

# Run Python test runner
python3 run_tests_with_stats.py "$@"

