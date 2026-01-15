#!/usr/bin/env bash
set -euo pipefail

# Run all tests in one command using conda environment smolagents-test, executed in current directory (no cd required)
ENV="smolagents-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running tests with conda environment: ${ENV}"
conda run -n "${ENV}" python "${SCRIPT_DIR}/run_all_tests.py"
EXITCODE=$?

if [ "${EXITCODE}" -ne 0 ]; then
  echo "Tests failed with exit code ${EXITCODE}"
else
  echo "Tests completed successfully."
fi

exit "${EXITCODE}"