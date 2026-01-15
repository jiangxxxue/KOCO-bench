#!/usr/bin/env bash
set -euo pipefail

# 一键运行所有测试，使用 conda 环境 smolagents-test，当前目录执行无需 cd
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