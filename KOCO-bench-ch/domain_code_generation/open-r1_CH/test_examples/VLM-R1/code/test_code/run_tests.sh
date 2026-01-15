#!/bin/bash

# VLM-R1 测试运行脚本
# 运行所有核心函数的测试

echo "=================================="
echo "VLM-R1 Core Function Tests"
echo "=================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 切换到测试目录
cd "$SCRIPT_DIR"

# 运行测试统计脚本
python3 run_tests_with_stats.py

# 保存退出码
EXIT_CODE=$?

echo ""
echo "=================================="
echo "Tests completed with exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
