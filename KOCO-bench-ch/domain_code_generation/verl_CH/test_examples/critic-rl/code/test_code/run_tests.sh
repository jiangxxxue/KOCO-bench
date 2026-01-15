#!/bin/bash

# CTRL 核心功能测试运行脚本
# 运行所有测试并显示详细统计信息

set -e

echo "========================================"
echo "CTRL 核心功能测试"
echo "========================================"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 运行测试统计脚本
python run_tests_with_stats.py

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"

