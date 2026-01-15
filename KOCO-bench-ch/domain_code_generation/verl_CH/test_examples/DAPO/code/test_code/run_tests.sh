#!/usr/bin/env bash
# DAPO 核心功能测试启动脚本
# 使用Python测试运行器，提供详细的测试统计信息

set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 设置Python路径
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# 切换到测试目录
cd "${SCRIPT_DIR}"

# 运行Python测试运行器
python3 run_tests_with_stats.py "$@"

