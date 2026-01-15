#!/bin/bash

# LUFFY测试运行脚本
# 用于运行所有LUFFY核心算法的测试

echo "=========================================="
echo "Running LUFFY Core Algorithm Tests"
echo "=========================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH="${SCRIPT_DIR}/../luffy/verl:${PYTHONPATH}"

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 运行单个测试文件的函数
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .py)
    
    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"
    
    if python3 "$test_file" -v; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    echo ""
}

# 运行所有测试
echo "Starting test execution..."
echo ""

# 测试1: compute_grpo_outcome_advantage_split
run_test "test_compute_grpo_outcome_advantage_split.py"

# 测试2: compute_token_on_off_policy_loss
run_test "test_compute_token_on_off_policy_loss.py"

# 测试3: compute_sft_pure_loss
run_test "test_compute_sft_pure_loss.py"

# 打印测试总结
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
else
    echo -e "Failed:       $FAILED_TESTS"
fi
echo "=========================================="

# 返回适当的退出码
if [ $FAILED_TESTS -gt 0 ]; then
    exit 1
else
    exit 0
fi
