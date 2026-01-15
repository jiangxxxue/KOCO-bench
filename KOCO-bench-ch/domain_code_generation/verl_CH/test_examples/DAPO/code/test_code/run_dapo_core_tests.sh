#!/bin/bash

# DAPO Core Functions Unit Tests Runner
# 测试DAPO核心算法函数的单元测试脚本

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "DAPO Core Functions Unit Tests"
echo "========================================="
echo ""

# 切换到测试代码目录
cd "$(dirname "$0")"

# 检查Python环境
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
echo ""

# 检查必要的Python包
echo -e "${YELLOW}Checking required packages...${NC}"
REQUIRED_PACKAGES=("torch" "numpy")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package is installed${NC}"
    else
        echo -e "${RED}✗ $package is not installed${NC}"
        echo "  Please install: pip install $package"
        exit 1
    fi
done
echo ""

# 运行测试统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 函数：运行单个测试文件
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo "========================================="
    echo "Testing: $test_name"
    echo "========================================="
    
    if python3 -m unittest $test_file -v; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# 运行各个测试
echo "========================================="
echo "Running DAPO Core Function Tests"
echo "========================================="
echo ""

# 测试1: agg_loss
run_test "test_agg_loss" "agg_loss (Flexible Loss Aggregation)"

# 测试2: compute_policy_loss_vanilla
run_test "test_compute_policy_loss_vanilla" "compute_policy_loss_vanilla (Decoupled Clipping)"

# 测试3: DAPORewardManager.__call__
run_test "test_dapo_reward_manager" "DAPORewardManager.__call__ (Overlong Reward Shaping)"

# 输出测试总结
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
else
    echo -e "Failed:       ${GREEN}$FAILED_TESTS${NC}"
fi
echo "========================================="

# 返回适当的退出代码
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed successfully!${NC}"
    exit 0
fi

