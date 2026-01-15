#!/bin/bash

# PRIME核心测试运行脚本 - 专注于核心算法和配置的回归测试
# 这些测试不需要复杂的分布式环境，专注于验证核心功能与ground-truth代码的一致性

# 配置Python环境
export PATH="/mnt/data/jiangxue/miniconda3/envs/verl_sglang/bin:$PATH"

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# 设置PYTHONPATH以包含必要的模块路径
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/code/recipe"

echo "=========================================="
echo "PRIME核心测试运行"
echo "专注于核心算法和配置的回归测试"
echo "=========================================="
echo "Python环境: $(which python)"
echo "Python版本: $(python --version)"
echo "工作目录: $(pwd)"
echo "=========================================="

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "运行PRIME核心算法完整测试 (7个独立模块)..."
echo "测试目标: 验证文档中描述的所有7个核心算法函数，每个函数一个独立测试文件"
echo ""

# 定义测试文件和描述
declare -a TEST_FILES=(
    "test_forward_micro_batch.py:隐式过程奖励计算(_forward_micro_batch)"
    "test_ce_dpo_loss.py:交叉熵DPO损失(compute_ce_dpo_loss_rm)"
    "test_detach_dpo_loss.py:分离式DPO损失(compute_detach_dpo_loss_rm)"
    "test_dpo_accuracy.py:DPO成对比较准确率(compute_dpo_accuracy)"
    "test_dpo_abs_accuracy.py:DPO绝对准确率(compute_dpo_abs_accuracy)"
    "test_rloo_advantage_return.py:RLOO优势估计(compute_rloo_advantage_return)"
    "test_filter_and_downsample.py:样本筛选与下采样(filter_and_downsample)"
)

# 运行所有测试并记录结果
declare -a EXIT_CODES=()
declare -a TEST_NAMES=()

for test_item in "${TEST_FILES[@]}"; do
    IFS=':' read -r test_file test_desc <<< "$test_item"
    TEST_NAMES+=("$test_desc")
    
    echo "运行测试: $test_desc"
    echo "文件: $test_file"
    python code/tests/$test_file
    exit_code=$?
    EXIT_CODES+=($exit_code)
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_desc: 通过"
    else
        echo "❌ $test_desc: 失败"
    fi
    echo ""
done

echo "=========================================="
echo "核心测试结果汇总"
echo "=========================================="

# 汇总所有测试结果
all_passed=true
for i in "${!TEST_NAMES[@]}"; do
    test_name="${TEST_NAMES[$i]}"
    exit_code="${EXIT_CODES[$i]}"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $test_name"
    else
        echo "❌ $test_name"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "🎉 所有核心测试全部通过！"
    echo "✅ 完整覆盖文档描述的7个核心算法函数"
    echo "✅ 每个功能点都有独立的专门测试文件"
    echo "✅ 删除了与核心算法无关的冗余测试，保持测试聚焦高效"
    echo "✅ 测试架构：一个py文件测试一个功能点，便于维护和理解"
    exit 0
else
    echo "⚠️  部分核心测试失败，需要检查上述失败的测试"
    exit 1
fi
