#!/usr/bin/env bash
# PURE 核心功能测试脚本
# 运行所有PURE核心算法的单元测试

set -xeuo pipefail

echo "=== PURE 核心功能测试开始 ==="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_CODE_DIR="${SCRIPT_DIR}"

# 设置Python路径
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

# 初始化测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 函数：运行单个测试文件
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    echo "--- 运行测试: ${test_name} ---"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 -m unittest "${test_file}" -v; then
        echo "✓ ${test_name} 测试通过"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ ${test_name} 测试失败"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# 函数：运行单个测试文件（容错模式）
run_test_tolerant() {
    local test_file="$1"
    local test_name="$2"
    
    echo "--- 运行测试 (容错模式): ${test_name} ---"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 -m unittest "${test_file}" -v 2>&1; then
        echo "✓ ${test_name} 测试通过或部分通过"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "✗ ${test_name} 测试遇到问题，但继续执行"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# 确保在正确的目录
cd "${TEST_CODE_DIR}"

# 验证数据路径
echo "=== 验证数据路径 ==="
if [ -f "/workspace/data/math/train.parquet" ]; then
    echo "✓ 训练数据文件存在"
else
    echo "✗ 训练数据文件不存在: /workspace/data/math/train.parquet"
fi

if [ -d "/workspace/data/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" ]; then
    echo "✓ 测试模型已缓存"
else
    echo "✗ 测试模型未缓存，某些集成测试可能失败"
fi

echo "=== 第1阶段: compute_return 测试 ==="
echo "该测试验证token级回报序列计算，包括sum和min两种方法"
echo "测试文件: test_compute_return.py - 直接导入verl.trainer.ppo.core_algos.compute_return函数"
run_test "test_compute_return" "回报序列计算"

echo "=== 第2阶段: ProcessRewardModelWorker._forward_micro_batch 测试 ==="
echo "该测试验证过程奖励模型的token级分数计算和最小式信用分配功能"
echo "测试文件: test_forward_micro_batch.py - 导入ProcessRewardModelWorker类并测试_forward_micro_batch方法"
run_test_tolerant "test_forward_micro_batch" "过程奖励模型前向推断"

echo "=== 第3阶段: ProcessRewardModelWorker.compute_rm_score 测试 ==="
echo "该测试验证批量PRM推断、数据预处理和聚合功能"
echo "测试文件: test_compute_rm_score.py - 导入ProcessRewardModelWorker类并测试compute_rm_score方法"
run_test_tolerant "test_compute_rm_score" "批量过程奖励计算"

echo "=== 测试结果汇总 ==="
echo "总测试数: ${TOTAL_TESTS}"
echo "通过测试: ${PASSED_TESTS}"
echo "失败测试: ${FAILED_TESTS}"

if [ ${FAILED_TESTS} -eq 0 ]; then
    echo "🎉 所有核心功能测试通过！"
    echo ""
    echo "PURE 核心算法功能验证完成 - 重写版本测试："
    echo "✓ compute_return函数 - 直接导入和测试真实函数"
    echo "✓ ProcessRewardModelWorker._forward_micro_batch - 测试PRM前向推断与最小式信用分配"  
    echo "✓ ProcessRewardModelWorker.compute_rm_score - 测试批量处理和数据聚合"
    echo "✓ 最小式信用分配的温度加权机制验证"
    echo "✓ PRM二分类概率差分计算验证"
    echo "✓ 批处理和micro-batch数据流验证"
    echo ""
    echo "测试改进说明："
    echo "- 所有测试现在直接导入和使用PURE项目的真实函数"
    echo "- 移除了重新实现的逻辑，确保测试的是实际代码"
    echo "- 保持了全面的输入输出样例和边缘情况测试"
    echo ""
    echo "建议下一步："
    echo "1. 运行集成测试: ./run_pure_integration.sh" 
    echo "2. 在实际数据上验证LLM生成的函数"
    echo "3. 与原始PURE实现进行对比验证"
    exit 0
else
    echo "❌ ${FAILED_TESTS} 个测试失败"
    echo ""
    echo "失败的测试可能原因："
    echo "1. 依赖环境未正确安装（torch, verl等）"
    echo "2. 测试数据或模型路径不存在"
    echo "3. LLM生成的函数实现与预期不符"
    echo ""
    echo "调试建议："
    echo "1. 检查Python环境和依赖包"
    echo "2. 单独运行失败的测试文件进行调试"
    echo "3. 对比LLM生成的函数与原始实现"
    exit 1
fi
