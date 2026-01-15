#!/bin/bash
# 批量执行代码评估（纯净模式）- 完全模拟手动操作，不做任何额外处理

# ========================================
# 配置变量
# ========================================

# 框架名称
FRAMEWORK="${FRAMEWORK:-lightRAG}"

# 模型名称（子目录）
MODEL_NAME="${MODEL_NAME:-qwen2.5-coder-7b-instruct}"

# 数据源类型：data 或 rag（默认：data）
DATA_SOURCE="${DATA_SOURCE:-data}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 数据路径（根据DATA_SOURCE切换）
DATA_DIR="${PROJECT_ROOT}/scripts/${DATA_SOURCE}/${FRAMEWORK}/${MODEL_NAME}"

# 评测脚本路径（纯净版）
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/run_execution_evaluation_pure.sh"

# ========================================
# 打印配置信息
# ========================================

echo "========================================================"
echo "🔬 批量执行代码评估（纯净模式）"
echo "========================================================"
echo "框架: ${FRAMEWORK}"
echo "模型: ${MODEL_NAME}"
echo "数据源: ${DATA_SOURCE}"
echo "数据目录: ${DATA_DIR}"
echo "模式: 纯净模式 - 完全模拟手动操作"
echo "========================================================"
echo ""

# ========================================
# 检查数据目录
# ========================================

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 数据目录不存在: $DATA_DIR"
    echo "可用的模型目录:"
    ls -d "${PROJECT_ROOT}/scripts/${DATA_SOURCE}/${FRAMEWORK}"/*/ 2>/dev/null || echo "  无"
    exit 1
fi

# ========================================
# 发现所有测试实例
# ========================================

echo "🔍 扫描测试实例..."
echo ""

# 查找所有 *_output.jsonl 文件并提取测试实例名称
TEST_EXAMPLES=()
while IFS= read -r file; do
    # 提取文件名
    filename=$(basename "$file")
    
    # 提取测试实例名称: algorithm_methods_data_{TEST_EXAMPLE}_output.jsonl
    if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
        test_example="${BASH_REMATCH[1]}"
        TEST_EXAMPLES+=("$test_example")
        echo "  ✓ 发现: $test_example"
    fi
done < <(find "$DATA_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)

# 检查是否找到测试实例
if [ ${#TEST_EXAMPLES[@]} -eq 0 ]; then
    echo ""
    echo "❌ 错误: 未找到任何测试实例"
    echo "请确保目录下存在 algorithm_methods_data_*_output.jsonl 文件"
    echo ""
    echo "当前目录内容:"
    ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "  (空)"
    exit 1
fi

echo ""
echo "📊 共发现 ${#TEST_EXAMPLES[@]} 个测试实例"
echo "========================================================"
echo ""

# ========================================
# 批量执行评测（纯净模式）
# ========================================

# 统计变量
TOTAL=${#TEST_EXAMPLES[@]}
SUCCESS=0
FAILED=0
FAILED_TESTS=()

# 开始时间
START_TIME=$(date +%s)

# 逐个执行评测
for i in "${!TEST_EXAMPLES[@]}"; do
    
    test_example="${TEST_EXAMPLES[$i]}"
    index=$((i + 1))
    
    echo ""
    echo "========================================================"
    echo "🔬 [$index/$TOTAL] 纯净模式评测: $test_example"
    echo "========================================================"
    echo ""
    
    # 设置环境变量并运行纯净版评测脚本
    FRAMEWORK="$FRAMEWORK" MODEL_NAME="$MODEL_NAME" DATA_SOURCE="$DATA_SOURCE" TEST_EXAMPLE="$test_example" bash "$EVAL_SCRIPT"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS + 1))
        echo ""
        echo "✅ [$index/$TOTAL] $test_example - 评测成功"
    else
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_example")
        echo ""
        echo "❌ [$index/$TOTAL] $test_example - 评测失败"
    fi
    
    echo "========================================================"
    
    # 如果不是最后一个，添加分隔
    if [ $index -lt $TOTAL ]; then
        echo ""
        sleep 1
    fi
done

# 结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ========================================
# 汇总结果
# ========================================

echo ""
echo ""
echo "========================================================"
echo "📈 批量评测完成！（纯净模式）"
echo "========================================================"
echo "框架: ${FRAMEWORK}"
echo "模型: ${MODEL_NAME}"
echo "数据源: ${DATA_SOURCE}"
echo "数据目录: ${DATA_DIR}"
echo "评测模式: 纯净模式（完全模拟手动操作）"
echo ""
echo "总计: $TOTAL 个测试实例"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo "耗时: ${DURATION}秒"
echo ""

# 显示失败的测试
if [ $FAILED -gt 0 ]; then
    echo "失败的测试实例:"
    for test_name in "${FAILED_TESTS[@]}"; do
        echo "  ❌ $test_name"
    done
    echo ""
fi

# ========================================
# 汇总所有指标
# ========================================

echo "========================================================"
echo "📊 所有测试实例的指标汇总（纯净模式）"
echo "========================================================"
echo ""

for test_example in "${TEST_EXAMPLES[@]}"; do

    output_file="${DATA_DIR}/algorithm_methods_data_${test_example}_result.jsonl"
    metrics_file="${output_file//_result.jsonl/_result.metrics.json}"
    
    if [ -f "$metrics_file" ]; then
        echo "【${test_example}】"
        cat "$metrics_file" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    for key, value in data.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.4f}')
        else:
            print(f'  {key}: {value}')
except:
    pass
" 2>/dev/null
        echo ""
    else
        echo "【${test_example}】"
        echo "  ⚠️  指标文件不存在: $(basename "$metrics_file")"
        echo ""
    fi
done

echo "========================================================"
echo ""

# ========================================
# 对比模式说明
# ========================================

# 返回适当的退出码
if [ $FAILED -gt 0 ]; then
    echo "⚠️  部分测试失败，请检查上述失败列表"
    exit 1
else
    echo "🎉 所有测试均成功完成！"
    exit 0
fi

