#!/bin/bash
# Step 1: Parse algorithm methods

# load common config
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# help information
show_usage() {
    echo "Usage: $0 --framework <name> [options]"
    echo ""
    echo "Required:"
    echo "  --framework FRAMEWORK  Framework name (e.g., verl, raganything)"
    echo ""
    echo "Optional:"
    echo "  --test-example NAME    Specify a single test example (default: process all)"
    echo "  --help                 Show help"
    echo ""
    echo "Examples:"
    echo "  bash $0 --framework verl"
    echo "  bash $0 --framework verl --test-example prime"
}

# Parse CLI args
parse_common_args "$@"

# test example name (process all if not specified)
TEST_EXAMPLE="${TEST_EXAMPLE:-}"

# check required parameters
validate_framework

# ========================================
# 函数：处理单个测试示例
# ========================================
process_single_example() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "解析算法核心方法"
    echo "========================================================"
    echo "框架: ${framework}"
    echo "测试示例: ${test_example}"
    echo "========================================================"
    
    # 构建路径
    local input_file="${PROJECT_ROOT}/${framework}/test_examples/${test_example}/requirements/03_algorithm_and_core_methods.md"
    local code_base="${PROJECT_ROOT}/${framework}/test_examples/${test_example}/code"
    local test_base="${PROJECT_ROOT}/${framework}/test_examples/${test_example}/code/tests"
    local output_dir="${PROJECT_ROOT}/scripts/data/${framework}"
    local output_file="${output_dir}/algorithm_methods_data_${test_example}.jsonl"
    
    echo "输入文件: ${input_file}"
    echo "代码库: ${code_base}"
    echo "输出文件: ${output_file}"
    echo ""
    
    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        echo "⚠️  跳过: 输入文件不存在"
        return 1
    fi
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行解析脚本
    cd "${SCRIPTS_DIR}"
    python3 parse_algorithm_methods.py \
        --input "$input_file" \
        --output "$output_file" \
        --code-base "$code_base" \
        --test-base "$test_base"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 解析完成！"
        echo "输出文件: $output_file"
        
        # 显示统计信息
        local num_functions=$(wc -l < "$output_file" 2>/dev/null || echo "0")
        echo "提取函数数量: $num_functions"
        return 0
    else
        echo ""
        echo "❌ 解析失败"
        return 1
    fi
}

# ========================================
# 主逻辑
# ========================================

if [ -n "$TEST_EXAMPLE" ]; then
    # 如果指定了 TEST_EXAMPLE，只处理单个
    echo "模式: 单个测试示例"
    echo ""
    process_single_example "$FRAMEWORK" "$TEST_EXAMPLE"
    exit $?
else
    # 未指定 TEST_EXAMPLE，处理所有
    echo "========================================================"
    echo "模式: 处理框架 ${FRAMEWORK} 的所有测试示例"
    echo "========================================================"
    echo ""
    
    # 获取所有测试示例目录
    TEST_EXAMPLES_DIR="${PROJECT_ROOT}/${FRAMEWORK}/test_examples"
    
    if [ ! -d "$TEST_EXAMPLES_DIR" ]; then
        echo "❌ 错误: 框架目录不存在: $TEST_EXAMPLES_DIR"
        exit 1
    fi
    
    # 查找所有测试示例
    test_examples=($(ls -d "$TEST_EXAMPLES_DIR"/*/ 2>/dev/null | xargs -n 1 basename))
    
    if [ ${#test_examples[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到任何测试示例"
        exit 1
    fi
    
    echo "发现 ${#test_examples[@]} 个测试示例: ${test_examples[*]}"
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0
    
    # 遍历处理每个测试示例
    for example in "${test_examples[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "处理: ${example}"
        echo "----------------------------------------"
        
        process_single_example "$FRAMEWORK" "$example"
        result=$?
        
        if [ $result -eq 0 ]; then
            ((SUCCESS_COUNT++))
        elif [ $result -eq 1 ]; then
            ((SKIP_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
        
        echo ""
    done
    
    # 显示汇总
    echo "========================================================"
    echo "批量解析完成"
    echo "========================================================"
    echo "框架: ${FRAMEWORK}"
    echo "成功: ${SUCCESS_COUNT}"
    echo "跳过: ${SKIP_COUNT}"
    echo "失败: ${FAIL_COUNT}"
    echo "========================================================"
    
    # 如果有失败的，返回失败状态
    [ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
fi
