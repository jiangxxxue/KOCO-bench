#!/bin/bash
# 构建提示词 - 支持多框架、多测试实例

# ========================================
# 配置变量
# ========================================

# verl tensorrt_model_optimizer
# 框架名称
FRAMEWORK="${FRAMEWORK:-raganything}"

# 测试示例名称（为空则处理所有）
TEST_EXAMPLE="${TEST_EXAMPLE:-}"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# ========================================
# 函数：处理单个测试示例
# ========================================
process_single_example() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "构建提示词"
    echo "========================================================"
    echo "框架: ${framework}"
    echo "测试示例: ${test_example}"
    echo "========================================================"
    
    # 构建路径
    local metadata_file="${PROJECT_ROOT}/${framework}/knowledge_corpus/metadata.json"
    local data_dir="${PROJECT_ROOT}/scripts/data/${framework}"
    local data_file="${data_dir}/algorithm_methods_data_${test_example}.jsonl"
    
    echo "元数据文件: ${metadata_file}"
    echo "数据文件: ${data_file}"
    echo ""
    
    # 检查数据文件是否存在
    if [ ! -f "$data_file" ]; then
        echo "⚠️  跳过: 数据文件不存在"
        echo "请先运行: FRAMEWORK=${framework} TEST_EXAMPLE=${test_example} ./scripts/run_parse_algorithm_methods.sh"
        return 1
    fi
    
    # 检查元数据文件
    if [ ! -f "$metadata_file" ]; then
        echo "⚠️  警告: 元数据文件不存在，将使用默认框架描述"
    fi
    
    # 运行构建脚本
    cd "${SCRIPT_DIR}"
    python3 prompts_construction.py \
        --input "$data_file" \
        --metadata "$metadata_file" \
        --output "$data_file"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 提示词构建完成！"
        return 0
    else
        echo ""
        echo "❌ 构建失败"
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
    # 未指定 TEST_EXAMPLE，处理所有已解析的数据文件
    echo "========================================================"
    echo "模式: 处理框架 ${FRAMEWORK} 的所有测试示例"
    echo "========================================================"
    echo ""
    
    DATA_DIR="${PROJECT_ROOT}/scripts/data/${FRAMEWORK}"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ 错误: 数据目录不存在: $DATA_DIR"
        echo "请先运行: FRAMEWORK=${FRAMEWORK} ./scripts/run_parse_algorithm_methods.sh"
        exit 1
    fi
    
    # 查找所有数据文件
    data_files=($(ls "$DATA_DIR"/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v "\.output$" | grep -v "\.result$"))
    
    if [ ${#data_files[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到任何数据文件"
        exit 1
    fi
    
    echo "发现 ${#data_files[@]} 个数据文件"
    echo ""
    
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    SKIP_COUNT=0
    
    # 遍历处理每个数据文件
    for data_file in "${data_files[@]}"; do
        # 从文件名提取测试示例名称
        filename=$(basename "$data_file")
        example=$(echo "$filename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
        
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
    done
    
    # 显示汇总
    echo ""
    echo "========================================================"
    echo "批量构建完成"
    echo "========================================================"
    echo "框架: ${FRAMEWORK}"
    echo "成功: ${SUCCESS_COUNT}"
    echo "跳过: ${SKIP_COUNT}"
    echo "失败: ${FAIL_COUNT}"
    echo "========================================================"
    
    [ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
fi
