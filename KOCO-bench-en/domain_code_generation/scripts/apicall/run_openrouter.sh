#!/bin/bash

# ========================================
# OpenRouter API 代码生成脚本
# ========================================

# 默认配置
DEFAULT_MODEL="qwen/qwen2.5-coder-7b-instruct"
DEFAULT_FRAMEWORK="raganything"
NUM_COMPLETIONS=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --test-example)
            TEST_EXAMPLE="$2"
            shift 2
            ;;
        --num-completions)
            NUM_COMPLETIONS="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model MODEL         模型名称 (默认: $DEFAULT_MODEL)"
            echo "  --framework FRAMEWORK 框架名称 (默认: $DEFAULT_FRAMEWORK)"
            echo "  --test-example NAME   指定单个测试实例"
            echo "  --num-completions N   每个样本生成数量 (默认: 1)"
            echo "  --help                显示帮助"
            echo ""
            echo "支持的模型:"
            echo "  - qwen/qwen2.5-coder-7b-instruct"
            echo "  - qwen/qwen2.5-coder-32b-instruct"
            echo ""
            echo "环境变量:"
            echo "  OPENROUTER_API_KEY    OpenRouter API Key (必需)"
            echo ""
            echo "示例:"
            echo "  export OPENROUTER_API_KEY='sk-or-v1-xxx'"
            echo "  $0 --model qwen/qwen2.5-coder-7b-instruct --framework verl"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 设置默认值
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
FRAMEWORK="${FRAMEWORK:-$DEFAULT_FRAMEWORK}"

# 检查 API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ 错误: 未设置 OPENROUTER_API_KEY"
    echo ""
    echo "请先设置 API Key:"
    echo "  export OPENROUTER_API_KEY='sk-or-v1-xxx'"
    echo ""
    echo "获取 API Key: https://openrouter.ai/keys"
    exit 1
fi

# 处理模型名称：只取最后一部分（去掉 qwen/ 等前缀）
MODEL_DIR_NAME=$(basename "${MODEL_NAME}")

# 设置路径
DATA_DIR="../data/${FRAMEWORK}"
MODEL_OUTPUT_DIR="../data/${FRAMEWORK}/${MODEL_DIR_NAME}"

# 创建输出目录
mkdir -p "${MODEL_OUTPUT_DIR}"

# 显示配置
echo "========================================================"
echo "🤖 OpenRouter API 代码生成"
echo "========================================================"
echo "模型: ${MODEL_NAME}"
echo "框架: ${FRAMEWORK}"
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${MODEL_OUTPUT_DIR}"
echo "目录名称: ${MODEL_DIR_NAME}"
echo "========================================================"
echo ""

# 处理数据
if [ -n "$TEST_EXAMPLE" ]; then
    # 处理单个实例
    echo "处理单个测试实例: ${TEST_EXAMPLE}"
    echo ""
    
    INPUT_FILE="${DATA_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.jsonl"
    OUTPUT_FILE="${MODEL_OUTPUT_DIR}/algorithm_methods_data_${TEST_EXAMPLE}_output.jsonl"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ 错误: 文件不存在: $INPUT_FILE"
        exit 1
    fi
    
    python generate_completions_openrouter.py \
        --model "${MODEL_NAME}" \
        --input_file "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --num_completions ${NUM_COMPLETIONS} \
        --max_tokens 2048 \
        --temperature 0.0 \
        --top_p 1.0 \
        --delay 0.5
    
else
    # 处理所有实例
    echo "处理所有测试实例..."
    echo ""
    
    TEST_FILES=($(ls ${DATA_DIR}/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v output))
    
    if [ ${#TEST_FILES[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到测试文件"
        echo "目录: ${DATA_DIR}"
        exit 1
    fi
    
    echo "找到 ${#TEST_FILES[@]} 个文件"
    echo ""
    
    SUCCESS=0
    FAIL=0
    
    for input_file in "${TEST_FILES[@]}"; do
        filename=$(basename "$input_file" .jsonl)
        output_file="${MODEL_OUTPUT_DIR}/${filename}_output.jsonl"
        
        echo "处理: $(basename $input_file)"
        
        if python generate_completions_openrouter.py \
            --model "${MODEL_NAME}" \
            --input_file "${input_file}" \
            --output_file "${output_file}" \
            --num_completions ${NUM_COMPLETIONS} \
            --max_tokens 2048 \
            --temperature 0.0 \
            --top_p 1.0 \
            --delay 0.5; then
            ((SUCCESS++))
            echo "✅ 完成"
        else
            ((FAIL++))
            echo "❌ 失败"
        fi
        
        echo ""
    done
    
    # 总结
    echo "========================================================"
    echo "📊 处理完成"
    echo "========================================================"
    echo "总数: ${#TEST_FILES[@]}"
    echo "✅ 成功: ${SUCCESS}"
    echo "❌ 失败: ${FAIL}"
    echo "输出: ${MODEL_OUTPUT_DIR}"
    echo "========================================================"
    
    if [ $FAIL -gt 0 ]; then
        exit 1
    fi
fi

echo ""
echo "🎉 完成！"

