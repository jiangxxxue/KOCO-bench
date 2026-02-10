#!/bin/bash
# Step 3: Generate code via OpenRouter API

# load common config
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

# help information
show_usage() {
    echo "Usage: $0 --framework <name> --model <name> [options]"
    echo ""
    echo "Required:"
    echo "  --framework FRAMEWORK  Framework name (e.g., verl, raganything)"
    echo "  --model MODEL          Full model name (e.g., qwen/qwen-2.5-coder-32b-instruct)"
    echo ""
    echo "Optional:"
    echo "  --test-example NAME    Specify a single test example (default: process all)"
    echo "  --num-completions N    Number of completions per sample (default: 1)"
    echo "  --help                 Show help"
    echo ""
    echo "Environment variables:"
    echo "  OPENROUTER_API_KEY     OpenRouter API Key (required, set via .env)"
    echo ""
    echo "Examples:"
    echo "  bash $0 --framework verl --model qwen/qwen-2.5-coder-32b-instruct"
    echo "  bash $0 --framework verl --model deepseek/deepseek-chat-v3.1 --test-example prime"
}

# Parse CLI args
parse_common_args "$@"

# number of completions (default: 1)
NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"

# check required parameters
validate_required_params

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

# set data and model output directory, use absolute path to ensure correct path from any directory
DATA_DIR="${SCRIPTS_DIR}/data/${FRAMEWORK}"
MODEL_OUTPUT_DIR="${SCRIPTS_DIR}/data/${FRAMEWORK}/${MODEL_DIR_NAME}"

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
    LOG_FILE="${MODEL_OUTPUT_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.log"
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ 错误: 文件不存在: $INPUT_FILE"
        exit 1
    fi
    
    python3 "${SCRIPTS_DIR}/apicall/generate_completions_openrouter.py" \
        --model "${MODEL_NAME}" \
        --input_file "${INPUT_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --num_completions ${NUM_COMPLETIONS} \
        --max_tokens 30000 \
        --temperature 0.0 \
        --top_p 1.0 \
        --delay 0.5 \
        --debug \
        2>&1 | tee "${LOG_FILE}"
    
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
        LOG_FILE="${MODEL_OUTPUT_DIR}/${filename}.log"
        
        echo "处理: $(basename $input_file)"
        
        if python3 "${SCRIPTS_DIR}/apicall/generate_completions_openrouter.py" \
            --model "${MODEL_NAME}" \
            --input_file "${input_file}" \
            --output_file "${output_file}" \
            --num_completions ${NUM_COMPLETIONS} \
            --max_tokens 30000 \
            --temperature 0.0 \
            --top_p 1.0 \
            --delay 0.5 \
            --debug \
            2>&1 | tee "${LOG_FILE}"; then
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

