#!/bin/bash

#########需要手动修改的四个地方###########
### 1. OPENROUTER_API_KEY
### 2. DEFAULT_MODEL
### 3. DEFAULT_FRAMEWORK
### 4. PROJECT_DIR

# 默认配置
#meta-llama/llama-3.1-8b-instruct  
#qwen/qwen2.5-coder-7b-instruct
#qwen/qwen-2.5-coder-32b-instruct
#deepseek/deepseek-chat-v3.1
#moonshotai/kimi-k2-0905
#google/gemini-3-pro-preview  这个不行，没办法成功生成 
#换成 google/gemini-2.5-pro
#anthropic/claude-sonnet-4.5
#openai/gpt-5-mini
#openai/o4-mini
# 修改为自己的api key

export OPENROUTER_API_KEY='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'

DEFAULT_MODEL="qwen/qwen-2.5-coder-32b-instruct"
DEFAULT_FRAMEWORK="verl"
NUM_COMPLETIONS=1

PROJECT_DIR="/KOCO-bench/KOCO-bench-en/domain_code_generation"

##################parse algorithm methods###################
# 测试示例名称（为空则处理所有）
FRAMEWORK="${FRAMEWORK:-$DEFAULT_FRAMEWORK}"
TEST_EXAMPLE="${TEST_EXAMPLE:-}"
#prime PURE
# 项目根目录
SCRIPT_DIR="${PROJECT_DIR}/scripts"

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
    local input_file="${PROJECT_DIR}/${framework}/test_examples/${test_example}/requirements/03_algorithm_and_core_methods.md"
    local code_base="${PROJECT_DIR}/${framework}/test_examples/${test_example}/code"
    local test_base="${PROJECT_DIR}/${framework}/test_examples/${test_example}/code/tests"


    local output_dir="${PROJECT_DIR}/scripts/data/${framework}"
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
    cd "${SCRIPT_DIR}"
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
# 主逻辑 - 第一部分：解析算法方法
# ========================================
run_parse_algorithm_methods() {
    echo ""
    echo "###########################################################"
    echo "# 第一步：解析算法核心方法"
    echo "###########################################################"
    echo ""

    if [ -n "$TEST_EXAMPLE" ]; then
        # 如果指定了 TEST_EXAMPLE，只处理单个
        echo "模式: 单个测试示例"
        echo ""
        process_single_example "$FRAMEWORK" "$TEST_EXAMPLE"
        return $?
    else
        # 未指定 TEST_EXAMPLE，处理所有
        echo "========================================================"
        echo "模式: 处理框架 ${FRAMEWORK} 的所有测试示例"
        echo "========================================================"
        echo ""
        
        # 获取所有测试示例目录
        TEST_EXAMPLES_DIR="${PROJECT_DIR}/${FRAMEWORK}/test_examples"
        
        if [ ! -d "$TEST_EXAMPLES_DIR" ]; then
            echo "❌ 错误: 框架目录不存在: $TEST_EXAMPLES_DIR"
            return 1
        fi
        
        # 查找所有测试示例
        test_examples=($(ls -d "$TEST_EXAMPLES_DIR"/*/ 2>/dev/null | xargs -n 1 basename))
        
        if [ ${#test_examples[@]} -eq 0 ]; then
            echo "❌ 错误: 未找到任何测试示例"
            return 1
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
        [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
    fi
}


############################################################
##########################run prompts construction###################
process_single_example_prompts() {
    local framework=$1
    local test_example=$2
    
    echo "========================================================"
    echo "构建提示词"
    echo "========================================================"
    echo "框架: ${framework}"
    echo "测试示例: ${test_example}"
    echo "========================================================"
    
    # 构建路径
    local metadata_file="${PROJECT_DIR}/${framework}/knowledge_corpus/metadata.json"


    # 222222222222这里需要改
    local data_dir="${PROJECT_DIR}/scripts/data/${framework}"
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
# 主逻辑 - 第二部分：构建提示词
# ========================================
run_prompts_construction() {
    echo ""
    echo "###########################################################"
    echo "# 第二步：构建提示词"
    echo "###########################################################"
    echo ""

    if [ -n "$TEST_EXAMPLE" ]; then
        # 如果指定了 TEST_EXAMPLE，只处理单个
        echo "模式: 单个测试示例"
        echo ""
        process_single_example_prompts "$FRAMEWORK" "$TEST_EXAMPLE"
        return $?
    else
        # 未指定 TEST_EXAMPLE，处理所有已解析的数据文件
        echo "========================================================"
        echo "模式: 处理框架 ${FRAMEWORK} 的所有测试示例"
        echo "========================================================"
        echo ""
        
        DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}"
        
        if [ ! -d "$DATA_DIR" ]; then
            echo "❌ 错误: 数据目录不存在: $DATA_DIR"
            echo "请先运行第一步：解析算法方法"
            return 1
        fi
        
        # 查找所有数据文件
        data_files=($(ls "$DATA_DIR"/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v "\.output$" | grep -v "\.result$"))
        
        if [ ${#data_files[@]} -eq 0 ]; then
            echo "❌ 错误: 未找到任何数据文件"
            return 1
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
            
            process_single_example_prompts "$FRAMEWORK" "$example"
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
        
        [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
    fi
}



##########################openrouter api###################
# ========================================
# 主逻辑 - 第三部分：OpenRouter API 调用
# ========================================
run_openrouter_api() {
    echo ""
    echo "###########################################################"
    echo "# 第三步：OpenRouter API 代码生成"
    echo "###########################################################"
    echo ""

    # 设置默认值
    MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"
    
    # 检查 API Key
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "❌ 错误: 未设置 OPENROUTER_API_KEY"
        echo ""
        echo "请先设置 API Key:"
        echo "  export OPENROUTER_API_KEY='sk-or-v1-xxx'"
        echo ""
        echo "获取 API Key: https://openrouter.ai/keys"
        return 1
    fi
    
    # 处理模型名称：只取最后一部分（去掉 qwen/ 等前缀）
    MODEL_DIR_NAME=$(basename "${MODEL_NAME}")
    
    # 设置路径
    DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}"
    MODEL_OUTPUT_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    
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
            return 1
        fi
        
        python3 ${PROJECT_DIR}/scripts/apicall/generate_completions_openrouter.py \
            --model "${MODEL_NAME}" \
            --input_file "${INPUT_FILE}" \
            --output_file "${OUTPUT_FILE}" \
            --num_completions ${NUM_COMPLETIONS} \
            --max_tokens 30000 \
            --temperature 0.0 \
            --top_p 1.0 \
            --delay 0.5 \
            --debug \
            2>&1 | tee ${LOG_FILE}    
        
        return $?
    else
        # 处理所有实例
        echo "处理所有测试实例..."
        echo ""
        
        TEST_FILES=($(ls ${DATA_DIR}/algorithm_methods_data_*.jsonl 2>/dev/null | grep -v output))
        
        if [ ${#TEST_FILES[@]} -eq 0 ]; then
            echo "❌ 错误: 未找到测试文件"
            echo "目录: ${DATA_DIR}"
            return 1
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
            
            # o4-mini 是推理模型，需要更多 tokens（推理+生成）
            if python3 ${PROJECT_DIR}/scripts/apicall/generate_completions_openrouter.py \
                --model "${MODEL_NAME}" \
                --input_file "${input_file}" \
                --output_file "${output_file}" \
                --num_completions ${NUM_COMPLETIONS} \
                --max_tokens 30000 \
                --temperature 0.0 \
                --top_p 1.0 \
                --delay 0.5 \
                --debug \
                2>&1 | tee ${LOG_FILE}; then
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
            return 1
        fi
    fi
    
    echo ""
    echo "✅ OpenRouter API 代码生成完成！"
    return 0
}

##########################batch execution evaluation###################
# ========================================
# 主逻辑 - 第四部分：批量执行代码评估
# ========================================
run_batch_execution_evaluation() {
    echo ""
    echo "###########################################################"
    echo "# 第四步：批量执行代码评估（纯净模式）"
    echo "###########################################################"
    echo ""

    # 处理模型名称：只取最后一部分（去掉 qwen/ 等前缀）
    MODEL_DIR_NAME=$(basename "${DEFAULT_MODEL}")
    
    # 数据路径
    DATA_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    
    # 评测脚本路径（纯净版）
    EVAL_SCRIPT="${PROJECT_DIR}/scripts/run_execution_evaluation_pure.sh"
    
    # 打印配置信息
    echo "========================================================"
    echo "🔬 批量执行代码评估（纯净模式）"
    echo "========================================================"
    echo "框架: ${FRAMEWORK}"
    echo "模型: ${MODEL_DIR_NAME}"
    echo "数据目录: ${DATA_DIR}"
    echo "模式: 纯净模式 - 完全模拟手动操作"
    echo "========================================================"
    echo ""
    
    # 检查数据目录
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ 错误: 数据目录不存在: $DATA_DIR"
        echo "可用的模型目录:"
        ls -d "${PROJECT_DIR}/scripts/data/${FRAMEWORK}"/*/ 2>/dev/null || echo "  无"
        return 1
    fi
    
    # 发现所有测试实例
    echo "🔍 扫描测试实例..."
    echo ""
    
    # 查找所有 *_output.jsonl 文件并提取测试实例名称
    TEST_EXAMPLES_EVAL=()
    while IFS= read -r file; do
        # 提取文件名
        filename=$(basename "$file")
        
        # 提取测试实例名称: algorithm_methods_data_{TEST_EXAMPLE}_output.jsonl
        if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
            test_example="${BASH_REMATCH[1]}"
            TEST_EXAMPLES_EVAL+=("$test_example")
            echo "  ✓ 发现: $test_example"
        fi
    done < <(find "$DATA_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)
    
    # 检查是否找到测试实例
    if [ ${#TEST_EXAMPLES_EVAL[@]} -eq 0 ]; then
        echo ""
        echo "❌ 错误: 未找到任何测试实例"
        echo "请确保目录下存在 algorithm_methods_data_*_output.jsonl 文件"
        echo ""
        echo "当前目录内容:"
        ls -lh "$DATA_DIR"/*.jsonl 2>/dev/null || echo "  (空)"
        return 1
    fi
    
    echo ""
    echo "📊 共发现 ${#TEST_EXAMPLES_EVAL[@]} 个测试实例"
    echo "========================================================"
    echo ""
    
    # 批量执行评测（纯净模式）
    # 统计变量
    TOTAL=${#TEST_EXAMPLES_EVAL[@]}
    SUCCESS=0
    FAILED=0
    FAILED_TESTS=()
    
    # 开始时间
    START_TIME=$(date +%s)
    
    # 逐个执行评测
    for i in "${!TEST_EXAMPLES_EVAL[@]}"; do
        
        test_example="${TEST_EXAMPLES_EVAL[$i]}"
        index=$((i + 1))
        
        echo ""
        echo "========================================================"
        echo "🔬 [$index/$TOTAL] 纯净模式评测: $test_example"
        echo "========================================================"
        echo ""
        
        # 设置环境变量并运行纯净版评测脚本
        FRAMEWORK="$FRAMEWORK" MODEL_NAME="$MODEL_DIR_NAME" TEST_EXAMPLE="$test_example" bash "$EVAL_SCRIPT"
        
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
    
    # 汇总结果
    echo ""
    echo ""
    echo "========================================================"
    echo "📈 批量评测完成！（纯净模式）"
    echo "========================================================"
    echo "框架: ${FRAMEWORK}"
    echo "模型: ${MODEL_DIR_NAME}"
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
    
    # 汇总所有指标
    echo "========================================================"
    echo "📊 所有测试实例的指标汇总（纯净模式）"
    echo "========================================================"
    echo ""
    
    for test_example in "${TEST_EXAMPLES_EVAL[@]}"; do
    
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
    
    # 返回适当的退出码
    if [ $FAILED -gt 0 ]; then
        echo "⚠️  部分测试失败，请检查上述失败列表"
        return 1
    else
        echo "✅ 所有测试均成功完成！"
        # 将成功的测试实例列表保存到全局变量，供下一步使用
        export SUCCESSFUL_TEST_EXAMPLES="${TEST_EXAMPLES_EVAL[*]}"
        return 0
    fi
}

##########################aggregate metrics###################
# ========================================
# 主逻辑 - 第五部分：聚合评估指标
# ========================================
run_aggregate_metrics() {
    echo ""
    echo "###########################################################"
    echo "# 第五步：聚合评估指标"
    echo "###########################################################"
    echo ""

    # 处理模型名称：只取最后一部分
    MODEL_DIR_NAME=$(basename "${DEFAULT_MODEL}")
    
    # 模型目录路径
    MODEL_DIR="${PROJECT_DIR}/scripts/data/${FRAMEWORK}/${MODEL_DIR_NAME}"
    
    # 检查模型目录
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ 错误: 模型目录不存在: ${MODEL_DIR}"
        return 1
    fi
    
    # 获取测试实例列表（从上一步的结果或重新扫描）
    if [ -z "$SUCCESSFUL_TEST_EXAMPLES" ]; then
        echo "🔍 扫描测试实例..."
        TEST_EXAMPLES_LIST=()
        while IFS= read -r file; do
            filename=$(basename "$file")
            if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
                test_example="${BASH_REMATCH[1]}"
                TEST_EXAMPLES_LIST+=("$test_example")
            fi
        done < <(find "$MODEL_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)
        TEST_EXAMPLES_STR="${TEST_EXAMPLES_LIST[*]}"
    else
        TEST_EXAMPLES_STR="$SUCCESSFUL_TEST_EXAMPLES"
    fi
    
    if [ -z "$TEST_EXAMPLES_STR" ]; then
        echo "❌ 错误: 未找到任何测试实例"
        return 1
    fi
    
    # 检查 Python 脚本
    AGGREGATE_SCRIPT="${PROJECT_DIR}/scripts/aggregate_metrics.py"
    if [ ! -f "$AGGREGATE_SCRIPT" ]; then
        echo "❌ 错误: 找不到 aggregate_metrics.py"
        return 1
    fi
    
    # 执行聚合
    echo "========================================================"
    echo "📊 聚合评估指标"
    echo "========================================================"
    echo "模型目录: ${MODEL_DIR}"
    echo "测试实例: ${TEST_EXAMPLES_STR}"
    echo "框架: ${FRAMEWORK}"
    echo "========================================================"
    echo ""
    
    # 切换到 scripts 目录执行
    cd "${PROJECT_DIR}/scripts"
    
    python3 aggregate_metrics.py \
        --model_dir "${MODEL_DIR}" \
        --test_examples ${TEST_EXAMPLES_STR} \
        --framework "${FRAMEWORK}"
    
    AGGREGATE_RESULT=$?
    
    if [ $AGGREGATE_RESULT -eq 0 ]; then
        echo ""
        echo "✅ 聚合完成！"
        return 0
    else
        echo ""
        echo "❌ 聚合失败 (退出码: $AGGREGATE_RESULT)"
        return 1
    fi
}

############################################################
# 主执行逻辑：依次执行五个步骤
############################################################

echo "============================================================"
echo "🚀 一键运行完整流程"
echo "============================================================"
echo "框架: ${FRAMEWORK}"
echo "模型: ${DEFAULT_MODEL}"
if [ -n "$TEST_EXAMPLE" ]; then
    echo "测试示例: ${TEST_EXAMPLE}"
else
    echo "测试示例: 所有"
fi
echo "============================================================"
echo ""

# 第一步：解析算法方法
run_parse_algorithm_methods
STEP1_RESULT=$?
if [ $STEP1_RESULT -ne 0 ]; then
    echo ""
    echo "❌ 第一步失败，停止执行"
    exit $STEP1_RESULT
fi

# 第二步：构建提示词
run_prompts_construction
STEP2_RESULT=$?
if [ $STEP2_RESULT -ne 0 ]; then
    echo ""
    echo "❌ 第二步失败，停止执行"
    exit $STEP2_RESULT
fi

# 第三步：OpenRouter API 调用
run_openrouter_api
STEP3_RESULT=$?
if [ $STEP3_RESULT -ne 0 ]; then
    echo ""
    echo "❌ 第三步失败，停止执行"
    exit $STEP3_RESULT
fi

# # 第四步：批量执行代码评估
# run_batch_execution_evaluation
# STEP4_RESULT=$?
# if [ $STEP4_RESULT -ne 0 ]; then
#     echo ""
#     echo "❌ 第四步失败，停止执行"
#     exit $STEP4_RESULT
# fi

# # 第五步：聚合评估指标
# run_aggregate_metrics
# STEP5_RESULT=$?
# if [ $STEP5_RESULT -ne 0 ]; then
#     echo ""
#     echo "❌ 第五步失败"
#     exit $STEP5_RESULT
# fi

# 全部完成
echo ""
echo "============================================================"
echo "🎉 全部流程执行完成！"
echo "============================================================"
echo "框架: ${FRAMEWORK}"
echo "模型: ${DEFAULT_MODEL}"
echo "共完成 5 个步骤："
echo "  ✅ 1. 解析算法核心方法"
echo "  ✅ 2. 构建提示词"
echo "  ✅ 3. OpenRouter API 代码生成"
echo "  ✅ 4. 批量执行代码评估"
echo "  ✅ 5. 聚合评估指标"
echo "============================================================"
exit 0

