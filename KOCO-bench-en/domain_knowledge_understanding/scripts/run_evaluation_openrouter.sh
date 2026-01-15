#!/bin/bash

#########需要手动修改的地方###########
### 1. OPENROUTER_API_KEY
### 2. DEFAULT_MODEL
### 3. PROJECT_DIR

# 默认配置
# 可选模型:
# meta-llama/llama-3.1-8b-instruct
# qwen/qwen2.5-coder-7b-instruct
# qwen/qwen-2.5-coder-32b-instruct
# deepseek/deepseek-chat-v3.1
# moonshotai/kimi-k2-0905
# anthropic/claude-sonnet-4.5
# openai/gpt-5-mini
# openai/o4-mini
#google/gemini-3-pro-preview  这个不行，没办法成功生成 
#换成 google/gemini-2.5-pro

# 修改为自己的 API key
export OPENROUTER_API_KEY='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'

DEFAULT_MODEL="qwen/qwen2.5-coder-7b-instruct"

# 项目根目录
PROJECT_DIR="/KOCO-bench/KOCO-bench-en/domain_knowledge_understanding"

# 脚本目录
SCRIPT_DIR="${PROJECT_DIR}/scripts"

# 问题文件目录
PROBLEMS_DIR="${PROJECT_DIR}/problems"

# 结果输出目录
RESULTS_DIR="${PROJECT_DIR}/results"

# 测试参数
TEMPERATURE=0.0
MAX_TOKENS=4096
TOP_P=1.0
DELAY=0.5

# ========================================
# 函数：获取所有问题文件
# ========================================
get_problem_files() {
    # 查找所有 problems_*_EN.json 文件
    find "${PROBLEMS_DIR}" -name "problems_*_EN.json" -type f | sort
}

# ========================================
# 函数：从文件名提取数据集名称
# ========================================
extract_dataset_name() {
    local filepath=$1
    local filename=$(basename "$filepath")
    # 从 problems_xxx_EN.json 提取 xxx
    local dataset=$(echo "$filename" | sed 's/problems_\(.*\)_EN\.json/\1/')
    echo "$dataset"
}

# ========================================
# 函数：处理单个问题文件
# ========================================
process_single_problem() {
    local model=$1
    local problem_file=$2
    
    local dataset=$(extract_dataset_name "$problem_file")
    
    echo "========================================================"
    echo "评测问题集"
    echo "========================================================"
    echo "模型: ${model}"
    echo "数据集: ${dataset}"
    echo "问题文件: ${problem_file}"
    echo "========================================================"
    echo ""
    
    # 检查问题文件是否存在
    if [ ! -f "$problem_file" ]; then
        echo "❌ 错误: 问题文件不存在"
        return 1
    fi
    
    # 处理模型名称：只取最后一部分（去掉 qwen/ 等前缀）
    local model_dir_name=$(basename "${model}")
    
    # 创建结果目录
    local result_dir="${RESULTS_DIR}/${model_dir_name}"
    mkdir -p "$result_dir"
    
    # 输出文件路径
    local output_file="${result_dir}/results_${dataset}.json"
    
    echo "输出文件: ${output_file}"
    echo ""
    
    # 运行评测
    cd "${SCRIPT_DIR}"
    python3 evaluation_openrouter.py \
        --model "${model}" \
        --input "${problem_file}" \
        --output "${output_file}" \
        --temperature ${TEMPERATURE} \
        --max_tokens ${MAX_TOKENS} \
        --top_p ${TOP_P} \
        --delay ${DELAY}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 评测完成！"
        echo "结果文件: ${output_file}"
        return 0
    else
        echo ""
        echo "❌ 评测失败"
        return 1
    fi
}

# ========================================
# 函数：批量处理所有问题文件
# ========================================
process_all_problems() {
    local model=$1
    
    echo ""
    echo "###########################################################"
    echo "# 批量评测所有问题集"
    echo "###########################################################"
    echo ""
    echo "模型: ${model}"
    echo "========================================================"
    echo ""
    
    # 获取所有问题文件
    local problem_files=($(get_problem_files))
    
    if [ ${#problem_files[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到任何问题文件"
        echo "目录: ${PROBLEMS_DIR}"
        return 1
    fi
    
    echo "发现 ${#problem_files[@]} 个问题文件:"
    for pf in "${problem_files[@]}"; do
        local dataset=$(extract_dataset_name "$pf")
        echo "  - ${dataset}"
    done
    echo ""
    
    # 统计变量
    SUCCESS_COUNT=0
    FAIL_COUNT=0
    
    # 开始时间
    START_TIME=$(date +%s)
    
    # 逐个处理
    for i in "${!problem_files[@]}"; do
        local problem_file="${problem_files[$i]}"
        local dataset=$(extract_dataset_name "$problem_file")
        local index=$((i + 1))
        
        echo ""
        echo "----------------------------------------"
        echo "[${index}/${#problem_files[@]}] 处理: ${dataset}"
        echo "----------------------------------------"
        
        process_single_problem "$model" "$problem_file"
        result=$?
        
        if [ $result -eq 0 ]; then
            ((SUCCESS_COUNT++))
        else
            ((FAIL_COUNT++))
        fi
        
        echo ""
    done
    
    # 结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 显示汇总
    echo ""
    echo "========================================================"
    echo "批量评测完成"
    echo "========================================================"
    echo "模型: ${model}"
    echo "总数: ${#problem_files[@]}"
    echo "✅ 成功: ${SUCCESS_COUNT}"
    echo "❌ 失败: ${FAIL_COUNT}"
    echo "耗时: ${DURATION}秒"
    echo "========================================================"
    
    # 如果有失败的，返回失败状态
    [ $FAIL_COUNT -eq 0 ] && return 0 || return 1
}

# ========================================
# 函数：生成汇总统计
# ========================================
generate_summary() {
    local model=$1
    local model_dir_name=$(basename "${model}")
    local result_dir="${RESULTS_DIR}/${model_dir_name}"
    
    echo ""
    echo "###########################################################"
    echo "# 生成汇总统计"
    echo "###########################################################"
    echo ""
    
    if [ ! -d "$result_dir" ]; then
        echo "❌ 错误: 结果目录不存在: ${result_dir}"
        return 1
    fi
    
    # 查找所有结果文件
    local result_files=($(find "$result_dir" -name "results_*.json" -type f | sort))
    
    if [ ${#result_files[@]} -eq 0 ]; then
        echo "❌ 错误: 未找到任何结果文件"
        return 1
    fi
    
    echo "发现 ${#result_files[@]} 个结果文件"
    echo ""
    
    # 统计变量
    TOTAL_PROBLEMS=0
    TOTAL_CORRECT=0
    
    echo "========================================================"
    echo "📊 各数据集详细结果"
    echo "========================================================"
    echo ""
    
    for result_file in "${result_files[@]}"; do
        local filename=$(basename "$result_file")
        local dataset=$(echo "$filename" | sed 's/results_\(.*\)\.json/\1/')
        
        # 提取统计信息
        local stats=$(python3 -c "
import json
import sys
try:
    with open('${result_file}', 'r', encoding='utf-8') as f:
        data = json.load(f)
        summary = data.get('summary', {})
        total = summary.get('total', 0)
        correct = summary.get('correct', 0)
        accuracy = summary.get('accuracy_percent', 0.0)
        print(f'{total},{correct},{accuracy:.2f}')
except Exception as e:
    print('0,0,0.00', file=sys.stderr)
    sys.exit(1)
")
        
        if [ $? -eq 0 ]; then
            IFS=',' read -r total correct accuracy <<< "$stats"
            TOTAL_PROBLEMS=$((TOTAL_PROBLEMS + total))
            TOTAL_CORRECT=$((TOTAL_CORRECT + correct))
            
            echo "【${dataset}】"
            echo "  Total: ${total}"
            echo "  Correct: ${correct}"
            echo "  Accuracy: ${accuracy}%"
            echo ""
        else
            echo "【${dataset}】"
            echo "  ⚠️  无法读取结果"
            echo ""
        fi
    done
    
    # 总体统计
    if [ $TOTAL_PROBLEMS -gt 0 ]; then
        OVERALL_ACCURACY=$(python3 -c "print(f'{$TOTAL_CORRECT / $TOTAL_PROBLEMS * 100:.2f}')")
    else
        OVERALL_ACCURACY="0.00"
    fi
    
    echo "========================================================"
    echo "📈 总体统计"
    echo "========================================================"
    echo "模型: ${model}"
    echo "数据集数量: ${#result_files[@]}"
    echo "总问题数: ${TOTAL_PROBLEMS}"
    echo "✅ 正确数: ${TOTAL_CORRECT}"
    echo "❌ 错误数: $((TOTAL_PROBLEMS - TOTAL_CORRECT))"
    echo "🎯 总体准确率: ${OVERALL_ACCURACY}%"
    echo "========================================================"
    
    # 保存汇总到文件
    local summary_file="${result_dir}/summary.json"
    python3 -c "
import json
summary = {
    'model': '${model}',
    'num_datasets': ${#result_files[@]},
    'total_problems': ${TOTAL_PROBLEMS},
    'total_correct': ${TOTAL_CORRECT},
    'total_incorrect': ${TOTAL_PROBLEMS} - ${TOTAL_CORRECT},
    'overall_accuracy_percent': ${OVERALL_ACCURACY}
}
with open('${summary_file}', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
"
    
    echo ""
    echo "💾 汇总结果已保存到: ${summary_file}"
    
    return 0
}

# ========================================
# 主执行逻辑
# ========================================

# 解析命令行参数
MODEL="${MODEL:-$DEFAULT_MODEL}"
DATASET="${DATASET:-}"

echo "============================================================"
echo "🚀 KOCO-BENCH Knowledge Understanding 评测"
echo "============================================================"
echo "模型: ${MODEL}"
if [ -n "$DATASET" ]; then
    echo "数据集: ${DATASET}"
else
    echo "数据集: 所有"
fi
echo "============================================================"
echo ""

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

# 创建结果目录
mkdir -p "${RESULTS_DIR}"

# 执行评测
if [ -n "$DATASET" ]; then
    # 单个数据集
    PROBLEM_FILE="${PROBLEMS_DIR}/problems_${DATASET}_EN.json"
    
    if [ ! -f "$PROBLEM_FILE" ]; then
        echo "❌ 错误: 问题文件不存在: ${PROBLEM_FILE}"
        exit 1
    fi
    
    process_single_problem "$MODEL" "$PROBLEM_FILE"
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo ""
        echo "✅ 评测成功完成"
        exit 0
    else
        echo ""
        echo "❌ 评测失败"
        exit 1
    fi
else
    # 所有数据集
    process_all_problems "$MODEL"
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        # 生成汇总统计
        generate_summary "$MODEL"
        
        echo ""
        echo "============================================================"
        echo "🎉 全部评测完成！"
        echo "============================================================"
        exit 0
    else
        echo ""
        echo "⚠️  部分评测失败"
        exit 1
    fi
fi

