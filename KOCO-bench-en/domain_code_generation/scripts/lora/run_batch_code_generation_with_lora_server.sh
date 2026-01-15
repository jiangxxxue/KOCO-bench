#!/bin/bash
# 批量代码生成脚本（使用 LoRA 推理服务器）
# 自动遍历所有输入文件，通过请求 LoRA 推理服务器为每个文件生成代码

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

FRAMEWORK="${FRAMEWORK:-verl}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-coder-7b-verl-lora}"
SERVER_URL="${SERVER_URL:-http://localhost:8001}"  # LoRA 服务器默认端口 8001

# 生成参数
NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-1}"  # 批处理大小

# 行为控制
SKIP_EXISTING="${SKIP_EXISTING:-false}"  # 默认覆盖已存在的文件

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 环境检查
# ========================================
echo -e "${BLUE}🔍 检查环境...${NC}"

# 检查 Python 环境
if ! python -c "import requests; print('✅ requests')" 2>/dev/null; then
    echo -e "${RED}❌ 错误: 无法导入 requests${NC}"
    echo "请安装 requests: pip install requests"
    exit 1
fi

# 检查脚本文件
if [ ! -f "inference_client_lora.py" ]; then
    echo -e "${RED}❌ 错误: 找不到 inference_client_lora.py${NC}"
    exit 1
fi

# ========================================
# 检查服务器健康状态
# ========================================

echo -e "${BLUE}🔍 检查 LoRA 推理服务器...${NC}"

if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ 错误: 无法连接到推理服务器: ${SERVER_URL}${NC}"
    echo ""
    echo "请先启动 LoRA 推理服务器:"
    echo "  bash scripts/lora/start_inference_server_lora.sh"
    echo ""
    echo "或者设置自定义服务器地址:"
    echo "  export SERVER_URL=http://your-server:8001"
    exit 1
fi

# 获取服务器信息
server_info=$(curl -s "${SERVER_URL}/health" 2>/dev/null)
server_base_model=$(echo "$server_info" | python -c "import sys, json; print(json.load(sys.stdin).get('base_model', 'unknown'))" 2>/dev/null || echo "unknown")
server_lora=$(echo "$server_info" | python -c "import sys, json; print(json.load(sys.stdin).get('lora_adapter', 'unknown'))" 2>/dev/null || echo "unknown")

echo -e "${GREEN}✅ 服务器连接成功${NC}"
echo "  地址: ${SERVER_URL}"
echo "  基础模型: ${server_base_model}"
echo "  LoRA adapter: ${server_lora}"

# ========================================
# 查找输入文件
# ========================================

DATA_DIR="../data/${FRAMEWORK}"
MODEL_OUTPUT_DIR="${DATA_DIR}/${MODEL_NAME}"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ 错误: 数据目录不存在: ${DATA_DIR}${NC}"
    exit 1
fi

# 创建模型输出目录
mkdir -p "${MODEL_OUTPUT_DIR}"

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 批量代码生成（使用 LoRA 推理服务器）${NC}"
echo "========================================================"
echo "框架: ${FRAMEWORK}"
echo "模型名称: ${MODEL_NAME}"
echo "服务器: ${SERVER_URL}"
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${MODEL_OUTPUT_DIR}"
echo "========================================================"
echo ""

# 查找所有输入文件（排除已生成的输出文件）
mapfile -t input_files < <(find "$DATA_DIR" -maxdepth 1 -name "algorithm_methods_data_*.jsonl" \
    ! -name "*_output.jsonl" \
    ! -name "*.result*" \
    ! -name "*.metrics*" \
    -type f | sort)

if [ ${#input_files[@]} -eq 0 ]; then
    echo -e "${RED}❌ 错误: 未找到任何输入文件${NC}"
    echo "目录: ${DATA_DIR}"
    echo "匹配模式: algorithm_methods_data_*.jsonl"
    exit 1
fi

echo -e "${GREEN}找到 ${#input_files[@]} 个测试实例:${NC}"
for file in "${input_files[@]}"; do
    basename=$(basename "$file")
    example=$(echo "$basename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
    echo "  ✓ $example"
done
echo ""
echo "开始批量处理..."
echo ""

# ========================================
# 批量处理
# ========================================

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

declare -a SUCCESS_LIST
declare -a FAIL_LIST
declare -a SKIP_LIST

total=${#input_files[@]}
current=0

for file in "${input_files[@]}"; do
    current=$((current + 1))
    basename=$(basename "$file")
    example=$(echo "$basename" | sed 's/algorithm_methods_data_\(.*\)\.jsonl/\1/')
    
    echo "========================================"
    echo -e "${BLUE}[${current}/${total}] 处理: ${example}${NC}"
    echo "========================================"
    echo "输入文件: $file"
    
    # 检查文件是否存在且非空
    if [ ! -s "$file" ]; then
        echo -e "${YELLOW}⚠️  跳过: 文件为空或不存在${NC}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        SKIP_LIST+=("$example")
        echo ""
        continue
    fi
    
    # 检查是否已经生成过
    expected_output="${MODEL_OUTPUT_DIR}/${basename%.jsonl}_output.jsonl"
    if [ -f "$expected_output" ]; then
        echo -e "${YELLOW}⚠️  输出文件已存在: $(basename $expected_output)${NC}"
        
        # 根据配置决定行为
        if [ "$SKIP_EXISTING" = "true" ]; then
            echo "自动跳过 ${example}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            SKIP_LIST+=("$example")
            echo ""
            continue
        else
            # 覆盖已存在的文件
            echo "将覆盖已存在的文件"
        fi
    fi
    
    # 执行生成（禁用 set -e 以捕获错误）
    set +e
    
    python inference_client_lora.py \
        --server_url "$SERVER_URL" \
        --input_file "$file" \
        --model_name "$MODEL_NAME" \
        --num_completions $NUM_COMPLETIONS \
        --max_tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "/tmp/gen_lora_${example}.log"
    
    exit_code=${PIPESTATUS[0]}
    
    set -e
    
    # 检查结果
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ${example} 生成成功${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        SUCCESS_LIST+=("$example")
    else
        echo -e "${RED}❌ ${example} 生成失败 (退出码: $exit_code)${NC}"
        echo "日志已保存到: /tmp/gen_lora_${example}.log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_LIST+=("$example")
    fi
    
    echo ""
done

# ========================================
# 输出总结
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}📊 批量生成完成${NC}"
echo "========================================================"
echo "总数: ${total}"
echo -e "${GREEN}✅ 成功: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}❌ 失败: ${FAIL_COUNT}${NC}"
echo -e "${YELLOW}⊗ 跳过: ${SKIP_COUNT}${NC}"
echo "========================================================"

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${GREEN}成功的测试实例:${NC}"
    for item in "${SUCCESS_LIST[@]}"; do
        echo "  ✓ $item"
    done
fi

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${RED}失败的测试实例:${NC}"
    for item in "${FAIL_LIST[@]}"; do
        echo "  ✗ $item"
    done
fi

if [ ${SKIP_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}跳过的测试实例:${NC}"
    for item in "${SKIP_LIST[@]}"; do
        echo "  ⊗ $item"
    done
fi

echo ""
echo "输出目录: ${MODEL_OUTPUT_DIR}/"

# 显示生成的文件
if [ -d "${MODEL_OUTPUT_DIR}" ]; then
    echo ""
    echo "已生成的文件:"
    ls -lh "${MODEL_OUTPUT_DIR}/"*_output.jsonl 2>/dev/null || echo "  (暂无)"
fi

echo ""

# 根据结果设置退出码
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠️  有 ${FAIL_COUNT} 个测试实例生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 所有代码生成完成！${NC}"
exit 0

