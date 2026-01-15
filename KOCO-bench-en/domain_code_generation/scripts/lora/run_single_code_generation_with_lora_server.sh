#!/bin/bash
# 单实例代码生成脚本（使用 LoRA 推理服务器）
# 为指定的单个测试实例生成代码

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

FRAMEWORK="${FRAMEWORK:-verl}"
TEST_EXAMPLE="${TEST_EXAMPLE:-prime}"  # 默认测试实例
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
# 解析命令行参数
# ========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --test-example)
            TEST_EXAMPLE="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        --num-completions)
            NUM_COMPLETIONS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --framework FRAMEWORK         框架名称 (默认: verl)"
            echo "  --test-example EXAMPLE        测试实例名称 (默认: prime)"
            echo "  --model-name MODEL            模型名称 (默认: qwen2.5-coder-7b-verl-lora)"
            echo "  --server-url URL              服务器地址 (默认: http://localhost:8001)"
            echo "  --num-completions N           生成数量 (默认: 1)"
            echo "  --temperature T               温度参数 (默认: 0.7)"
            echo "  --help                        显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  FRAMEWORK          框架名称"
            echo "  TEST_EXAMPLE       测试实例名称"
            echo "  MODEL_NAME         模型名称"
            echo "  SERVER_URL         服务器地址"
            echo "  NUM_COMPLETIONS    生成数量"
            echo "  MAX_TOKENS         最大生成长度"
            echo "  TEMPERATURE        温度参数"
            echo "  TOP_P              Top-p 采样"
            echo "  BATCH_SIZE         批处理大小"
            echo "  SKIP_EXISTING      是否跳过已存在文件 (true/false)"
            echo ""
            echo "示例:"
            echo "  # 使用命令行参数"
            echo "  $0 --framework verl --test-example prime"
            echo ""
            echo "  # 使用环境变量"
            echo "  FRAMEWORK=verl TEST_EXAMPLE=ARES $0"
            echo ""
            echo "  # 生成多个补全"
            echo "  $0 --framework verl --test-example prime --num-completions 4 --temperature 0.8"
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

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

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ 错误: 数据目录不存在: ${DATA_DIR}${NC}"
    echo "请先运行数据准备脚本:"
    echo "  FRAMEWORK=${FRAMEWORK} bash scripts/run_parse_algorithm_methods.sh"
    echo "  FRAMEWORK=${FRAMEWORK} bash scripts/run_prompts_construction.sh"
    exit 1
fi

# 构建输入文件路径
INPUT_FILE="${DATA_DIR}/algorithm_methods_data_${TEST_EXAMPLE}.jsonl"

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ 错误: 输入文件不存在: ${INPUT_FILE}${NC}"
    echo ""
    echo "可用的测试实例:"
    ls -1 "${DATA_DIR}"/algorithm_methods_data_*.jsonl 2>/dev/null | \
        sed 's/.*algorithm_methods_data_\(.*\)\.jsonl/  - \1/' || \
        echo "  (未找到任何测试实例)"
    echo ""
    echo "请先运行数据准备脚本:"
    echo "  FRAMEWORK=${FRAMEWORK} TEST_EXAMPLE=${TEST_EXAMPLE} bash scripts/run_parse_algorithm_methods.sh"
    echo "  FRAMEWORK=${FRAMEWORK} TEST_EXAMPLE=${TEST_EXAMPLE} bash scripts/run_prompts_construction.sh"
    exit 1
fi

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 单实例代码生成（使用 LoRA 推理服务器）${NC}"
echo "========================================================"
echo "框架: ${FRAMEWORK}"
echo "测试实例: ${TEST_EXAMPLE}"
echo "模型名称: ${MODEL_NAME}"
echo "服务器: ${SERVER_URL}"
echo "输入文件: ${INPUT_FILE}"
echo "生成数量: ${NUM_COMPLETIONS}"
echo "温度参数: ${TEMPERATURE}"
echo "========================================================"
echo ""

# 检查文件是否为空
if [ ! -s "$INPUT_FILE" ]; then
    echo -e "${RED}❌ 错误: 输入文件为空${NC}"
    exit 1
fi

# 检查是否已经生成过
EXPECTED_OUTPUT="${DATA_DIR}/${MODEL_NAME}/algorithm_methods_data_${TEST_EXAMPLE}_output.jsonl"
if [ -f "$EXPECTED_OUTPUT" ]; then
    echo -e "${YELLOW}⚠️  输出文件已存在: ${EXPECTED_OUTPUT}${NC}"
    
    if [ "$SKIP_EXISTING" = "true" ]; then
        echo "跳过生成 (SKIP_EXISTING=true)"
        echo ""
        echo "如需重新生成，请设置: SKIP_EXISTING=false"
        exit 0
    else
        echo "将覆盖已存在的文件"
        echo ""
    fi
fi

# ========================================
# 执行生成
# ========================================

echo -e "${BLUE}🤖 开始生成代码...${NC}"
echo ""

# 执行生成
set +e

python inference_client_lora.py \
    --server_url "$SERVER_URL" \
    --input_file "$INPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --num_completions $NUM_COMPLETIONS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "/tmp/gen_lora_${TEST_EXAMPLE}.log"

exit_code=${PIPESTATUS[0]}

set -e

# ========================================
# 检查结果
# ========================================

echo ""
echo "========================================================"

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ 代码生成成功！${NC}"
    echo "========================================================"
    echo ""
    echo "输出文件: ${EXPECTED_OUTPUT}"
    
    # 显示统计信息
    if [ -f "$EXPECTED_OUTPUT" ]; then
        num_records=$(wc -l < "$EXPECTED_OUTPUT" 2>/dev/null || echo "0")
        echo "生成记录数: ${num_records}"
    fi
    
    echo ""
    echo "🎉 完成！"
    exit 0
else
    echo -e "${RED}❌ 代码生成失败 (退出码: ${exit_code})${NC}"
    echo "========================================================"
    echo ""
    echo "日志文件: /tmp/gen_lora_${TEST_EXAMPLE}.log"
    echo ""
    echo "请检查:"
    echo "  1. 服务器是否正常运行: curl ${SERVER_URL}/health"
    echo "  2. 输入文件格式是否正确"
    echo "  3. 服务器日志: tail -f ../logs/inference_server_lora.log"
    exit 1
fi

