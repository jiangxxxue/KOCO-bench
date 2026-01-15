#!/bin/bash
# 启动 LoRA 推理服务器脚本
# 后台运行推理服务，加载基础模型 + LoRA adapter

set -e

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

export CUDA_VISIBLE_DEVICES=1

# 基础模型路径（必须指定）
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/shixianjie/models/Qwen2.5-Coder-7B-Instruct}"

# LoRA adapter 路径
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-../models/qwen2.5-coder-7b-verl-lora}"

# 服务器配置
SERVER_PORT="${SERVER_PORT:-8001}"  # 使用不同端口避免和 SFT 服务器冲突
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

# 日志文件
LOG_FILE="${LOG_FILE:-../logs/inference_server_lora.log}"
PID_FILE="${PID_FILE:-../logs/inference_server_lora.pid}"

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 创建日志目录
# ========================================
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$PID_FILE")"

# ========================================
# 检查是否已经运行
# ========================================

check_server_running() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # 运行中
        else
            # PID 文件存在但进程不存在，清理 PID 文件
            rm -f "$PID_FILE"
            return 1  # 未运行
        fi
    else
        return 1  # 未运行
    fi
}

# ========================================
# 健康检查
# ========================================

check_server_health() {
    local max_retries=600  # 最多等待 1200 秒（20分钟，LoRA加载较慢）
    local retry_delay=2
    
    for i in $(seq 1 $max_retries); do
        if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
            return 0  # 健康
        fi
        sleep $retry_delay
    done
    
    return 1  # 不健康
}

# ========================================
# 环境检查
# ========================================

echo -e "${BLUE}🔍 检查环境...${NC}"

if ! python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null; then
    echo -e "${RED}❌ 错误: 无法导入 PyTorch${NC}"
    echo "请先激活正确的 conda 环境"
    exit 1
fi

if ! python -c "import fastapi; print('✅ FastAPI')" 2>/dev/null; then
    echo -e "${RED}❌ 错误: 无法导入 FastAPI${NC}"
    echo "请安装 FastAPI: pip install fastapi uvicorn"
    exit 1
fi

if ! python -c "import peft; print('✅ PEFT')" 2>/dev/null; then
    echo -e "${RED}❌ 错误: 无法导入 PEFT${NC}"
    echo "请安装 PEFT: pip install peft"
    exit 1
fi

if [ ! -f "inference_server_lora.py" ]; then
    echo -e "${RED}❌ 错误: 找不到 inference_server_lora.py${NC}"
    exit 1
fi

if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo -e "${RED}❌ 错误: 基础模型路径不存在: ${BASE_MODEL_PATH}${NC}"
    echo "请设置正确的 BASE_MODEL_PATH 环境变量"
    exit 1
fi

if [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo -e "${RED}❌ 错误: LoRA adapter 路径不存在: ${LORA_ADAPTER_PATH}${NC}"
    echo "请设置正确的 LORA_ADAPTER_PATH 环境变量"
    exit 1
fi

# ========================================
# 检查现有服务器
# ========================================

if check_server_running; then
    pid=$(cat "$PID_FILE")
    echo -e "${YELLOW}⚠️  LoRA 推理服务器已经在运行中 (PID: ${pid})${NC}"
    echo ""
    echo "服务器信息:"
    echo "  地址: http://localhost:${SERVER_PORT}"
    echo "  日志: ${LOG_FILE}"
    echo "  PID 文件: ${PID_FILE}"
    echo ""
    echo "如果需要重启服务器，请先停止："
    echo "  kill ${pid}"
    echo "  或者运行: bash scripts/lora/stop_inference_server_lora.sh"
    exit 0
fi

# ========================================
# 启动服务器
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}🚀 启动 LoRA 推理服务器${NC}"
echo "========================================================"
echo "基础模型: ${BASE_MODEL_PATH}"
echo "LoRA adapter: ${LORA_ADAPTER_PATH}"
echo "服务器地址: http://${SERVER_HOST}:${SERVER_PORT}"
echo "最大上下文长度: ${MAX_CONTEXT_LEN}"
echo "数据类型: ${TORCH_DTYPE}"
echo "日志文件: ${LOG_FILE}"
echo "========================================================"
echo ""

echo -e "${BLUE}正在启动服务器（后台运行）...${NC}"
echo "这可能需要几分钟时间来加载基础模型和 LoRA adapter..."
echo ""

# 启动服务器（后台运行）
nohup python inference_server_lora.py \
    --base_model "$BASE_MODEL_PATH" \
    --lora_adapter "$LORA_ADAPTER_PATH" \
    --port "$SERVER_PORT" \
    --host "$SERVER_HOST" \
    --max_context_len "$MAX_CONTEXT_LEN" \
    --torch_dtype "$TORCH_DTYPE" \
    > "$LOG_FILE" 2>&1 &

# 保存 PID
server_pid=$!
echo $server_pid > "$PID_FILE"

echo -e "${GREEN}✓ 服务器已启动 (PID: ${server_pid})${NC}"
echo ""

# ========================================
# 等待服务器就绪
# ========================================

echo -e "${BLUE}等待服务器就绪...${NC}"
echo "你可以通过以下命令查看日志："
echo "  tail -f ${LOG_FILE}"
echo ""

if check_server_health; then
    echo -e "${GREEN}✅ LoRA 服务器启动成功！${NC}"
    echo ""
    echo "服务器信息:"
    echo "  健康检查: http://localhost:${SERVER_PORT}/health"
    echo "  生成接口: http://localhost:${SERVER_PORT}/generate"
    echo "  日志文件: ${LOG_FILE}"
    echo "  PID 文件: ${PID_FILE}"
    echo ""
    echo "测试健康检查:"
    echo "  curl http://localhost:${SERVER_PORT}/health"
    echo ""
    echo "停止服务器:"
    echo "  kill ${server_pid}"
    echo "  或者运行: bash scripts/lora/stop_inference_server_lora.sh"
    echo ""
else
    echo -e "${RED}❌ 服务器启动失败或超时${NC}"
    echo ""
    echo "请检查日志文件: ${LOG_FILE}"
    echo ""
    echo "最后 20 行日志:"
    echo "========================================"
    tail -n 20 "$LOG_FILE" 2>/dev/null || echo "日志文件为空或不存在"
    echo "========================================"
    
    # 清理
    if ps -p "$server_pid" > /dev/null 2>&1; then
        echo ""
        echo "正在停止失败的服务器进程..."
        kill "$server_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    
    exit 1
fi

echo -e "${GREEN}🎉 LoRA 推理服务器准备就绪！${NC}"
exit 0

