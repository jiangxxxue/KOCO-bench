#!/bin/bash
# 停止推理服务器脚本

set -e

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

PID_FILE="${PID_FILE:-../logs/inference_server.pid}"
LOG_FILE="${LOG_FILE:-../logs/inference_server.log}"

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 检查服务器是否运行
# ========================================

if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  推理服务器未运行（PID 文件不存在）${NC}"
    exit 0
fi

pid=$(cat "$PID_FILE")

if ! ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  推理服务器未运行（进程 ${pid} 不存在）${NC}"
    echo "清理 PID 文件..."
    rm -f "$PID_FILE"
    exit 0
fi

# ========================================
# 停止服务器
# ========================================

echo ""
echo "========================================================"
echo -e "${BLUE}🛑 停止推理服务器${NC}"
echo "========================================================"
echo "PID: ${pid}"
echo "日志: ${LOG_FILE}"
echo "========================================================"
echo ""

echo -e "${BLUE}正在停止服务器...${NC}"

# 发送 SIGTERM 信号
kill "$pid" 2>/dev/null || {
    echo -e "${RED}❌ 无法停止进程 ${pid}${NC}"
    exit 1
}

# 等待进程结束
max_wait=10
for i in $(seq 1 $max_wait); do
    if ! ps -p "$pid" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# 检查是否成功停止
if ps -p "$pid" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  进程未响应 SIGTERM，发送 SIGKILL...${NC}"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
fi

# 清理 PID 文件
rm -f "$PID_FILE"

echo -e "${GREEN}✅ 推理服务器已停止${NC}"
echo ""

exit 0

