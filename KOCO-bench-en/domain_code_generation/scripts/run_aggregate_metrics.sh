#!/bin/bash
# 聚合多个测试实例的评估指标

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

# 默认参数（可通过环境变量覆盖）
MODEL_DIR="${MODEL_DIR:-}"
TEST_EXAMPLES="${TEST_EXAMPLES:-}"
FRAMEWORK="${FRAMEWORK:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========================================
# 帮助信息
# ========================================

show_help() {
    cat << EOF
用法: bash run_aggregate_metrics.sh [选项]

聚合多个测试实例的评估指标，计算综合 pass@1 和 avg_pass_ratio

选项:
  --model_dir DIR          模型输出目录路径（必需）
  --test_examples NAMES    测试实例名称列表，空格分隔（可选，不指定则自动发现）
  --framework NAME         框架名称（可选）
  --output FILE            输出 JSON 文件路径（可选）
  -h, --help               显示此帮助信息

环境变量:
  MODEL_DIR                模型输出目录路径
  TEST_EXAMPLES            测试实例名称列表
  FRAMEWORK                框架名称
  OUTPUT_FILE              输出文件路径

示例:

  # 自动发现所有测试实例（推荐）
  bash run_aggregate_metrics.sh \\
    --model_dir data/verl/qwen2.5-coder-32b-instruct-simple

  # 指定特定测试实例
  bash run_aggregate_metrics.sh \\
    --model_dir data/verl/qwen2.5-coder-32b-instruct-simple \\
    --test_examples "prime ARES LUFFY PURE"

  # 使用环境变量
  export MODEL_DIR="data/verl/qwen2.5-coder-7b-lora"
  bash run_aggregate_metrics.sh

  # 保存结果到文件
  bash run_aggregate_metrics.sh \\
    --model_dir data/verl/qwen2.5-coder-32b-instruct-simple \\
    --output aggregate_result.json

  # 指定框架
  bash run_aggregate_metrics.sh \\
    --model_dir data/verl/qwen2.5-coder-7b-lora \\
    --framework verl

输出说明:
  - pass@1: 所有实例中通过的函数数 / 总函数数
  - avg_pass_ratio: 所有实例的 avg_pass_ratio 按函数数加权平均

EOF
}

# ========================================
# 参数解析
# ========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --test_examples)
            TEST_EXAMPLES="$2"
            shift 2
            ;;
        --framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 错误: 未知参数 '$1'${NC}"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ========================================
# 参数验证
# ========================================

if [ -z "$MODEL_DIR" ]; then
    echo -e "${RED}❌ 错误: 必须指定 --model_dir${NC}"
    echo "使用 -h 或 --help 查看帮助信息"
    exit 1
fi

# TEST_EXAMPLES 现在是可选的，不指定则自动发现

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}❌ 错误: 模型目录不存在: ${MODEL_DIR}${NC}"
    exit 1
fi

# ========================================
# 检查 Python 脚本
# ========================================

if [ ! -f "aggregate_metrics.py" ]; then
    echo -e "${RED}❌ 错误: 找不到 aggregate_metrics.py${NC}"
    exit 1
fi

# ========================================
# 执行聚合
# ========================================

echo "========================================================"
echo -e "${BLUE}📊 聚合评估指标${NC}"
echo "========================================================"
echo "模型目录: ${MODEL_DIR}"
if [ -n "$TEST_EXAMPLES" ]; then
    echo "测试实例: ${TEST_EXAMPLES}"
else
    echo "测试实例: (自动发现)"
fi
if [ -n "$FRAMEWORK" ]; then
    echo "框架: ${FRAMEWORK}"
fi
if [ -n "$OUTPUT_FILE" ]; then
    echo "输出文件: ${OUTPUT_FILE}"
fi
echo "========================================================"
echo ""

# 构建命令
CMD="python aggregate_metrics.py --model_dir \"${MODEL_DIR}\""

if [ -n "$TEST_EXAMPLES" ]; then
    CMD="$CMD --test_examples ${TEST_EXAMPLES}"
fi

if [ -n "$FRAMEWORK" ]; then
    CMD="$CMD --framework \"${FRAMEWORK}\""
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"${OUTPUT_FILE}\""
fi

# 执行
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ 聚合完成！${NC}"
else
    echo ""
    echo -e "${RED}❌ 聚合失败 (退出码: $exit_code)${NC}"
fi

exit $exit_code

