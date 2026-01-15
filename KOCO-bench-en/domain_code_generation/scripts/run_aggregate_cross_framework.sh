#!/bin/bash
# 跨框架聚合评估指标

set -eo pipefail

cd "$(dirname "$0")"

# ========================================
# 配置
# ========================================

# 默认参数（可通过环境变量覆盖）
MODEL_NAME="${MODEL_NAME:-qwen2.5-coder-7b-modelopt-sft}"
DATA_DIR="${DATA_DIR:-data}"
FRAMEWORKS="${FRAMEWORKS:-}"
OUTPUT_FILE="${OUTPUT_FILE:-data-cross_framework_result.json}"
OUTPUT_CSV="${OUTPUT_CSV:-data-cross_framework_result.csv}"

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
用法: bash run_aggregate_cross_framework.sh [选项]

跨框架聚合评估指标，计算多个框架的综合 pass@1 和 avg_pass_ratio

选项:
  --model_name NAME        模型名称（必需）
  --data_dir DIR           数据目录路径（必需，如 scripts/data）
  --frameworks NAMES       框架名称列表，空格分隔（可选，不指定则自动发现）
  --output FILE            输出 JSON 文件路径（可选）
  --output_csv FILE        输出 CSV 文件路径（可选）
  -h, --help               显示此帮助信息

环境变量:
  MODEL_NAME               模型名称
  DATA_DIR                 数据目录路径
  FRAMEWORKS               框架名称列表
  OUTPUT_FILE              输出 JSON 文件路径
  OUTPUT_CSV               输出 CSV 文件路径

示例:

  # 自动发现所有框架（推荐）
  bash run_aggregate_cross_framework.sh \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data

  # 指定特定框架
  bash run_aggregate_cross_framework.sh \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data \\
    --frameworks "verl open-r1 smolagents"

  # 使用环境变量
  export MODEL_NAME="qwen2.5-coder-7b-instruct"
  export DATA_DIR="scripts/data"
  bash run_aggregate_cross_framework.sh

  # 保存结果到文件
  bash run_aggregate_cross_framework.sh \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data \\
    --output result.json \\
    --output_csv result.csv

输出说明:
  - pass@1: 所有框架所有实例的加权平均 pass@1
  - avg_pass_ratio: 所有框架所有实例的加权平均 avg_pass_ratio

EOF
}

# ========================================
# 参数解析
# ========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --frameworks)
            FRAMEWORKS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --output_csv)
            OUTPUT_CSV="$2"
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

if [ -z "$MODEL_NAME" ]; then
    echo -e "${RED}❌ 错误: 必须指定 --model_name${NC}"
    echo "使用 -h 或 --help 查看帮助信息"
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    echo -e "${RED}❌ 错误: 必须指定 --data_dir${NC}"
    echo "使用 -h 或 --help 查看帮助信息"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ 错误: 数据目录不存在: ${DATA_DIR}${NC}"
    exit 1
fi

# ========================================
# 检查 Python 脚本
# ========================================

if [ ! -f "aggregate_cross_framework.py" ]; then
    echo -e "${RED}❌ 错误: 找不到 aggregate_cross_framework.py${NC}"
    exit 1
fi

# ========================================
# 执行聚合
# ========================================

echo "========================================================"
echo -e "${BLUE}📊 跨框架聚合评估指标${NC}"
echo "========================================================"
echo "模型名称: ${MODEL_NAME}"
echo "数据目录: ${DATA_DIR}"
if [ -n "$FRAMEWORKS" ]; then
    echo "框架列表: ${FRAMEWORKS}"
else
    echo "框架列表: (自动发现)"
fi
if [ -n "$OUTPUT_FILE" ]; then
    echo "输出 JSON: ${OUTPUT_FILE}"
fi
if [ -n "$OUTPUT_CSV" ]; then
    echo "输出 CSV: ${OUTPUT_CSV}"
fi
echo "========================================================"
echo ""

# 构建命令
CMD="python aggregate_cross_framework.py --model_name \"${MODEL_NAME}\" --data_dir \"${DATA_DIR}\""

if [ -n "$FRAMEWORKS" ]; then
    CMD="$CMD --frameworks ${FRAMEWORKS}"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"${OUTPUT_FILE}\""
fi

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --output_csv \"${OUTPUT_CSV}\""
fi

# 执行
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ 跨框架聚合完成！${NC}"
else
    echo ""
    echo -e "${RED}❌ 聚合失败 (退出码: $exit_code)${NC}"
fi

exit $exit_code

