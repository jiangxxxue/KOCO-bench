#!/bin/bash
# 构建框架训练数据集 - 支持多个框架/repo
#
# 使用方法:
#   默认构建 verl 框架:
#     ./scripts/build_verl_training_dataset.sh
#
#   构建其他框架:
#     FRAMEWORK=tensorrt_model_optimizer ./scripts/build_verl_training_dataset.sh
#
#   指定特定的 repo 名称:
#     FRAMEWORK=verl REPO_NAME=custom-repo ./scripts/build_verl_training_dataset.sh
#
#   设置最大文件大小（字节）:
#     MAX_FILE_SIZE=2097152 ./scripts/build_verl_training_dataset.sh

set -e

# ========================================
# 配置变量
# ========================================

# 框架名称
FRAMEWORK="${FRAMEWORK:-verl}"

# Repo 名称（知识库中的目录名）
REPO_NAME="${REPO_NAME:-${FRAMEWORK}-main}"

FRAMEWORK=verl
REPO_NAME=verl-main

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# 源目录：知识库
SOURCE_DIR="${PROJECT_ROOT}/${FRAMEWORK}/knowledge_corpus/${REPO_NAME}"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/scripts/data/${FRAMEWORK}"
OUTPUT_FILE="${OUTPUT_DIR}/${FRAMEWORK}_training_dataset.jsonl"

# 参数
MAX_FILE_SIZE="${MAX_FILE_SIZE:-1048576}"  # 1MB
TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/models/Qwen2.5-Coder-7B-Instruct}"

# ========================================
# 执行构建
# ========================================

echo "========================================================"
echo "构建训练数据集"
echo "========================================================"
echo "框架: ${FRAMEWORK}"
echo "Repo: ${REPO_NAME}"
echo "源目录: ${SOURCE_DIR}"
echo "输出文件: ${OUTPUT_FILE}"
echo "最大文件大小: ${MAX_FILE_SIZE} bytes"
echo "Tokenizer模型: ${TOKENIZER_PATH}"
echo "========================================================"
echo ""

# 检查源目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误: 源目录不存在: $SOURCE_DIR"
    echo ""
    echo "提示: 请确保以下路径存在:"
    echo "  ${PROJECT_ROOT}/${FRAMEWORK}/knowledge_corpus/${REPO_NAME}"
    echo ""
    echo "或者使用环境变量指定其他框架/repo:"
    echo "  FRAMEWORK=your_framework REPO_NAME=your_repo ./scripts/build_verl_training_dataset.sh"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行数据集构建器
cd "${SCRIPT_DIR}/sft"
python3 finetune_dataset_builder.py \
    --source-dir "$SOURCE_DIR" \
    --output-file "$OUTPUT_FILE" \
    --format jsonl \
    --max-file-size "$MAX_FILE_SIZE" \
    --tokenizer-path "$TOKENIZER_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "✅ 数据集构建完成！"
    echo "========================================================"
    echo "数据文件: ${OUTPUT_FILE}"
    echo "统计文件: ${OUTPUT_FILE%.jsonl}.stats.json"
    echo ""
    
    # 显示统计信息
    if [ -f "${OUTPUT_FILE%.jsonl}.stats.json" ]; then
        echo "📊 数据集统计:"
        cat "${OUTPUT_FILE%.jsonl}.stats.json" | python3 -c "
import json, sys
stats = json.load(sys.stdin)
print(f\"  总文件数: {stats['total_files']}")
print(f\"  处理成功: {stats['processed_files']}")
print(f\"  跳过文件: {stats['skipped_files']}")
print(f\"  总字符数: {stats['total_size_chars']:,}")
print(f\"  总行数: {stats['total_lines']:,}")
if 'total_tokens' in stats and stats['total_tokens'] > 0:
    print(f\"  总Token数: {stats['total_tokens']:,}")
    print(f\"  平均每文件Token数: {stats.get('average_tokens_per_file', 0):.1f}")
print(f\"  文件类型分布:\")
for ftype, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
    print(f\"    {ftype}: {count}\")
"
    fi
    
    echo ""
    echo "========================================================"
else
    echo ""
    echo "❌ 数据集构建失败"
    exit 1
fi

