#!/bin/bash
# 评估指标聚合工具使用示例

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "评估指标聚合工具使用示例"
echo "=========================================="
echo ""

# ========================================
# 示例 1: 单个模型的 RL 领域综合指标
# ========================================

echo "示例 1: 计算单个模型在 RL 领域的综合指标"
echo "------------------------------------------"
echo ""

python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-32b-instruct-simple \
  --test_examples prime ARES LUFFY PURE

echo ""
echo "按 Enter 继续下一个示例..."
read

# ========================================
# 示例 2: 保存结果到 JSON 文件
# ========================================

echo ""
echo "示例 2: 保存结果到 JSON 文件"
echo "------------------------------------------"
echo ""

python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-32b-instruct-simple \
  --test_examples prime ARES LUFFY PURE \
  --output data/verl/qwen2.5-coder-32b-instruct-simple/RL_aggregate.json

echo ""
echo "结果已保存到: data/verl/qwen2.5-coder-32b-instruct-simple/RL_aggregate.json"
echo ""
echo "按 Enter 继续下一个示例..."
read

# ========================================
# 示例 3: 批量对比多个模型
# ========================================

echo ""
echo "示例 3: 批量对比多个模型"
echo "------------------------------------------"
echo ""

python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names \
    qwen2.5-coder-32b-instruct-simple \
    qwen2.5-coder-7b-instruct-simple \
    qwen2.5-coder-7b-verl-lora \
  --test_examples prime ARES LUFFY PURE

echo ""
echo "按 Enter 继续下一个示例..."
read

# ========================================
# 示例 4: 导出为 CSV 文件
# ========================================

echo ""
echo "示例 4: 导出对比结果为 CSV"
echo "------------------------------------------"
echo ""

python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names \
    qwen2.5-coder-32b-instruct-simple \
    qwen2.5-coder-7b-instruct-simple \
    qwen2.5-coder-7b-verl-lora \
  --test_examples prime ARES LUFFY PURE \
  --output_csv data/verl/RL_model_comparison.csv

echo ""
echo "CSV 文件已保存到: data/verl/RL_model_comparison.csv"
echo ""
echo "查看 CSV 内容:"
echo "------------------------------------------"
cat data/verl/RL_model_comparison.csv | column -t -s ','
echo ""

# ========================================
# 示例 5: 使用 Shell 脚本
# ========================================

echo ""
echo "示例 5: 使用 Shell 脚本"
echo "------------------------------------------"
echo ""

bash run_aggregate_metrics.sh \
  --model_dir data/verl/qwen2.5-coder-7b-instruct-simple \
  --test_examples "prime ARES LUFFY PURE"

echo ""
echo "=========================================="
echo "所有示例完成！"
echo "=========================================="
echo ""
echo "更多信息请查看: AGGREGATE_METRICS_README.md"

