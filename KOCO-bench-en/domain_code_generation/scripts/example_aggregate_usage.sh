#!/bin/bash
# Example usage of metrics aggregation tools

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Metrics Aggregation Tools - Example Usage"
echo "=========================================="
echo ""

# ========================================
# Example 1: Single model comprehensive metrics
# ========================================

echo "Example 1: Calculate comprehensive metrics for a single model"
echo "------------------------------------------"
echo ""

python aggregate_metrics.py \
  --model_dir data/your_framework/your_model \
  --test_examples example1 example2 example3

echo ""
echo "Press Enter to continue to next example..."
read

# ========================================
# Example 2: Save results to JSON file
# ========================================

echo ""
echo "Example 2: Save results to JSON file"
echo "------------------------------------------"
echo ""

python aggregate_metrics.py \
  --model_dir data/your_framework/your_model \
  --test_examples example1 example2 example3 \
  --output data/your_framework/your_model/aggregate.json

echo ""
echo "Results saved to: data/your_framework/your_model/aggregate.json"
echo ""
echo "Press Enter to continue to next example..."
read

# ========================================
# Example 3: Batch comparison of multiple models
# ========================================

echo ""
echo "Example 3: Batch comparison of multiple models"
echo "------------------------------------------"
echo ""

python batch_aggregate_metrics.py \
  --base_dir data/your_framework \
  --model_names \
    model1 \
    model2 \
    model3 \
  --test_examples example1 example2 example3

echo ""
echo "Press Enter to continue to next example..."
read

# ========================================
# Example 4: Export to CSV file
# ========================================

echo ""
echo "Example 4: Export comparison results as CSV"
echo "------------------------------------------"
echo ""

python batch_aggregate_metrics.py \
  --base_dir data/your_framework \
  --model_names \
    model1 \
    model2 \
    model3 \
  --test_examples example1 example2 example3 \
  --output_csv data/your_framework/model_comparison.csv

echo ""
echo "CSV file saved to: data/your_framework/model_comparison.csv"
echo ""
echo "View CSV content:"
echo "------------------------------------------"
cat data/your_framework/model_comparison.csv | column -t -s ',' 2>/dev/null || cat data/your_framework/model_comparison.csv
echo ""

# ========================================
# Example 5: Using shell script
# ========================================

echo ""
echo "Example 5: Using shell script"
echo "------------------------------------------"
echo ""

bash run_aggregate_metrics.sh \
  --model_dir data/your_framework/your_model \
  --test_examples "example1 example2 example3"

echo ""
echo "=========================================="
echo "All examples completed!"
echo "=========================================="
echo ""
echo "For more information, see: AGGREGATE_METRICS_README.md"

