#!/bin/bash
# Step 4 & 5: Docker Execution Evaluation + Aggregate Metrics

# Load common config (.env, SCRIPTS_DIR, PROJECT_ROOT, parse_common_args, etc.)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# help information
show_usage() {
    cat << EOF
Usage: $0 --framework <name> --model <name> [options]

Run Step 4 (Docker execution evaluation) & Step 5 (aggregate metrics)
for already generated code (_output.jsonl files).

Required:
  --framework FRAMEWORK    Framework name (e.g., verl, raganything, smolagents, open-r1)
  --model MODEL            Full model name (e.g., qwen/qwen-2.5-coder-32b-instruct)

Optional:
  --test-example NAME      Run only a single test example (default: all)
  --help                   Show help

Environment Variables:
  DATA_SOURCE              data or rag (default: data)

Examples:
  bash $0 --framework verl --model qwen/qwen-2.5-coder-32b-instruct
  bash $0 --framework verl --model deepseek/deepseek-v3.2 --test-example prime
  DATA_SOURCE=rag bash $0 --framework verl --model openai/gpt-5-mini
EOF
}


# parse CLI arguments
parse_common_args "$@"

DATA_SOURCE="${DATA_SOURCE:-data}"
# NOTE: validate_required_params is handled by the sub-scripts (Step 4)

# derived paths (needed for Step 5 aggregate output path)
MODEL_DIR_NAME=$(basename "${MODEL_NAME}")
DATA_DIR="${PROJECT_ROOT}/scripts/${DATA_SOURCE}/${FRAMEWORK}/${MODEL_DIR_NAME}"


# Banner
echo "Step 4 & 5: Docker Execution Evaluation + Aggregate Metrics"
echo "Framework:    ${FRAMEWORK}"
echo "Model:        ${MODEL_NAME}"
echo "Data source:  ${DATA_SOURCE}"
echo "Data dir:     ${DATA_DIR}"
if [ -n "${TEST_EXAMPLE:-}" ]; then
    echo "Test example: ${TEST_EXAMPLE}"
else
    echo "Test example: all (auto-discover)"
fi
echo "============================================================"
echo ""

# NOTE: Docker pre-flight checks (daemon running, image exists) are handled by the sub-scripts (run_execution_evaluation_pure.sh)


# Step 4: Execution Evaluation (Docker)
START_TIME=$(date +%s)

echo ">>> Step 4: Batch execution evaluation (Docker)"
echo ""

if [ -n "${TEST_EXAMPLE:-}" ]; then
    # Single test example -> call run_execution_evaluation_pure.sh directly
    FRAMEWORK="$FRAMEWORK" \
    MODEL_NAME="$MODEL_NAME" \
    DATA_SOURCE="$DATA_SOURCE" \
    TEST_EXAMPLE="$TEST_EXAMPLE" \
        bash "$SCRIPTS_DIR/run_execution_evaluation_pure.sh"
else
    # All test examples -> call run_batch_execution_evaluation_pure.sh
    DATA_SOURCE="$DATA_SOURCE" \
        bash "$SCRIPTS_DIR/run_batch_execution_evaluation_pure.sh" \
            --framework "$FRAMEWORK" \
            --model "$MODEL_NAME"
fi

STEP4_STATUS=$?
STEP4_END_TIME=$(date +%s)
STEP4_DURATION=$((STEP4_END_TIME - START_TIME))

echo ""
if [ $STEP4_STATUS -eq 0 ]; then
    echo "✅ Step 4 complete (${STEP4_DURATION}s)"
else
    echo "⚠️  Step 4 finished with errors (${STEP4_DURATION}s), continuing to Step 5..."
fi
echo ""


# Step 5: Aggregate Metrics
echo ">>> Step 5: Aggregate metrics"
echo ""

AGGREGATE_OUTPUT="${DATA_DIR}/aggregate_result.json"

python3 "$SCRIPTS_DIR/aggregate_metrics.py" \
    --model_dir "$DATA_DIR" \
    --framework "$FRAMEWORK" \
    --output "$AGGREGATE_OUTPUT"

STEP5_STATUS=$?

if [ $STEP5_STATUS -eq 0 ]; then
    echo ""
    echo "✅ Aggregate metrics saved to: $AGGREGATE_OUTPUT"
else
    echo ""
    echo "❌ Step 5 failed (exit code: $STEP5_STATUS)"
fi


# final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "🏁 Step 4 & 5 complete"
echo "============================================================"
echo "Step 4 (Docker eval):  ${STEP4_DURATION}s  |  $( [ $STEP4_STATUS -eq 0 ] && echo 'OK' || echo 'PARTIAL FAILURE' )"
echo "Step 5 (Aggregate):    $( [ $STEP5_STATUS -eq 0 ] && echo 'OK' || echo 'FAILED' )"
echo "Total duration:        ${TOTAL_DURATION}s"
echo "============================================================"
echo ""

if [ $STEP4_STATUS -ne 0 ] || [ $STEP5_STATUS -ne 0 ]; then
    echo "⚠️  Some steps had failures. Check the output above."
    exit 1
else
    echo "🎉 All steps completed successfully!"
    exit 0
fi
