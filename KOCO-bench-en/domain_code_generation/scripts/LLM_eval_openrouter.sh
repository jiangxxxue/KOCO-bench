#!/bin/bash
# One-click evaluation pipeline

# Load common config (.env, SCRIPTS_DIR, PROJECT_ROOT, parse_common_args, etc.)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# help information
show_usage() {
    echo "Usage: $0 --framework <name> --model <name> [options]"
    echo ""
    echo "Required:"
    echo "  --framework FRAMEWORK  Framework name (e.g., verl, raganything)"
    echo "  --model MODEL          Full model name (e.g., qwen/qwen-2.5-coder-32b-instruct)"
    echo ""
    echo "Optional:"
    echo "  --test-example NAME    Specify a single test example (default: process all)"
    echo "  --num-completions N    Number of completions per sample (default: 1)"
    echo "  --help                 Show help"
    echo ""
    echo "Supported models (not limited to):"
    echo "  meta-llama/llama-3.1-8b-instruct"
    echo "  qwen/qwen2.5-coder-7b-instruct"
    echo "  qwen/qwen-2.5-coder-32b-instruct"
    echo "  deepseek/deepseek-chat-v3.1"
    echo "  moonshotai/kimi-k2-0905"
    echo "  google/gemini-2.5-pro"
    echo "  anthropic/claude-sonnet-4.5"
    echo "  openai/gpt-5-mini"
    echo "  openai/o4-mini"
    echo ""
    echo "Examples:"
    echo "  bash $0 --framework verl --model qwen/qwen-2.5-coder-32b-instruct"
    echo "  bash $0 --framework verl --model deepseek/deepseek-chat-v3.1 --test-example prime"
}


# Parse args & validate
parse_common_args "$@"

NUM_COMPLETIONS="${NUM_COMPLETIONS:-1}"
TEST_EXAMPLE="${TEST_EXAMPLE:-}"

validate_required_params

# Model directory name: strip provider prefix (qwen/xxx -> xxx)
MODEL_DIR_NAME="$(basename "${MODEL_NAME}")"


# Banner
echo "============================================================"
echo "One-click evaluation pipeline (OpenRouter API)"
echo "============================================================"
echo "Framework:  ${FRAMEWORK}"
echo "Model:      ${MODEL_NAME}"
if [ -n "$TEST_EXAMPLE" ]; then
    echo "Test example: ${TEST_EXAMPLE}"
else
    echo "Test example: all"
fi
echo "============================================================"
echo ""

# Step 1: Parse algorithm methods
echo ">>> Step 1/5: Parse algorithm methods"
bash "$SCRIPTS_DIR/run_parse_algorithm_methods.sh" \
    --framework "$FRAMEWORK" \
    ${TEST_EXAMPLE:+--test-example "$TEST_EXAMPLE"} || {
    echo ""; echo "❌ Step 1 failed, aborting."; exit 1
}

# Step 2: Construct prompts
echo ">>> Step 2/5: Construct prompts"
bash "$SCRIPTS_DIR/run_prompts_construction.sh" \
    --framework "$FRAMEWORK" \
    ${TEST_EXAMPLE:+--test-example "$TEST_EXAMPLE"} || {
    echo ""; echo "❌ Step 2 failed, aborting."; exit 1
}

# Step 3: Generate code via OpenRouter API
echo ">>> Step 3/5: Generate code via OpenRouter API"
bash "$SCRIPTS_DIR/apicall/run_openrouter.sh" \
    --framework "$FRAMEWORK" \
    --model "$MODEL_NAME" \
    --num-completions "$NUM_COMPLETIONS" \
    ${TEST_EXAMPLE:+--test-example "$TEST_EXAMPLE"} || {
    echo ""; echo "❌ Step 3 failed, aborting."; exit 1
}

# Step 4: Batch execution evaluation
echo ">>> Step 4/5: Batch execution evaluation"
bash "$SCRIPTS_DIR/run_batch_execution_evaluation_pure.sh" \
    --framework "$FRAMEWORK" \
    --model "$MODEL_NAME" || {
    echo ""; echo "❌ Step 4 failed, aborting."; exit 1
}

# Step 5: Aggregate metrics
echo ">>> Step 5/5: Aggregate metrics"
cd "$SCRIPTS_DIR"
python3 "$SCRIPTS_DIR/aggregate_metrics.py" \
    --model_dir "$SCRIPTS_DIR/data/$FRAMEWORK/$MODEL_DIR_NAME" \
    --framework "$FRAMEWORK" || {
    echo ""; echo "❌ Step 5 failed, aborting."; exit 1
}

# Done
echo ""
echo "============================================================"
echo "All 5 steps completed successfully!"
echo "============================================================"
echo "Framework:  ${FRAMEWORK}"
echo "Model:      ${MODEL_NAME}"
echo "  1. Parse algorithm methods"
echo "  2. Construct prompts"
echo "  3. Generate code via OpenRouter API"
echo "  4. Batch execution evaluation"
echo "  5. Aggregate metrics"
echo "============================================================"
exit 0
