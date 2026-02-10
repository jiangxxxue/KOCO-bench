#!/bin/bash
# Common configuration loader

# Locate the scripts/ directory (where common.sh itself resides)
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-derive project root (parent of scripts/, i.e. domain_code_generation directory)
export PROJECT_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"

# Load .env config file (parse line by line, skip comments and blank lines, only export valid assignments)
_ENV_FILE="$SCRIPTS_DIR/.env"
if [ -f "$_ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip blank lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # Only process lines matching KEY=VALUE
        if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Strip surrounding quotes (single or double)
            value="${value#\"}"
            value="${value%\"}"
            value="${value#\'}"
            value="${value%\'}"
            # Only export if the variable is not already set (preserve existing env var priority)
            if [ -z "${!key+x}" ]; then
                export "$key=$value"
            fi
        fi
    done < "$_ENV_FILE"
else
    echo "⚠️  Warning: .env file not found ($_ENV_FILE)"
    echo "Please copy .env.example to .env and fill in the actual values"
    echo ""
fi
unset _ENV_FILE

# Validation: check FRAMEWORK only (for scripts that don't need MODEL_NAME)
validate_framework() {
    if [ -z "${FRAMEWORK:-}" ]; then
        echo "❌ Error: FRAMEWORK is not set (required parameter)"
        echo ""
        show_usage
        exit 1
    fi
}

# Common validation: check required experiment parameters
validate_required_params() {
    local missing=0
    if [ -z "${FRAMEWORK:-}" ]; then
        echo "❌ Error: FRAMEWORK is not set (required parameter)"
        missing=1
    fi
    if [ -z "${MODEL_NAME:-}" ]; then
        echo "❌ Error: MODEL_NAME is not set (required parameter)"
        missing=1
    fi
    if [ $missing -eq 1 ]; then
        echo ""
        echo "Usage:"
        echo "  bash $0 --framework verl --model qwen/qwen-2.5-coder-32b-instruct"
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
        exit 1
    fi
}

# Parse common CLI arguments into env vars.
# Each script defines its own show_usage() before calling this.
parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --framework)       FRAMEWORK="$2"; shift 2 ;;
            --model)           MODEL_NAME="$2"; shift 2 ;;
            --test-example)    TEST_EXAMPLE="$2"; shift 2 ;;
            --num-completions) NUM_COMPLETIONS="$2"; shift 2 ;;
            --help) show_usage; exit 0 ;;
            *) echo "❌ Unknown argument: $1"; echo ""; show_usage; exit 1 ;;
        esac
    done
}
