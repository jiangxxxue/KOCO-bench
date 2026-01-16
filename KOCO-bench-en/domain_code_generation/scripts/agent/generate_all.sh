#!/bin/bash
# One-Click Script to Generate All Prompt Files with Keywords
# This script generates complete *_with_prompts_v2.jsonl files for all frameworks

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR"
DATASET_DIR="$SOURCE_DIR/dataset"
DATA_DIR="$SOURCE_DIR/data"
OUTPUT_DIR="$SOURCE_DIR/output"
PROMPT_SCRIPT_DIR="$SOURCE_DIR/prompt"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# Helper Functions
# ============================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================
# Step 0: Check Prerequisites
# ============================================================

check_prerequisites() {
    print_header "Step 0: Checking Prerequisites"

    local all_ok=true

    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python 3 found: $(python3 --version)"
    else
        print_error "Python 3 not found"
        all_ok=false
    fi

    # Check required directories
    if [ -d "$DATASET_DIR" ]; then
        print_success "Dataset directory found: $DATASET_DIR"
    else
        print_error "Dataset directory not found: $DATASET_DIR"
        all_ok=false
    fi

    if [ -d "$PROMPT_SCRIPT_DIR" ]; then
        print_success "Prompt scripts directory found: $PROMPT_SCRIPT_DIR"
    else
        print_error "Prompt scripts directory not found: $PROMPT_SCRIPT_DIR"
        all_ok=false
    fi

    # Check required scripts
    if [ -f "$PROMPT_SCRIPT_DIR/parse_algorithm_methods.py" ]; then
        print_success "parse_algorithm_methods.py found"
    else
        print_error "parse_algorithm_methods.py not found"
        all_ok=false
    fi

    if [ -f "$PROMPT_SCRIPT_DIR/prompts_construction.py" ]; then
        print_success "prompts_construction.py found"
    else
        print_error "prompts_construction.py not found"
        all_ok=false
    fi

    if [ "$all_ok" = false ]; then
        print_error "Prerequisites check failed. Please fix the issues above."
        exit 1
    fi

    print_success "All prerequisites satisfied!"
}

# ============================================================
# Step 1: Create Output Directories
# ============================================================

create_directories() {
    print_header "Step 1: Creating Output Directories"

    mkdir -p "$DATA_DIR"
    mkdir -p "$OUTPUT_DIR"

    print_success "Created: $DATA_DIR"
    print_success "Created: $OUTPUT_DIR"
}

# ============================================================
# Step 2: Process Single Test Example
# ============================================================

process_test_example() {
    local framework=$1
    local test_example=$2

    echo "  Processing: $framework / $test_example"

    # Paths
    local md_file="$DATASET_DIR/$framework/test_examples/$test_example/requirements/03_algorithm_and_core_methods.md"
    local code_base="$DATASET_DIR/$framework/test_examples/$test_example/code"
    local metadata="$DATASET_DIR/$framework/knowledge_corpus/metadata.json"
    local data_file="$DATA_DIR/$framework/algorithm_methods_data_$test_example.jsonl"
    local output_file="$OUTPUT_DIR/${test_example}_with_prompts_v2.jsonl"

    # Check if markdown file exists
    if [ ! -f "$md_file" ]; then
        print_warning "Skip: Markdown file not found: $md_file"
        return 1
    fi

    # Create framework data directory
    mkdir -p "$DATA_DIR/$framework"

    # Step 2.1: Parse markdown
    cd "$PROMPT_SCRIPT_DIR"
    python3 parse_algorithm_methods.py \
        --input "$md_file" \
        --output "$data_file" \
        --code-base "$code_base" > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        print_error "Failed to parse markdown for $test_example"
        return 2
    fi

    # Step 2.2: Generate prompts with keywords
    python3 prompts_construction.py \
        --input "$data_file" \
        --metadata "$metadata" \
        --output "$output_file" \
        --base-dir "$DATASET_DIR" \
        --framework "$framework" \
        --test-example "$test_example" > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        print_error "Failed to generate prompts for $test_example"
        return 2
    fi

    # Count records
    local record_count=$(wc -l < "$output_file")
    print_success "$test_example ($record_count records)"

    return 0
}

# ============================================================
# Step 3: Process All Frameworks
# ============================================================

process_all_frameworks() {
    print_header "Step 2: Generating Prompt Files for All Frameworks"

    local total_success=0
    local total_failed=0
    local total_skipped=0

    # Define frameworks and their test examples
    declare -A frameworks

    # Discover frameworks automatically
    for framework_dir in "$DATASET_DIR"/*; do
        if [ -d "$framework_dir/test_examples" ]; then
            framework=$(basename "$framework_dir")

            echo ""
            echo "Framework: $framework"
            echo "----------------------------------------"

            # Process each test example
            for test_dir in "$framework_dir/test_examples"/*; do
                if [ -d "$test_dir" ]; then
                    test_example=$(basename "$test_dir")

                    process_test_example "$framework" "$test_example"
                    result=$?

                    if [ $result -eq 0 ]; then
                        ((total_success++))
                    elif [ $result -eq 1 ]; then
                        ((total_skipped++))
                    else
                        ((total_failed++))
                    fi
                fi
            done
        fi
    done

    echo ""
    echo "Summary:"
    echo "  Success: $total_success"
    echo "  Skipped: $total_skipped"
    echo "  Failed: $total_failed"

    return $total_failed
}

# ============================================================
# Step 4: Verify Generated Files
# ============================================================

verify_files() {
    print_header "Step 3: Verifying Generated Files"

    cd "$SOURCE_DIR"

    if [ -f "verify_prompts.py" ]; then
        python3 verify_prompts.py

        if [ $? -eq 0 ]; then
            print_success "All files verified successfully!"
            return 0
        else
            print_error "Verification failed"
            return 1
        fi
    else
        print_warning "verify_prompts.py not found, skipping verification"
        return 0
    fi
}

# ============================================================
# Step 5: Generate Summary Report
# ============================================================

generate_report() {
    print_header "Step 4: Generation Complete - Summary Report"

    local total_files=$(ls "$OUTPUT_DIR"/*_with_prompts_v2.jsonl 2>/dev/null | wc -l)
    local total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    local total_records=$(cat "$OUTPUT_DIR"/*_with_prompts_v2.jsonl 2>/dev/null | wc -l)

    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    echo "Statistics:"
    echo "  Total Files: $total_files"
    echo "  Total Size: $total_size"
    echo "  Total Records: $total_records"
    echo ""

    if [ $total_files -gt 0 ]; then
        echo "Generated Files:"
        ls -lh "$OUTPUT_DIR"/*_with_prompts_v2.jsonl | awk '{print "  " $9 " (" $5 ")"}'
    fi

    echo ""
    print_success "All prompt files have been generated successfully!"
    echo ""
    echo "Next Steps:"
    echo "  1. Verify files: python3 verify_prompts.py"
    echo "  2. Use with Claude Code: cd claude_code && python3 incremental_runner.py --data-dir ../output"
    echo ""
}

# ============================================================
# Main Execution
# ============================================================

main() {
    print_header "One-Click Prompt Generation Script"
    echo "This script will generate all *_with_prompts_v2.jsonl files"
    echo "Source Directory: $SOURCE_DIR"
    echo ""

    # Run all steps
    check_prerequisites
    create_directories
    process_all_frameworks

    local generation_result=$?

    verify_files
    generate_report

    if [ $generation_result -eq 0 ]; then
        print_header "✓ SUCCESS - All Files Generated!"
        exit 0
    else
        print_header "⚠ COMPLETED WITH ERRORS"
        echo "Some files failed to generate. Check the output above for details."
        exit 1
    fi
}

# Run main function
main "$@"
