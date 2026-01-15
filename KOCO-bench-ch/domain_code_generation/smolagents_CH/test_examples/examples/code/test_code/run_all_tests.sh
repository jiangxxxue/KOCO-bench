#!/bin/bash
# This script runs all Python unit tests in the current directory.

# Activate the conda environment
conda activate smolagents-test

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'smolagents-test'."
    echo "Please make sure conda is installed and the environment exists."
    exit 1
fi

# Run the test runner Python script
python "$(dirname "$0")/run_all_tests.py"

# Deactivate the environment
conda deactivate