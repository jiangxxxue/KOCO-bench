#!/bin/bash
# FlagScale test environment configuration script
# For running tests in exper environment

# Activate exper environment
source ~/miniconda3/bin/activate exper

# Set FlagScale code path
FLAGSCALE_CODE_DIR="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code"

# Add to PYTHONPATH
export PYTHONPATH="${FLAGSCALE_CODE_DIR}:${PYTHONPATH}"

# Set CUDA related environment variables (if needed)
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Disable transformer_engine (avoid CUDA library dependency issues)
# By setting environment variable to make megatron-core skip transformer_engine
export NVTE_DISABLE_IMPORT=1

echo "✓ FlagScale test environment configured"
echo "  - Python: $(python --version)"
echo "  - PYTHONPATH: ${FLAGSCALE_CODE_DIR}"
echo ""
echo "Now you can run tests:"
echo "  python test_model_provider.py"
echo "  python test_loss_func.py"
echo "  python test_forward_step.py"
