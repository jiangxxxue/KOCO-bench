#!/bin/bash
# FlagScale test environment configuration script (modelopt environment)
# modelopt environment already includes required dependencies: megatron-core, transformer-engine, etc.

# Activate modelopt environment
source ~/miniconda3/bin/activate modelopt

# Set FlagScale code path
FLAGSCALE_CODE_DIR="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code"

# Add to PYTHONPATH
export PYTHONPATH="${FLAGSCALE_CODE_DIR}:${PYTHONPATH}"

echo "✓ FlagScale test environment configured (modelopt)"
echo "  - Python: $(python --version)"
echo "  - PYTHONPATH: ${FLAGSCALE_CODE_DIR}"
echo "  - megatron-core: $(python -c 'import megatron.core; print(megatron.core.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""
echo "Now you can run tests:"
echo "  python test_model_provider.py"
echo "  python test_loss_func.py"
echo "  python test_forward_step.py"
