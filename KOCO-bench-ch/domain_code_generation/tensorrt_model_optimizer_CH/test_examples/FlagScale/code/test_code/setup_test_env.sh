#!/bin/bash
# FlagScale 测试环境配置脚本
# 用于在 exper 环境下运行测试

# 激活 exper 环境
source ~/miniconda3/bin/activate exper

# 设置 FlagScale 代码路径
FLAGSCALE_CODE_DIR="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code"

# 添加到 PYTHONPATH
export PYTHONPATH="${FLAGSCALE_CODE_DIR}:${PYTHONPATH}"

# 设置 CUDA 相关环境变量（如果需要）
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 禁用 transformer_engine（避免 CUDA 库依赖问题）
# 通过设置环境变量让 megatron-core 跳过 transformer_engine
export NVTE_DISABLE_IMPORT=1

echo "✓ FlagScale 测试环境已配置"
echo "  - Python: $(python --version)"
echo "  - PYTHONPATH: ${FLAGSCALE_CODE_DIR}"
echo ""
echo "现在可以运行测试："
echo "  python test_model_provider.py"
echo "  python test_loss_func.py"
echo "  python test_forward_step.py"



