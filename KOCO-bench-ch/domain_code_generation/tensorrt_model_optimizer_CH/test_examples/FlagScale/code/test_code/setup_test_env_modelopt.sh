#!/bin/bash
# FlagScale 测试环境配置脚本（modelopt 环境）
# modelopt 环境已包含所需依赖：megatron-core, transformer-engine 等

# 激活 modelopt 环境
source ~/miniconda3/bin/activate modelopt

# 设置 FlagScale 代码路径
FLAGSCALE_CODE_DIR="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code"

# 添加到 PYTHONPATH
export PYTHONPATH="${FLAGSCALE_CODE_DIR}:${PYTHONPATH}"

echo "✓ FlagScale 测试环境已配置（modelopt）"
echo "  - Python: $(python --version)"
echo "  - PYTHONPATH: ${FLAGSCALE_CODE_DIR}"
echo "  - megatron-core: $(python -c 'import megatron.core; print(megatron.core.__version__)' 2>/dev/null || echo '未安装')"
echo ""
echo "现在可以运行测试："
echo "  python test_model_provider.py"
echo "  python test_loss_func.py"
echo "  python test_forward_step.py"



