#!/usr/bin/env bash
# PURE 配置测试脚本 - 验证路径和环境是否正确

set -xeuo pipefail

echo "=== PURE 配置验证测试 ==="

# 环境变量检查
echo "--- 环境变量检查 ---"
echo "HF_HOME: ${HF_HOME:-not set}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-not set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"

# 模型检查
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
PRM_MODEL_PATH=jinachris/PURE-PRM-7B

echo "--- 模型缓存检查 ---"
if [ -d "/workspace/data/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" ]; then
    echo "✓ Qwen2.5-0.5B-Instruct 模型已缓存"
else
    echo "✗ Qwen2.5-0.5B-Instruct 模型未缓存"
fi

if [ -d "/workspace/data/huggingface_cache/hub/models--jinachris--PURE-PRM-7B" ]; then
    echo "✓ PURE-PRM-7B 模型已缓存"
else
    echo "✗ PURE-PRM-7B 模型未缓存"
fi

# 数据文件检查
echo "--- 数据文件检查 ---"
TRAIN_FILES=/workspace/data/math/train.parquet
VAL_FILE1=/workspace/data/aime2024/train.parquet
VAL_FILE2=/workspace/data/math500/test.parquet

if [ -f "${TRAIN_FILES}" ]; then
    echo "✓ 训练数据文件存在: ${TRAIN_FILES}"
    echo "  文件大小: $(du -h "${TRAIN_FILES}" | cut -f1)"
else
    echo "✗ 训练数据文件不存在: ${TRAIN_FILES}"
fi

if [ -f "${VAL_FILE1}" ]; then
    echo "✓ 验证数据文件1存在: ${VAL_FILE1}"
    echo "  文件大小: $(du -h "${VAL_FILE1}" | cut -f1)"
else
    echo "✗ 验证数据文件1不存在: ${VAL_FILE1}"
fi

if [ -f "${VAL_FILE2}" ]; then
    echo "✓ 验证数据文件2存在: ${VAL_FILE2}"
    echo "  文件大小: $(du -h "${VAL_FILE2}" | cut -f1)"
else
    echo "✗ 验证数据文件2不存在: ${VAL_FILE2}"
fi

# PURE代码目录检查
echo "--- PURE代码检查 ---"
PURE_CODE_DIR=/KOCO-bench/KOCO-bench-ch/domain_code_generation/verl/test_examples/PURE/code
if [ -d "${PURE_CODE_DIR}" ]; then
    echo "✓ PURE代码目录存在: ${PURE_CODE_DIR}"
    
    if [ -f "${PURE_CODE_DIR}/verl/trainer/main_ppo.py" ]; then
        echo "✓ main_ppo.py 存在"
    else
        echo "✗ main_ppo.py 不存在"
    fi
    
    if [ -f "${PURE_CODE_DIR}/verl/trainer/config/ppo_trainer.yaml" ]; then
        echo "✓ ppo_trainer.yaml 配置文件存在"
    else
        echo "✗ ppo_trainer.yaml 配置文件不存在"
    fi
else
    echo "✗ PURE代码目录不存在: ${PURE_CODE_DIR}"
fi

# Python环境检查
echo "--- Python环境检查 ---"
cd "${PURE_CODE_DIR}"

if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"; then
    echo "✓ PyTorch环境正常"
else
    echo "✗ PyTorch环境异常"
fi

if python3 -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"; then
    echo "✓ Transformers可用"
else
    echo "✗ Transformers不可用"
fi

# 尝试导入verl模块
if python3 -c "from verl.trainer import main_ppo; print('✓ verl.trainer.main_ppo 可导入')"; then
    echo "✓ VERL模块正常"
else
    echo "✗ VERL模块导入失败"
fi

echo "--- 配置验证完成 ---"
echo ""
echo "如果所有检查都通过，可以运行完整的集成测试："
echo "./run_pure_integration.sh"
echo ""
echo "如果有任何检查失败，请先解决相应问题。"
