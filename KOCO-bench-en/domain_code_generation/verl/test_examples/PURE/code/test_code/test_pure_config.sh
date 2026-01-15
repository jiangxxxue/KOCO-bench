#!/usr/bin/env bash
# PURE configuration test script - verify paths and environment are correct

set -xeuo pipefail

echo "=== PURE Configuration Verification Test ==="

# Environment variable check
echo "--- Environment Variable Check ---"
echo "HF_HOME: ${HF_HOME:-not set}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-not set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"

# Model check
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
PRM_MODEL_PATH=jinachris/PURE-PRM-7B

echo "--- Model Cache Check ---"
if [ -d "/workspace/data/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" ]; then
    echo "✓ Qwen2.5-0.5B-Instruct model is cached"
else
    echo "✗ Qwen2.5-0.5B-Instruct model is not cached"
fi

if [ -d "/workspace/data/huggingface_cache/hub/models--jinachris--PURE-PRM-7B" ]; then
    echo "✓ PURE-PRM-7B model is cached"
else
    echo "✗ PURE-PRM-7B model is not cached"
fi

# Data file check
echo "--- Data File Check ---"
TRAIN_FILES=/workspace/data/math/train.parquet
VAL_FILE1=/workspace/data/aime2024/train.parquet
VAL_FILE2=/workspace/data/math500/test.parquet

if [ -f "${TRAIN_FILES}" ]; then
    echo "✓ Training data file exists: ${TRAIN_FILES}"
    echo "  File size: $(du -h "${TRAIN_FILES}" | cut -f1)"
else
    echo "✗ Training data file does not exist: ${TRAIN_FILES}"
fi

if [ -f "${VAL_FILE1}" ]; then
    echo "✓ Validation data file 1 exists: ${VAL_FILE1}"
    echo "  File size: $(du -h "${VAL_FILE1}" | cut -f1)"
else
    echo "✗ Validation data file 1 does not exist: ${VAL_FILE1}"
fi

if [ -f "${VAL_FILE2}" ]; then
    echo "✓ Validation data file 2 exists: ${VAL_FILE2}"
    echo "  File size: $(du -h "${VAL_FILE2}" | cut -f1)"
else
    echo "✗ Validation data file 2 does not exist: ${VAL_FILE2}"
fi

# PURE code directory check
echo "--- PURE Code Check ---"
PURE_CODE_DIR=/KOCO-bench/KOCO-bench-en/domain_code_generation/verl/test_examples/PURE/code
if [ -d "${PURE_CODE_DIR}" ]; then
    echo "✓ PURE code directory exists: ${PURE_CODE_DIR}"
    
    if [ -f "${PURE_CODE_DIR}/verl/trainer/main_ppo.py" ]; then
        echo "✓ main_ppo.py exists"
    else
        echo "✗ main_ppo.py does not exist"
    fi
    
    if [ -f "${PURE_CODE_DIR}/verl/trainer/config/ppo_trainer.yaml" ]; then
        echo "✓ ppo_trainer.yaml config file exists"
    else
        echo "✗ ppo_trainer.yaml config file does not exist"
    fi
else
    echo "✗ PURE code directory does not exist: ${PURE_CODE_DIR}"
fi

# Python environment check
echo "--- Python Environment Check ---"
cd "${PURE_CODE_DIR}"

if python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"; then
    echo "✓ PyTorch environment normal"
else
    echo "✗ PyTorch environment abnormal"
fi

if python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"; then
    echo "✓ Transformers available"
else
    echo "✗ Transformers not available"
fi

# Try importing verl module
if python3 -c "from verl.trainer import main_ppo; print('✓ verl.trainer.main_ppo can be imported')"; then
    echo "✓ VERL module normal"
else
    echo "✗ VERL module import failed"
fi

echo "--- Configuration Verification Complete ---"
echo ""
echo "If all checks pass, you can run the full integration test:"
echo "./run_pure_integration.sh"
echo ""
echo "If any checks fail, please resolve the corresponding issues first."
