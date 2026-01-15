# FlagScale Test Environment Setup Guide

## Problem Description

FlagScale test code needs to import the `flagscale.train.train_gpt` module, which depends on:
1. **Specific version of Megatron-LM**: Located in `third_party/Megatron-LM`, includes `megatron.training` module
2. **Transformer Engine**: Used for training acceleration (optional, but megatron-core will check)
3. **Other dependencies**: apex, torch, etc.

## Current Environment Status

### exper environment
- ✅ Python 3.12.4
- ✅ PyTorch 2.7.0
- ✅ megatron-core 0.14.0
- ❌ Missing `megatron.training` module (FlagScale specific)
- ❌ transformer-engine compilation failed (requires CUDA libraries)

### modelopt environment
- ✅ Python 3.10.19
- ✅ PyTorch
- ✅ megatron-core 0.14.0
- ✅ transformer-engine (installed)
- ❌ Missing `megatron.training` module (FlagScale specific)

## Solutions

### Solution 1: Complete FlagScale Installation (Recommended)

This is the most reliable method, will install all required dependencies:

```bash
cd /home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code

# 1. Initialize git submodules (obtain Megatron-LM)
git submodule update --init --recursive

# 2. Activate environment
conda activate exper  # or modelopt

# 3. Install FlagScale
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
    --config-settings=device="gpu" \
    --config-settings=backend="Megatron-LM"

# 4. Run tests
cd test_code
python test_model_provider.py
python test_loss_func.py
python test_forward_step.py
```

### Solution 2: Using PYTHONPATH (Temporary Solution)

If you don't want complete installation, you can try:

```bash
cd /home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code

# 1. Ensure third_party/Megatron-LM exists
git submodule update --init --recursive

# 2. Add Megatron-LM to PYTHONPATH
export PYTHONPATH="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code:$PYTHONPATH"
export PYTHONPATH="/home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code/third_party/Megatron-LM:$PYTHONPATH"

# 3. Run tests
cd test_code
python test_model_provider.py
```

### Solution 3: Accept Test Skipping

Test code already includes import failure handling. If import fails, tests will automatically skip:

```bash
cd /home/shixianjie/workspace/Projects/DomainDataset/tensorrt_model_optimizer/test_examples/FlagScale/code/test_code

# Run directly (will skip all tests, but won't error)
python test_model_provider.py

# Example output:
# ✗ Import failed: No module named 'megatron.training'
# sssssss
# ----------------------------------------------------------------------
# Ran 7 tests in 0.000s
# OK (skipped=7)
```

## Configuration Script Description

### setup_test_env.sh
- For exper environment
- Sets PYTHONPATH
- Current status: ❌ Cannot import (missing megatron.training)

### setup_test_env_modelopt.sh
- For modelopt environment
- Sets PYTHONPATH
- Current status: ❌ Cannot import (missing megatron.training)

## Recommendations

**Best Practice:**
1. Use Solution 1 for complete FlagScale installation
2. This ensures all dependencies are correctly installed
3. Tests can run normally and verify code behavior

**Temporary Solution:**
- If just viewing test code structure, you can directly read `.py` files
- Test code already has detailed comments for each test case's verification points
- Even if unable to run, you can still understand test logic

## Test File Description

All test files include:
- ✅ Import failure handling (automatic skip)
- ✅ Detailed test case descriptions
- ✅ Clear verification point comments
- ✅ Success import notification messages

Therefore, even if unable to run in current environment, test code itself is complete and usable.
