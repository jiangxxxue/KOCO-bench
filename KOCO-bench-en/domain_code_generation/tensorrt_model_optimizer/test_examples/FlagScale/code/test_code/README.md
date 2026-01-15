# FlagScale Test Code Documentation

## Overview

This directory contains unit test code for FlagScale core functions, written based on source code implementation to verify function behavior consistency with the source code.

## Test Files

### 1. test_model_provider.py
Tests the `model_provider` function (flagscale/train/train_gpt.py:line 69-191)

**Test Cases (7):**
1. Standard model creation flow
2. Legacy model creation
3. YAML configuration loading
4. Configuration from parallel_context
5. Transformer Engine spec selection
6. ModelOpt model provider
7. Model parameter passing integrity

### 2. test_loss_func.py
Tests the `loss_func` function (flagscale/train/train_gpt.py:line 216-275)

**Test Cases (7):**
1. Basic loss calculation
2. view(-1) operation
3. num_tokens calculation
4. reporting_loss structure
5. NaN check enabled
6. Spiky loss check
7. ModelOpt loss_func

### 3. test_forward_step.py
Tests the `forward_step` function (flagscale/train/train_gpt.py:line 278-304)

**Test Cases (6):**
1. Calling get_batch
2. Core model forward call
3. Legacy model forward call
4. Return value structure
5. Usage of partial loss_func
6. stimer context manager

## Environment Requirements

Test code needs to run in a Python environment with necessary dependencies.

### Method 1: Using modelopt environment (Recommended)

The modelopt environment already includes required dependencies (megatron-core, transformer-engine, etc.):

```bash
cd /path/to/FlagScale/code/test_code

# Use configuration script (automatically activates modelopt environment and sets PYTHONPATH)
source setup_test_env_modelopt.sh

# Run tests
python test_model_provider.py
python test_loss_func.py
python test_forward_step.py
```

### Method 2: Using exper environment

If running in exper environment, additional dependencies need to be installed:

```bash
conda activate exper

# Install transformer-engine (requires CUDA support)
pip install transformer-engine[pytorch]

# Or if no CUDA, you can uninstall transformer-engine
# megatron-core will skip transformer-engine related functionality at runtime
pip uninstall transformer-engine transformer-engine_cu12

cd /path/to/FlagScale/code/test_code
source setup_test_env.sh
python test_model_provider.py
```

### Method 3: Complete FlagScale installation

```bash
cd /path/to/FlagScale/code

# Install FlagScale and Megatron-LM backend
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
    --config-settings=device="gpu" \
    --config-settings=backend="Megatron-LM"

# Run tests
cd test_code
python test_model_provider.py
```

## Test Design Principles

1. **Based on source code**: Each test case is written based on actual implementation logic in source code
2. **Verify behavior consistency**: Tests verify that function behavior is consistent with source code description
3. **Mock external dependencies**: Use unittest.mock to simulate external dependencies, focusing on target function testing
4. **Clear output**: Each test prints success message when passing, making it easy to track test progress

## Test Output Example

```
✓ Successfully imported model_provider module

======================================================================
model_provider tests
======================================================================
✓ Test 1 passed: Standard model creation flow
✓ Test 2 passed: Legacy model creation
✓ Test 3 passed: YAML configuration loading
✓ Test 4 passed: Configuration from parallel_context
✓ Test 5 passed: Transformer Engine spec selection
✓ Test 6 passed: ModelOpt model provider
✓ Test 7 passed: Model parameter passing integrity
.......
----------------------------------------------------------------------
Ran 7 tests in 0.123s

OK
```

## Notes

1. **Import failure handling**: If import fails, tests will be skipped, not error out
2. **Environment dependencies**: Tests require complete FlagScale environment, including specific version of Megatron-LM
3. **GPU not required**: Tests use Mock, no actual GPU hardware needed
4. **Megatron-LM version**: FlagScale uses a customized version of Megatron-LM (located in `third_party/Megatron-LM`), incompatible with `megatron-core` on PyPI

## Current Environment Limitations

Since FlagScale depends on a specific Megatron-LM version (requires complete installation to obtain), without complete FlagScale installation, test code will fail to import due to missing `megatron.training` module.

### Solutions

**Recommended: Complete FlagScale installation**

```bash
cd /path/to/FlagScale/code

# 1. Initialize submodules (obtain Megatron-LM)
git submodule update --init --recursive

# 2. Install FlagScale
PYTHONPATH=./:$PYTHONPATH pip install . --no-build-isolation --verbose \
    --config-settings=device="gpu" \
    --config-settings=backend="Megatron-LM"

# 3. Run tests
cd test_code
python test_model_provider.py
python test_loss_func.py
python test_forward_step.py
```

If import still fails, tests will automatically skip without affecting test execution.

## Documentation Reference

Test code corresponding function documentation:
- `requirements/03_algorithm_and_core_methods.md`

Source code location:
- `flagscale/train/train_gpt.py`
