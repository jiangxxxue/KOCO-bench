# PACS Test Code Documentation

This directory contains test code for PACS project core functions, used to verify whether LLM-generated code meets requirements.

## Test Files

### 1. test_compute_reward.py
Test `compute_reward` function - Policy improvement reward calculation

**Test Content:**
- Method 1: Sum of log probability differences calculation
- Method 2: Normalized log probability sum calculation
- Beta parameter impact
- response_mask application
- Positive and negative policy improvement cases
- Edge cases and batch processing

### 2. test_compute_weight.py
Test `compute_weight` function - Sample weight calculation

**Test Content:**
- question mode: Weight calculation for balanced and imbalanced samples
- only_positive mode: Keep only positive samples
- only_negative mode: Keep only negative samples
- Weight calculation for multiple groups
- Edge cases and batch processing consistency

### 3. test_compute_pacs_loss.py
Test `compute_pacs_loss` function - PACS core loss function

**Test Content:**
- RLOO advantage estimation
- GRPO advantage estimation
- Naive advantage estimation
- Sample weight application
- Impact of different reward_method
- response_mask handling
- Prompt validation
- End-to-end workflow

## Running Tests

### Run All Tests
```bash
cd verl/test_examples/PACS/code/test_code
bash run_pacs_tests.sh
```

### Run Individual Test Files
```bash
python test_compute_reward.py -v
python test_compute_weight.py -v
python test_compute_pacs_loss.py -v
```

### Run Specific Test Cases
```bash
python test_compute_reward.py TestComputeReward.test_compute_reward_method_1_basic -v
```

## Test Design Principles

1. **Direct Import of Original Functions**: Test code directly imports functions from `pacs.pacs_core_algos` instead of copying code
2. **Comprehensive Coverage**: Tests cover basic functionality, edge cases, parameter variations, batch processing, and more
3. **Precise Verification**: Uses manually calculated expected values to verify function output correctness
4. **Independence**: Each test case is independent and can be run separately

## Test Purpose

These test codes are used to:
1. Verify that LLM-generated function implementations are correct
2. Ensure functions work properly under various input conditions
3. Check that function output matches documented behavior

## Notes

- Test code assumes original functions are correctly implemented in `src/pacs/pacs_core_algos.py`
- When evaluating LLM-generated code, the LLM-generated functions need to be replaced in the original file first
- Passing tests indicates that LLM-generated code meets requirements
