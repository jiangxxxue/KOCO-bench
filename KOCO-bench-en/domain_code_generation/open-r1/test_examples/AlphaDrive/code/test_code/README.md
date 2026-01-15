# AlphaDrive Test Code

This directory contains test code for AlphaDrive project core functions, used to verify that LLM-generated code meets requirements.

## Test File Description

### 1. test_plan_speed_reward.py
Tests the speed planning reward function `plan_speed_reward`, verifying:
- F1 score calculation (precision, recall)
- Complexity factor calculation
- Diversity factor calculation
- Reward fusion logic

### 2. test_plan_path_reward.py
Tests the path planning reward function `plan_path_reward`, verifying:
- F1 score calculation (precision, recall)
- Complexity factor calculation
- Diversity factor calculation
- Reward fusion logic

### 3. test_plan_format_reward.py
Tests the format validation reward function `plan_format_reward`, verifying:
- Correct format recognition: `<think>...</think><answer>...</answer>`
- Error format detection
- Edge case handling

### 4. test_get_per_token_logps.py
Tests the per-token log probability calculation function `_get_per_token_logps`, verifying:
- Alignment of logits and input_ids
- log_softmax calculation
- Correctness of gather operation
- Memory-efficient row-by-row processing

### 5. test_compute_loss.py
Tests the GRPO loss calculation function `compute_loss`, verifying:
- Multimodal input processing
- Prompt and completion separation
- EOS mask creation
- KL divergence calculation
- Reward aggregation
- GRPO advantage estimation
- Policy gradient loss calculation
- Masked average loss

## Running Tests

### Run All Tests
```bash
cd test_code
./run_tests.sh
```

### Run Individual Tests
```bash
python test_plan_speed_reward.py
python test_plan_path_reward.py
python test_plan_format_reward.py
python test_get_per_token_logps.py
python test_compute_loss.py
```

### Run Specific Test Cases
```bash
python test_plan_speed_reward.py TestPlanSpeedReward.test_plan_speed_reward_perfect_match
```

## Test Design Principles

1. **Direct Import of Original Functions**: Test code directly imports functions from the original code rather than copying them
2. **Standard Answer Verification**: Construct accurate test cases based on algorithm descriptions in documentation
3. **Cover Multiple Scenarios**:
   - Basic functionality tests
   - Edge case tests
   - Integration tests
   - Error handling tests

## Test Case Structure

Each test file contains:
- **Basic Test Class**: Tests core functionality
- **Integration Test Class**: Tests integration with other components
- **Edge Case Test Class** (if applicable): Tests special cases

## Notes

- Test code must be able to successfully import functions from the original code
- Test cases are written based on function descriptions in `03_algorithm_and_core_methods.md`
- The purpose of testing is to verify that LLM-generated code correctly implements the algorithm logic
