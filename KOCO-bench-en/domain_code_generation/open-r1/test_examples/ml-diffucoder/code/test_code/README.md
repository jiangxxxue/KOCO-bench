# ml-diffucoder Test Code

This directory contains test code for the core algorithm functions of the ml-diffucoder project. These tests are used to verify that the LLM-generated code implementation meets the requirements.

## Test File Descriptions

### 1. test_forward_process.py
Tests the `DiffuGRPOTrainer.forward_process` function
- **Functionality**: Generates three versions of masked sequences for the Coupled Sampling scheme
- **Test Coverage**:
  - Basic output structure validation
  - Version 1: Mask all completion tokens
  - Version 2: Random masking by mask_ratio
  - Version 3: Inverse masking (complementary to Version 2)
  - Weight calculation correctness
  - Random seed reproducibility
  - Edge case handling

### 2. test_selective_log_softmax.py
Tests the `selective_log_softmax` function
- **Functionality**: Efficiently computes weighted log probabilities, avoiding memory overflow
- **Test Coverage**:
  - Basic output shape validation
  - Behavior without weights
  - Weighted average correctness
  - Multiple iteration processing
  - Mask effect verification
  - Memory efficiency testing
  - Numerical stability

### 3. test_get_per_token_logps.py
Tests the `DiffuGRPOTrainer._get_per_token_logps` function
- **Functionality**: Computes token-level log probabilities using the Coupled Sampling scheme
- **Test Coverage**:
  - Input validation (dimensions, mask_seeds length)
  - Output shape correctness
  - prompt_index creation
  - Multiple iteration processing
  - Batch concatenation
  - Completion part extraction
  - Weight propagation
  - Output dimension arrangement
  - Data type conversion

### 4. test_generate_and_score_completions.py
Tests the `DiffuGRPOTrainer._generate_and_score_completions` function
- **Functionality**: Generates code completions and computes rewards
- **Test Coverage**:
  - Basic output structure
  - Prompt processing
  - Masking after EOS token
  - Mask seeds generation (random/fixed)
  - Advantage calculation (leave-one-out baseline)
  - Weighted multiple reward functions
  - Metrics recording

### 5. test_code_reward.py
Tests the `code_reward` function
- **Functionality**: Evaluates generated code and computes test case pass rate
- **Test Coverage**:
  - Basic output structure
  - Format checking
  - Code extraction
  - Test case execution
  - Partial pass scenarios
  - Different executor provider types
  - Parallel execution
  - Evaluation script templates
  - Edge cases

### 6. test_code_format_reward.py
Tests the `get_code_format_reward` function
- **Functionality**: Checks code format and syntax
- **Test Coverage**:
  - Correctly formatted code (returns 1.0)
  - Missing code block markers (returns 0.0)
  - Syntax errors (returns 0.5)
  - Extra text handling
  - Multiple code blocks
  - Empty code blocks
  - Complex code
  - Various syntax errors
  - Different programming language support

## Running Tests

### Run All Tests
```bash
cd open-r1/test_examples/ml-diffucoder/code/test_code
./run_all_tests.sh
```

### Run Individual Test Files
```bash
python3 test_forward_process.py
python3 test_selective_log_softmax.py
python3 test_get_per_token_logps.py
python3 test_generate_and_score_completions.py
python3 test_code_reward.py
python3 test_code_format_reward.py
```

### Run Specific Test Class or Method
```bash
# Run specific test class
python3 test_forward_process.py TestForwardProcess

# Run specific test method
python3 test_forward_process.py TestForwardProcess.test_forward_process_basic_output_structure
```

## Test Design Principles

1. **Direct Import Testing**: Test code directly imports functions from source code rather than copying implementations
2. **Comprehensive Coverage**: Includes normal cases, edge cases, and error cases
3. **Independence**: Each test runs independently without depending on other tests
4. **Reproducibility**: Uses fixed random seeds to ensure reproducible results
5. **Clear Assertions**: Each assertion has a clear error message

## Test Data

Tests use simulated data and mock objects without depending on external resources:
- Uses `torch.manual_seed()` to ensure reproducibility
- Uses `unittest.mock` to simulate complex dependencies
- Uses small-scale data to ensure tests run quickly

## Expected Results

All tests should pass. If tests fail, it indicates:
1. The LLM-generated code implementation is incorrect
2. The code does not meet specification requirements
3. Edge cases are not handled

## Notes

1. Test code assumes source code is located in the `../src/open_r1/` directory
2. Necessary dependencies must be installed: `torch`, `transformers`, `trl`, etc.
3. Some tests use mock objects to simulate complex dependencies, which may need adjustment during actual execution
4. Tests primarily focus on function input/output behavior rather than internal implementation details

## Extending Tests

If you need to add new tests:
1. Add new test methods to the corresponding test file
2. Follow existing naming conventions: `test_<function_name>_<test_aspect>`
3. Add clear docstrings explaining the test purpose
4. Update `run_all_tests.sh` if adding new test files
