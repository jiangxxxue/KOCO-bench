# CANOE Algorithm Core Function Descriptions

# FUNCTION: accuracy_reward

## Function Overview
Validates if model-generated answer is correct, supporting both symbolic verification and exact string matching. First attempts symbolic verification (using latex2sympy and math_verify), falls back to string matching if that fails.

## Function Signature
def accuracy_reward(completions, solution, **kwargs):


## Input Parameters
- `completions`: List of model generation results, each element is dictionary list containing `{"role": "assistant", "content": "..."}`
- `solution`: List of correct answers, each element is string possibly containing `<answer>` tags
- `**kwargs`: Additional parameters (not used)

## Detailed Description

This reward function adopts a two-layer verification strategy to judge if generated mathematical answer is correct. First attempts symbolic verification, parsing both model-generated answer and ground truth answer as symbolic expressions, then verifying their mathematical equivalence through symbolic computation; if verification succeeds, directly rewards 1.0. If symbolic verification fails, extracts content wrapped by <answer> tags from ground truth and generated answer (removing leading/trailing whitespace) for exact string comparison; if tags not found, uses complete string. If completely identical, rewards 1.0. In debug mode, writes verification information for each attempt to log file, recording timestamp, reward value, generated content, and correct answer. Log filename is `accuracy_reward.log`.

## Output
- `rewards`: Float list, each element is 0.0 or 1.0

## Function Implementation
code/train/src/open_r1/grpo.py:line 59-122

## Test Code
code/test_code/test_accuracy_reward.py

---

# FUNCTION: influence_reward

## Function Overview
Evaluates influence of long answer on final answer. By replacing original context with long answer and regenerating, checks if newly generated answer is still correct.

## Function Signature
def influence_reward(completions, solution, completions_long_answer, **kwargs):


## Input Parameters
- `completions`: List of model generation results, each element is dictionary list containing `{"role": "assistant", "content": "..."}`
- `solution`: List of correct answers
- `completions_long_answer`: List of results regenerated based on long answer, each element may be None
- `**kwargs`: Additional parameters (not used)

## Detailed Description

Function receives additional long answer generation results as input; if long answer is None, reward is 0.0 and skips subsequent verification, otherwise extracts content field. Then extracts answer portion from <answer> tags in long answer and ground truth (removing leading/trailing whitespace) for exact string comparison; if completely identical, rewards 1.0. In debug mode, writes verification information for each attempt to log file, recording timestamp, reward value, generated content, and correct answer. Log filename is `influence_accuracy_reward.log`.


## Output
- `rewards`: Float list, each element is 0.0 or 1.0

## Function Implementation
code/train/src/open_r1/grpo.py:line 124-168

## Test Code
code/test_code/test_influence_reward.py

---

# FUNCTION: format_reward

## Function Overview
Checks if generated content conforms to predefined format requirements, i.e., whether it contains complete `<think>...</think><long_answer>...</long_answer><answer>...</answer>` structure.

## Function Signature
def format_reward(completions, **kwargs):


## Input Parameters
- `completions`: List of model generation results, each element is dictionary list containing `{"role": "assistant", "content": "..."}`
- `**kwargs`: Additional parameters (not used)

## Detailed Description

Checks if generated content conforms to predefined format requirements, i.e., whether it contains complete `<think>...</think><long_answer>...</long_answer><answer>...</answer>` structure; returns 1.0 if match succeeds, 0.0 if fails.

## Output
- `rewards`: Float list, each element is 0.0 or 1.0

## Function Implementation
code/train/src/open_r1/grpo.py:line 171-176

## Test Code
code/test_code/test_format_reward.py

---

# FUNCTION: len_reward

## Function Overview
Constrains long answer length within reasonable range, ensuring its length is between 20%-80% of original context length.

## Function Signature
def len_reward(completions, **kwargs):


## Input Parameters
- `completions`: List of model generation results, each element is dictionary list containing `{"role": "assistant", "content": "..."}`
- `**kwargs`: Must contain `problem` field, i.e., original input (containing context)

## Detailed Description
Checks if <long_answer> length is between 20%-80% of corresponding <context> length. If yes, rewards 1.0; if length doesn't meet this range or cannot extract <context> or <long_answer>, reward is 0.0.

## Output
- `rewards`: Float list, each element is 0.0 or 1.0

## Function Implementation
code/train/src/open_r1/grpo.py:line 178-215

## Test Code
code/test_code/test_len_reward.py


