# DiffuCoder Algorithm Core Function Descriptions

# FUNCTION: DiffuGRPOTrainer.forward_process

## Function Overview
Generates three versions of masked sequences for Coupled Sampling scheme. Ensures each token is calculated in at least one forward pass through paired masks, reducing variance and improving training efficiency.

## Function Signature
def forward_process(self, batch, prompt_index, mask_id, seed=None, accumulate=False)

## Input Parameters
- `batch`: Input sequence tensor, shape `[batch_size, seq_len]`
- `prompt_index`: Boolean tensor marking which positions are prompt tokens, shape `[seq_len]`
- `mask_id`: Mask token ID (usually `tokenizer.mask_token_id`)
- `seed`: Random seed, used to control mask pattern repeatability
- `accumulate`: Whether to share random matrix across gradient accumulation steps

## Detailed Description
The function generates three masked sequences for diffusion model training. The first version fully masks all generation parts, the second version masks generation tokens by random probability, and the third version masks generation tokens not masked in the second version.

Mask probability is randomly sampled from interval [0.2, 0.8]. The function needs to generate a random matrix with same shape as input sequence to decide whether each token is masked.

In gradient accumulation scenarios, multiple accumulation steps share the same random matrix to maintain mask pattern consistency. Supports controlling mask repeatability through random seed.

## Output
- `noisy_batch`: List of three versions of masked sequences, each with shape `[batch_size, seq_len]`
- `weights`: Weight list `[1, 1/mask_ratio, 1/(1-mask_ratio)]`
- `completion_mask`: Boolean tensor marking which tokens are masked in Version 2, shape `[batch_size, seq_len]`

## Function Implementation
code/src/open_r1/coupled_grpo.py:line 254-279

## Test Code
code/test_code/test_forward_process.py

---

# FUNCTION: selective_log_softmax

## Function Overview
Efficiently calculates weighted log probabilities, avoiding memory overflow. For each sequence, calculates log probabilities of three versions (original, masked, reverse masked), and performs weighted average based on mask status and weights.

## Function Signature
def selective_log_softmax(logits, index, weights=None, mask=None)

## Input Parameters
- `logits`: Model output logits, shape `[num_iterations * 3 * batch_size, seq_len, vocab_size]`
- `index`: Target token IDs, shape `[num_iterations * batch_size, seq_len]`
- `weights`: Weight tensor, shape `[num_iterations * 3]`, used to weight log probabilities of different versions
- `mask`: Mask indicator, shape `[num_iterations * batch_size, seq_len]`, marking which tokens are masked

## Detailed Description

The function calculates weighted log probabilities from three versions of logits. Input logits tensor is organized in iteration blocks, each iteration block contains three consecutive batches, corresponding to three mask versions of all sequences. The function processes sequences one by one, extracts logits of three versions, applies log softmax to vocabulary dimension, then uses gather operation to extract log probabilities at corresponding positions based on target token indices. For each token position, if masked in second version, uses log probability of second version multiplied by second weight; if not masked, uses log probability of third version multiplied by third weight. Adds weighted result to log probability of first version and divides by 2 to get final log probability. Stacks log probabilities of all sequences into tensor and returns.

## Output
- `per_token_logps`: Token-level log probabilities, shape `[num_iterations * batch_size, seq_len]`

## Function Implementation
code/src/open_r1/coupled_grpo.py:line 59-131

## Test Code
code/test_code/test_selective_log_softmax.py

---

# FUNCTION: DiffuGRPOTrainer._get_per_token_logps

## Function Overview
Calculates token-level log probabilities using Coupled Sampling scheme. For each iteration, generates three versions of masked sequences, forward propagates to calculate logits, then uses `selective_log_softmax` to calculate weighted log probabilities.

## Function Signature
def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds)

## Input Parameters
- `model`: Policy model or reference model
- `input_ids`: Input sequences, shape `[num_iterations, batch_size, seq_len]`
- `logits_to_keep`: Number of tokens to calculate logits (usually completion part length)
- `mask_seeds`: Mask seed list, length is `num_iterations`

## Detailed Description

The function determines prompt part position based on sequence length and completion length. For each iteration, uses corresponding mask seed and calls related function to generate three versions of masked sequences, while obtaining coefficients for weighting and mask marking masked positions.

Inputs masked sequences to model to get logits, but only keeps output of completion part at end of sequence. Also extracts corresponding completion part from original input and masking mask as target tokens and masking indicator, combines with logits and weight coefficients, calls related function to calculate weighted log probability at each token position.

Calculation results are reorganized into three-dimensional tensor, dimension order adjusted to batch, iteration, completion length, so that log probabilities of each sample under different iterations are organized together. The function releases memory occupied by intermediate calculations before returning.

## Output
- `per_token_logps`: Token-level log probabilities, shape `[batch_size, num_iterations, logits_to_keep]`

## Function Implementation
code/src/open_r1/coupled_grpo.py:line 290-362

## Test Code
code/test_code/test_get_per_token_logps.py

---

# FUNCTION: DiffuGRPOTrainer._generate_and_score_completions

## Function Overview
Generates code completions and calculates rewards. Uses diffusion generation method to generate k responses, then calls reward functions to calculate reward score for each response, finally calculates leave-one-out advantage.

## Function Signature
def _generate_and_score_completions(self, inputs)

## Input Parameters
- `inputs`: Input data dictionary, containing `prompt` and other fields

## Detailed Description
The function receives input dictionary containing prompt, applies chat template to prompt, encodes and optionally truncates length, then uses diffusion generation method to produce completion.

Diffusion generation process uses iterative denoising mechanism, gradually recovering complete code sequence from masked state through specified diffusion steps, temperature parameters, and entropy algorithm. Generation process is performed in batches.

The function decides mask seed generation method based on configuration. In random masking mode, generates different random seed for each iteration; in fixed mode, all iterations use same seed.

After generated completions are decoded to text, if in conversation format, constructs conversation messages, then passes to multiple reward functions for evaluation. Each reward function returns a score, the function weights and sums these scores according to preset weights to get final reward. After reward calculation completes, the function uses leave-one-out method to calculate advantage estimate. Optionally, advantages can be normalized using within-group standard deviation.

The function logs multiple metrics, including completion length, individual reward function scores, overall reward mean and standard deviation, and proportion of zero standard deviation samples. Finally returns dictionary containing prompt and completion token IDs, masks, log probabilities, advantage estimates, and masking seeds.

## Output
- Dictionary containing following fields:
  - `prompt_ids`: Prompt token IDs
  - `prompt_mask`: Prompt mask
  - `completion_ids`: Completion token IDs
  - `completion_mask`: Completion mask
  - `old_per_token_logps`: Policy model's log probabilities (if `num_iterations > 1`)
  - `ref_per_token_logps`: Reference model's log probabilities (if `beta > 0`)
  - `advantages`: Advantage estimates
  - `mask_seeds`: Mask seed list

## Function Implementation
code/src/open_r1/coupled_grpo.py:line 383-614

## Test Code
code/test_code/test_generate_and_score_completions.py

---

# FUNCTION: code_reward

## Function Overview
Evaluates generated code and calculates test case pass rate. Extracts code blocks, builds evaluation script, executes script through code execution provider, parses pass rate from output as reward.

## Function Signature
def code_reward(completions, num_parallel=2, provider_type="e2b", enforce_same_language=False, **kwargs)

## Input Parameters
- `completions`: List of model-generated completions
- `num_parallel`: Number of parallel code executions
- `provider_type`: Code execution provider type (`e2b`, `local`, `morph`)
- `enforce_same_language`: Whether to enforce all problems use same language
- `**kwargs`: Additional parameters, including `test_cases` field

## Detailed Description
The function first calls related function to check code format of each completion. Directly assigns zero score to completions with incorrect format, extracts code content for correctly formatted ones and combines with test cases to build evaluation script. Evaluation script is independent Python program that runs test cases one by one through subprocess, calculates pass rate and outputs.

Built scripts are submitted to code execution provider for parallel execution, after execution completes parses test pass rate from output as reward score. The function returns reward list with same length as input, with zero for positions with format errors or execution failures.

## Output
- `rewards`: Reward list, each element is test case pass rate (range [0, 1])

## Function Implementation
code/src/open_r1/rewards.py:line 443-525

## Test Code
code/test_code/test_code_reward.py

---

# FUNCTION: get_code_format_reward

## Function Overview
Checks if code format is correct, including code block markers and syntax checking. Returns closure function for calculating format reward.

## Function Signature
def get_code_format_reward(language="python")

## Input Parameters
- `language`: Programming language (default `python`)

## Detailed Description
Outer function get_code_format_reward receives language parameter, configures format pattern. Format pattern requires string to start code block with ```python\n, end with ```\n, can have other content before and after code block but cannot contain extra code block markers.

Inner function code_format_reward checks if each completion matches format pattern, extracts code block content if matched.

For Python code, function further uses abstract syntax tree parser to check syntax correctness. Function returns three-level reward: 0.0 for incorrect format, 0.5 for correct format but syntax error, 1.0 for both format and syntax correct. For other languages, currently only checks format, full score for correct format.

## Output
- `code_format_reward`: Closure function, accepts `completions` parameter, returns format reward list

## Function Implementation
code/src/open_r1/rewards.py:line 529-580

## Test Code
code/test_code/test_code_format_reward.py
