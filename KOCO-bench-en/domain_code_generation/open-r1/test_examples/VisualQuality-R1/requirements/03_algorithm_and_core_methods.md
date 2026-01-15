# VisualQuality-R1 Algorithm Core Function Descriptions

# FUNCTION: fidelity_reward

## Function Overview
Calculates ranking fidelity reward between a single sample pair, measuring the consistency between the model's predicted ranking relationship and ground truth quality labels.

## Function Signature
def fidelity_reward(pred1, pred2, var1, var2, gt, device):

## Input Parameters
- `pred1`: Predicted score of the first sample (scalar or tensor)
- `pred2`: Mean predicted score of the second sample (scalar or tensor)
- `var1`: Prediction variance of the first sample (scalar or tensor)
- `var2`: Prediction variance of the second sample (scalar or tensor)
- `gt`: Ground truth ranking relationship (0.0 means pred2 better than pred1, 1.0 means pred1 better than pred2, 0.5 means equal)
- `device`: Computing device

## Detailed Description
The function divides the difference between two predicted values by their total uncertainty to obtain a standardized difference value.
Δ = (pred1 - pred2) / √(var1 + var2 + ε), where ε = 1e-6 prevents division by zero

This standardized difference is converted to a probability through the cumulative distribution function of the standard normal distribution, representing the likelihood that the first prediction is greater than the second prediction. p = Φ(Δ)

Based on this probability and the ground truth ranking relationship, the function calculates the reward value.
reward = √(p × gt + ε) + √((1-p) × (1-gt) + ε)

## Output
- `reward`: Fidelity reward value (scalar or tensor)

## Function Implementation
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 128-139

## Test Code
code/test_code/test_fidelity_reward.py

---

# FUNCTION: accuracy_reward

## Function Overview
Batch calculates fidelity rewards for all samples, learning relative ranking relationships of image quality through pairwise comparison.

## Function Signature
def accuracy_reward(completions, solution, **kwargs):

## Input Parameters
- `completions`: List of generated responses, each response contains `[{"role": "assistant", "content": "..."}]` format
- `solution`: List of ground truth MOS scores (containing `<answer>` tags)
- `kwargs`: Additional parameters
  - `device`: Computing device
  - `num_generations`: Number of generations per sample G
  - `image_path`: Image path (for debug logging)
  - `problem`: Question text (for debug logging)

## Detailed Description
The function groups inputs by prompt, each group contains multiple generation results and ground truth answer for that prompt. Extracts numerical labels from ground truth answers, extracts numerical predictions from completions (if extraction fails, uses random number between 1 and 5). For each prompt group, the function calculates mean and variance of all predicted values in that group as statistical features of that group.

Reward calculation uses cross-group comparison. For each generation result, the function compares it with all other groups. Each comparison uses current predicted value, another group's average predicted value, current group's variance, and another group's variance as inputs. Determines expected comparison result based on size relationship of two ground truth answers.

Each pairwise comparison calculates score through fidelity reward function. The function averages comparison scores of current generation result with all other groups as final reward for that generation result.

In debug mode, logs reward value, image path, question, and response content.

## Output
- `rewards`: Fidelity reward list, length is `batch_size × num_generations`

## Function Implementation
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 142-225

## Test Code
code/test_code/test_accuracy_reward.py

---

# FUNCTION: format_reward

## Function Overview
Checks if generated responses conform to specified format requirements (containing `<think>` and `<answer>` tags).

## Function Signature
def format_reward(completions, **kwargs):

## Input Parameters
- `completions`: List of generated responses, each response contains `[{"role": "assistant", "content": "..."}]` format
- `kwargs`: Additional parameters (for debug logging)

## Detailed Description
The function uses regular expressions to verify if content exactly matches a specific pattern. Required format includes two parts: think tag <think> and answer tag <answer>, with think part first, answer part second, and whitespace allowed between the two parts. The function uses exact match mode, not allowing extra content before, after, or in the middle of the format.

The function returns binary reward list, with 1.0 for positions with exact format match, otherwise 0.0. In debug mode, logs each response's content and format check result.

## Output
- `rewards`: Format reward list, each element is 1.0 or 0.0

## Function Implementation
code/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py:line 228-243

## Test Code
code/test_code/test_format_reward.py


---

# FUNCTION: VLMGRPOTrainer.compute_loss

## Function Overview
Calculates loss function for GRPO training, including PPO clipped loss and KL divergence penalty.

## Function Signature
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

## Input Parameters
- `model`: Policy model
- `inputs`: Input data (may be original data or cached generation results)
- `return_outputs`: Whether to return outputs (GRPO does not support, must be False)
- `num_items_in_batch`: Number of samples in batch (unused)

## Detailed Description
The function decides whether to generate new completions or use cached results based on current training step. Generates new completions and caches them at fixed iteration intervals, reads from cache at other steps, implementing multiple iterations optimizing same batch of generation results.

The function extracts prompt, completion, and multimodal input parameters from inputs. After concatenating prompt and completion, passes to current policy model to calculate token-level log probabilities, passing multimodal inputs to support vision-language models. Only retains completion part for loss calculation.

Loss calculation uses PPO clipped objective, calculating probability ratio between current policy and old policy and its clipped version, multiplying each by advantage and taking minimum value's negative as policy loss. If KL coefficient is non-zero, calculates KL divergence between current policy and reference model and adds weighted to loss.

Final loss only calculates average of valid tokens through completion mask. The function logs training metrics such as KL divergence and clipping ratio, returning loss value.

## Output
- `loss`: Scalar loss value

## Function Implementation
code/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py:line 828-886

## Test Code
code/test_code/test_compute_loss.py

---


