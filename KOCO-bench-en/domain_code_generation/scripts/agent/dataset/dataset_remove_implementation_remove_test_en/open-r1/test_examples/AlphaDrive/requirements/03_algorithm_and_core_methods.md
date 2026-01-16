# AlphaDrive Algorithm Core Function Descriptions

# FUNCTION: plan_speed_reward

## Function Overview
Computes reward score for speed planning decisions, comprehensively considering accuracy (F1 score), decision complexity, and diversity factor.

## Function Signature
```python
def plan_speed_reward(completions, solution, diversity_weight=0.4, complexity_weights=None, **kwargs)
```

## Input Parameters
- `completions`: List of model-generated responses, each element is dictionary containing `content` field
- `solution`: Ground truth answer list containing `<answer>` tags
- `diversity_weight`: Diversity factor weight, default 0.4
- `complexity_weights`: Complexity weight dictionary for each decision action, default:
  ```python
  {
      "ACCELERATE": 0.9,
      "DECELERATE": 1.0,
      "STOP": 1.0,
      "KEEP": 0.8
  }
  ```

## Detailed Description

Function implements reward mechanism for evaluating autonomous driving speed planning decision quality, scoring from three dimensions. First is accuracy, computing F1 score by comparing predicted decisions with ground truth. Second is complexity, where different decisions have different weights, requiring averaging complexity weights of all predicted decision actions. Finally is diversity, where reward is given when all decision actions appear for first time, repeated patterns receive penalty. Final reward score is product of accuracy and complexity, plus diversity.

## Output
- `rewards`: Float list, each element corresponds to reward score for one completion
 
## Function Implementation
code/src/r1-v/src/open_r1/grpo.py:line 54-109

## Test Code
code/test_code/test_plan_speed_reward.py


---

# FUNCTION: plan_path_reward

## Function Overview
Computes reward score for path planning decisions, comprehensively considering accuracy (F1 score), decision complexity, and diversity factor.

## Function Signature
```python
def plan_path_reward(completions, solution, diversity_weight=0.4, complexity_weights=None, **kwargs)
```

## Input Parameters
- `completions`: List of model-generated responses, each element is dictionary containing `content` field
- `solution`: Ground truth answer list containing `<answer>` tags
- `diversity_weight`: Diversity factor weight, default 0.4
- `complexity_weights`: Complexity weight dictionary for each decision action, default:
  ```python
  {
      "LEFT_TURN": 1.0,
      "RIGHT_TURN": 1.0,
      "LEFT_CHANGE": 1.0,
      "RIGHT_CHANGE": 1.0,
      "STRAIGHT": 0.8
  }
  ```

## Detailed Description

Function implements reward mechanism for evaluating autonomous driving path planning decision quality, scoring from three dimensions. First is accuracy, computing F1 score by comparing predicted decisions with ground truth. Second is complexity, where different decisions have different weights, requiring averaging complexity weights of all predicted decision actions. Finally is diversity, where reward is given when all decision actions appear for first time, repeated patterns receive penalty. Final reward score is product of accuracy and complexity, plus diversity.

## Output
- `rewards`: Float list, each element corresponds to reward score for one completion

## Function Implementation
code/src/r1-v/src/open_r1/grpo.py:line 112-168

## Test Code
code/test_code/test_plan_path_reward.py


---

# FUNCTION: plan_format_reward

## Function Overview
Validates if model output conforms to expected structured format: `<think>...</think><answer>...</answer>`.

## Function Signature
```python
def plan_format_reward(completions, **kwargs)
```

## Input Parameters
- `completions`: List of model-generated responses, each element is dictionary containing `content` field

## Detailed Description

Function implements format checking reward mechanism. It validates if model output strictly follows <think>thinking content</think><answer>answer content</answer> format specification. Whitespace characters are allowed between tags, tag content can be any characters. If output completely conforms to this format, full reward (1.0) is given, otherwise no score (0.0).

## Output
- `rewards`: Float list, each element is 1.0 (format correct) or 0.0 (format incorrect)

## Function Implementation
code/src/r1-v/src/open_r1/grpo.py:line 171-178

## Test Code
code/test_code/test_plan_format_reward.py


---

# FUNCTION: Qwen2VLGRPOTrainer.compute_loss

## Function Overview
Core function of GRPO training, computes policy optimization loss. This function processes multimodal input, generates responses, computes rewards, estimates advantages, and updates policy.

## Function Signature
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)
```

## Input Parameters
- `model`: Policy model (VLM)
- `inputs`: Batch data containing `prompt` (dialogue), `image` (image path), etc.
- `return_outputs`: Whether to return outputs (GRPO doesn't support, must be False)
- `num_items_in_batch`: Number of samples in batch (optional)

## Detailed Description
Training flow implements GRPO reinforcement learning training based on multimodal input. First loads driving scene images and text descriptions, converting them to model input format. Generates multiple candidate driving decision responses for each scene, processing end-of-sequence token masks for generated sequences.

Computes log probabilities of current model and reference model for generated content, used for subsequent KL divergence constraints. After decoding generated decision text, calls configured reward evaluation functions for scoring. Reward functions are divided into two categories: pretrained reward models (directly score combination of prompts and responses) and custom evaluation functions (need to receive additional fields from input data, such as scene parameters, ground truth answers, complexity weights, etc.).

After aggregating scores from all reward functions, uses GRPO's within-group normalization method to compute relative advantage values.

Finally multiplies log probability of each token by corresponding advantage value, subtracts KL divergence penalty term, averages over all tokens in generation portion. Also records training metrics including generation length, scores from each reward function, total reward, reward standard deviation, and KL divergence.

## Output
- `loss`: Scalar tensor for backpropagation and parameter update

## Function Implementation
code/src/r1-v/src/open_r1/trainer/grpo_trainer.py:line 363-500

## Test Code
code/test_code/test_compute_loss.py



