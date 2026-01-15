# ARES Algorithm Core Function Descriptions

# FUNCTION: compute_score

## Function Overview
Computes ARES hierarchical entropy rewards, dynamically adjusting reward signals based on sample difficulty, accuracy, and High Window Entropy (HWE) token count.


## Function Signature
def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, alpha_length_bonus: float = 0.5, alpha_entropy: float = 0.5) -> List[Dict[str, float]]:

## Input Parameters
- `reward_inputs`: List of reward input dictionaries, each containing:
  - `response`: Model-generated response text
  - `ground_truth`: Ground truth answer
  - `difficulty`: Sample difficulty label ("easy" / "medium" / "hard")
  - `high_entropy_token_num`: Number of HWE tokens in response
  - `target_high_entropy_token_num`: Target HWE token count (difficulty-related)
  - `alpha_entropy`: Entropy reward coefficient (difficulty-related)
  - `accuracy`: Accuracy label (optional, if not provided, extracted from response and evaluated)
- `format_weight`: Format reward weight, default 0.1
- `alpha_length_bonus`: Length reward coefficient (legacy parameter, actual usage is alpha_entropy)
- `alpha_entropy`: Default entropy reward coefficient, default 0.5 (used if sample doesn't specify)

## Detailed Description
This function implements ARES's core reward shaping mechanism, adaptively adjusting reasoning resource allocation based on task difficulty and model performance. The function computes format reward, accuracy reward, and entropy reward. For easy questions, when the answer is correct, it penalizes overly long reasoning chains, while when the answer is incorrect, it provides moderate positive rewards to encourage the model to explore more possibilities. Medium-difficulty questions require reasoning length closer to the target value; both excessive and insufficient reasoning are penalized when the answer is correct, but longer reasoning is still encouraged when the answer is incorrect. Hard questions adopt the opposite strategy: when the answer is correct, reasoning chains close to or exceeding the target are rewarded, and penalties are only applied when reasoning is significantly insufficient; when the answer is incorrect, longer reasoning is also encouraged to find the correct path.

The entire reward mechanism uses Huber loss function to smoothly handle penalty terms, Sigmoid function to achieve reward saturation effects, and sets different tolerance boundaries and reward caps for different difficulties.

Difficulty-related hyperparameters that may be used:
- `MARGIN_FRAC`: Tolerance band width (easy: 15%, medium: 25%, hard: 35%)
- `KAPPA`: Huber function inflection point (easy: 2.0, medium: 3.0, hard: 4.0)
- `TEMP`: Sigmoid temperature parameter (easy: 2.0, medium: 2.5, hard: 3.0)
- `CAP_SCALE`: Entropy reward cap coefficient (easy: 1.0, medium: 1.0, hard: 1.2)

## Output
- `scores`: List of score dictionaries, each containing:
  - `overall`: Total score = accuracy + entropy_score
  - `accuracy`: Accuracy score (0.0 or 1.0)
  - `format`: Format score (0.0 or 1.0)
  - `high_entropy_token_num_score`: Entropy reward score

## Function Implementation
code/examples/reward_function/aepo.py:line 58-133

## Test Code
code/test_code/test_compute_score.py

---

# FUNCTION: RayPPOTrainer.update_difficulty_and_skip_gid_set

## Function Overview
Updates sample difficulty dictionary and adaptive entropy coefficients based on current batch rollout results, implementing online difficulty bucketing and resource budget adjustment.

## Function Signature
def update_difficulty_and_skip_gid_set(self, batch: DataProto) -> None:

## Input Parameters
- `batch`: DataProto object containing:
  - `global_id`: Sample global ID (for cross-batch tracking)
  - `accuracy`: Accuracy list (accuracy of each response)
  - `entropies`: Average entropy list (average token entropy of each response)
  - `high_entropy_token_num`: HWE token count list (HWE token count of each response)

## Detailed Description
This function is responsible for dynamically updating question difficulty classification and adaptively adjusting entropy penalty coefficients during training. The function needs to aggregate accuracy, entropy, and high-entropy token counts for each question by global identifier. Questions are categorized into easy, medium, and hard levels based on accuracy. Based on high-entropy token counts, the function needs to compute target reasoning length for each difficulty level and adjust entropy penalty coefficients based on deviation between actual generated length and target value. When adjusting entropy penalty coefficients, gradient ascent can be applied. For easy and medium questions, if reasoning is too long, increase penalty strength; for hard questions, increase encouragement when reasoning is insufficient.

Difficulty assignment based on accuracy thresholds:
  - `acc_rate ≥ 2/3` → "easy"
  - `1/3 ≤ acc_rate < 2/3` → "medium"
  - `acc_rate < 1/3` → "hard"

(Optional) The function adds samples that are "all correct with extremely low average entropy" to `self.skip_gid_set`, and subsequent training will skip these overly simple samples, focusing on challenging data.

## Output
- No direct return value, but updates the following class attributes:
  - `self.difficulty_dict_train`: Sample difficulty dictionary `{global_id: difficulty}`
  - `self.target_high_entropy_token_num_dict`: Target HWE token count `{difficulty: target_num}`
  - `self.alpha_entropy_dict`: Entropy coefficient dictionary `{difficulty: alpha}`
  - `self.skip_gid_set`: Skip sample set (optional)

## Function Implementation
code/verl/trainer/ray_trainer.py:line 775-825

## Test Code
code/test_code/test_update_difficulty.py

---


# FUNCTION: TokenLevelAdaptiveKLController.update

## Function Overview
Implements update logic for token-level adaptive KL controller, dynamically adjusting KL coefficients using dual ascent mechanism based on deviation between actual KL divergence and target for each difficulty bucket.

## Function Signature
def update(self, current_kls: Dict[str, float], n_steps: Dict[str, int]):

## Input Parameters
- `current_kls`: Average KL divergence for each difficulty bucket at current step, dictionary format `{difficulty: kl_value}`
  - Example: `{"easy": 0.06, "medium": 0.12, "hard": 0.18}`
- `n_steps`: Sample count for each difficulty bucket (used for weighting, may be simplified in actual implementation)

## Detailed Description
Token-level adaptive KL control allows using different KL budgets for different difficulty tasks, automatically adjusting KL divergence coefficients for each difficulty bucket through dual ascent mechanism. When actual KL divergence exceeds target value, increase penalty coefficient; when below target value, decrease coefficient.

## Output
- No direct return value, but updates class attributes:
  - `self.kl_coefs`: Updated KL coefficient dictionary `{difficulty: beta}`
  - `self.lambdas`: Updated Lagrange multiplier dictionary `{difficulty: lambda}`

## Function Implementation
code/verl/trainer/core_algos.py:line 77-86

## Test Code
code/test_code/test_kl_controller.py

---


# FUNCTION: apply_kl_penalty

## Function Overview
Applies KL divergence penalty to token-level rewards and updates KL controller.

## Function Signature
def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):

## Input Parameters
- `data`: DataProto object containing:
  - `token_level_scores`: Original token-level rewards [batch_size, response_length]
  - `old_log_probs`: Actor old policy log probabilities [batch_size, response_length]
  - `ref_log_probs`: Ref policy log probabilities [batch_size, response_length]
  - `response_mask`: Response mask [batch_size, response_length]
- `kl_ctrl`: KL controller (FixedKLController or AdaptiveKLController)
- `kl_penalty`: KL penalty type ("kl", "abs", "mse", "low_var_kl", "full")

## Detailed Description
This function implements computation and application of token-level KL penalty. It computes token-level KL divergence based on `kl_penalty` type, then subtracts KL penalty term from original rewards to obtain token-level rewards, and computes sequence-level KL divergence and batch-level KL divergence.

The function also needs to implement KL controller update.

## Output
- `data`: Updated DataProto where `data.batch["token_level_rewards"]` has KL penalty applied
- `metrics`: Metrics dictionary `{"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}`

## Function Implementation
code/verl/trainer/ray_trainer.py:line 107-124

## Test Code
code/test_code/test_apply_kl_penalty.py

