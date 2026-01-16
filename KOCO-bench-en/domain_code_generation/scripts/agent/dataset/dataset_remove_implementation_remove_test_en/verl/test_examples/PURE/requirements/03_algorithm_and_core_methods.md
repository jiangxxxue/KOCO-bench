# PURE Algorithm Core Function Descriptions

# FUNCTION: ProcessRewardModelWorker._forward_micro_batch

## Function Overview
Computes token-level scores from the Process Reward Model (PRM) and performs approximate weighting for min-form credit assignment based on configuration.

## Function Signature
def _forward_micro_batch(self, micro_batch):

## Input Parameters
- `micro_batch`: Micro-batch data containing `input_ids`, `attention_mask`, `position_ids`, `responses`, `reward_mask`, etc.
- `response_length`: Response sequence length (inferred from batch tensors)
- `temperature`: Temperature for approximate min-form (when `reward_model.credit_assignment` is a float)

## Detailed Description
Performs forward pass on the Process Reward Model (PRM), computing token-level scores via softmax to obtain the positive-negative class probability difference. Uses masking to retain only valid scores from the response portion.
If depadding or sequence parallelism optimization is enabled, padding must be removed before computation and restored afterward.
Credit assignment weighting executes different strategies based on configuration mode: min-form, strict min-form, or gamma-decay. Lower-scoring tokens should receive higher weights.
   
## Output
- `rm_score`: Token-level process reward scores (shape `[batch_size, response_length]`)

## Function Implementation
code/verl/workers/fsdp_workers.py:line 1462-1539

## Test Code
code/test_code/test_forward_micro_batch.py


---

# FUNCTION: ProcessRewardModelWorker.compute_rm_score

## Function Overview
Batch-constructs PRM inputs (splitting by steps and appending newlines after each step), performs inference with (optional) variable-length sharding/backfilling, and aggregates to obtain batch-level token-level `rm_scores`.

## Function Signature
def compute_rm_score(self, data:DataProto):

## Input Parameters
- `data`: `DataProto` containing `responses`, `reward_mask`, etc.;

## Detailed Description
Splits response sequences by steps to construct input format for the process reward model.
In dynamic batch mode, reorganizes data into multiple mini-batches based on total token count limit and records original order; in fixed batch mode, evenly splits by sample count.
Calls forward function for each mini-batch to compute token-level scores, concatenating results from all batches. In dynamic batch mode, restores results to original order based on recorded indices.

## Output
- `DataProto`: `batch['rm_scores']` with shape `[batch_size, response_length]`

## Function Implementation
code/verl/workers/fsdp_workers.py:line 1541-1594

## Test Code
code/test_code/test_compute_rm_score.py

---

# FUNCTION: compute_return

## Function Overview
Computes token-level return sequences based on `method`

## Function Signature
def compute_return(token_level_rewards, eos_mask, method='sum', gamma=1.0):

## Input Parameters
- `token_level_rewards`: `[batch_size, response_length]`
- `eos_mask`: `[batch_size, response_length]`
- `method`: `'sum'|'min'`
- `gamma`: Discount factor (applicable in `'sum'` mode)

## Detailed Description
Computes token-level return sequences based on method, using masking to ensure only valid positions are processed.

"method" includes "sum" and "min".

## Output
- `returns`: `[batch_size, response_length]`

## Function Implementation
code/verl/trainer/ppo/core_algos.py:line 70-88

## Test Code
code/test_code/test_compute_return.py


