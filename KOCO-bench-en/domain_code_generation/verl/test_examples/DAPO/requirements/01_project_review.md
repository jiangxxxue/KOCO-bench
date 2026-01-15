# DAPO Project Overview (Decoupled Clip and Dynamic Sampling Policy Optimization)

## 1. Algorithm Overview

DAPO is based on the PPO algorithm framework, improving training efficiency and stability through the following enhancements:

### 1.1 Decoupled Clipping (Clip-Higher)

Traditional PPO uses a single clipping parameter ε, limiting the importance sampling ratio within [1-ε, 1+ε]. DAPO introduces asymmetric clipping:

```
L_clip = max(-A * r, -A * clip(r, 1-ε_low, 1+ε_high))
```

Where:
- ε_low: Clipping threshold for negative advantage (A < 0) direction
- ε_high: Clipping threshold for positive advantage (A > 0) direction
- Typically set ε_high > ε_low, allowing more aggressive promotion of good behaviors while conservatively suppressing bad behaviors

### 1.2 Dynamic Sampling (Dynamic Sampling with Group Filtering)

In standard PPO, after generating n responses for each prompt, they are directly used for training. DAPO introduces a group filtering mechanism:

1. For n responses from each prompt, compute a metric (such as accuracy acc, final reward seq_final_reward)
2. Filter out groups where all response metrics are identical (e.g., all correct or all incorrect), as these groups cannot provide effective contrastive signals
3. If the number of valid groups after filtering < train_batch_size, continue sampling new batches
4. Repeat sampling until reaching train_batch_size or hitting the limit max_num_gen_batches

This mechanism significantly improves sample utilization efficiency, especially when data difficulty is high.

### 1.3 Flexible Loss Aggregation

DAPO supports multiple loss aggregation modes (loss_agg_mode):

- **token-mean** (recommended): Average over all tokens, making each token's contribution to loss equal
- **seq-mean-token-sum**: Sum tokens within sequences first, then average across sequences
- **seq-mean-token-mean**: Average tokens within sequences first, then average across sequences

The token-mean mode makes policy gradients focus more on the quality of each token rather than sequence length.

### 1.4 Overlong Reward Shaping

To prevent the model from generating excessively long but meaningless outputs, DAPO introduces linear penalty:

```
if response_len > expected_len:
    penalty = min(-(response_len - expected_len) / buffer_len * penalty_factor, 0)
    reward += penalty
```

Where expected_len = max_response_length - overlong_buffer_len, and the penalty increases linearly within the buffer interval.

## 2. Training Pipeline (Pseudocode)

```
Algorithm: DAPO Training Pipeline
Input: Dataset D, Policy π_θ, Reference π_ref, RewardManager R
Config: rollout.n, filter_groups.enable, filter_groups.metric,
        clip_ratio_low, clip_ratio_high, loss_agg_mode

1: for epoch in 1..E do
2:   for batch B ~ D do
3:       batch_data ← []
4:       num_gen_batches ← 0
5:       # Dynamic sampling loop
6:       while len(batch_data) < train_batch_size do
7:           # Generate n responses
8:           responses ← rollout(π_θ, B, n)
9:           
10:          # Compute rewards
11:          rewards ← R.compute(B, responses)
12:          
13:          # Group filtering (if enabled)
14:          if filter_groups.enable then
15:              # Group by prompt, compute metric variance for each group
16:              for each prompt_group in responses do
17:                  metric_vals ← [compute_metric(resp) for resp in prompt_group]
18:                  if std(metric_vals) == 0 then  # All identical
19:                      discard prompt_group
20:              end for
21:          end if
22:          
23:          batch_data ← batch_data + filtered_responses
24:          num_gen_batches += 1
25:          
26:          if num_gen_batches >= max_num_gen_batches then
27:              break  # Reached sampling limit
28:          end if
29:       end while
30:       
31:       # Clip to target batch size
32:       batch_data ← batch_data[:train_batch_size * n]
33:       
34:       # Compute old_log_prob and advantages
35:       old_log_prob ← π_θ.compute_log_prob(batch_data)
36:       ref_log_prob ← π_ref.compute_log_prob(batch_data)
37:       advantages ← compute_advantage(rewards, ref_log_prob, ...)
38:       
39:       # PPO update (with decoupled clipping)
40:       for ppo_epoch in 1..K do
41:           log_prob ← π_θ.compute_log_prob(batch_data)
42:           ratio ← exp(log_prob - old_log_prob)
43:           pg_loss1 ← -advantages * ratio
44:           pg_loss2 ← -advantages * clip(ratio, 1-ε_low, 1+ε_high)
45:           pg_loss ← agg_loss(max(pg_loss1, pg_loss2), loss_agg_mode)
46:           θ ← θ + Adam(∇_θ pg_loss)
47:       end for
48:   end for
49: end for
```


