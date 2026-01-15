# Quick Start: Aggregating Evaluation Metrics

## Quick Usage

### Aggregate Metrics for a Single Model

Calculate comprehensive performance of a model across multiple test examples:

```bash
cd scripts

python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-32b-instruct-simple \
  --test_examples prime ARES LUFFY PURE
```

### Batch Compare Multiple Models

Compare performance across multiple models at once:

```bash
cd scripts

python batch_aggregate_metrics.py \
  --base_dir data/tensorrt_model_optimizer \
  --model_names \
    qwen2.5-coder-7b-instruct \
    qwen2.5-coder-7b-instruct-simple \
    qwen2.5-coder-32b-instruct \
    qwen2.5-coder-32b-instruct-simple \
    qwen2.5-coder-7b-modelopt-lora \
    qwen2.5-coder-7b-modelopt-sft\
    qwen2.5-coder-7b-modelopt-sft-simple \
  --test_examples FlagScale byteperf  \
  --output_csv RL_comparison_OPT.csv
```

## Reproducing Your Results Table

Based on typical use cases, you might want a table like this:

```
Model Name            RL-prime  RL-PURE  RL-ARES  RL-LUFFY  Overall pass@1  Overall avg_pass_ratio
7b                    0         0        0        0         0               0.031
7b-sft                0         0        0        0.33      0               0.375
32b                   0.032     0        0        0.06      0               0.06
7b-lora               0         0        0        0.042     0               0.042
7b-detail             0.032     0        0        0.33      0               0.33
7b-sft-detail         0.14      0.13     0.0375   0.42      0               0.42
32b-detail            0.22      0.33     0.025    0.3125    0               0.3125
7b-lora-detail        0.032     0.09     0.1      0.33      0               0.33
```

### Step 1: Prepare Data

Ensure you have run the evaluation scripts and generated metrics files for all models:

```bash
# Each model directory should contain files like:
# data/verl/{model_name}/algorithm_methods_data_prime_result.metrics.json
# data/verl/{model_name}/algorithm_methods_data_ARES_result.metrics.json
# data/verl/{model_name}/algorithm_methods_data_LUFFY_result.metrics.json
# data/verl/{model_name}/algorithm_methods_data_PURE_result.metrics.json
```

### Step 2: Batch Aggregation

Run the batch aggregation script:

```bash
cd scripts

python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names \
    qwen2.5-coder-7b \
    qwen2.5-coder-7b-sft \
    qwen2.5-coder-32b \
    qwen2.5-coder-7b-lora \
    qwen2.5-coder-7b-detail \
    qwen2.5-coder-7b-sft-detail \
    qwen2.5-coder-32b-detail \
    qwen2.5-coder-7b-lora-detail \
  --test_examples prime PURE ARES LUFFY \
  --output_csv verl_RL_comparison.csv
```

### Step 3: View Results

**Terminal Output:**
```
================================================================================
📋 Summary Table
================================================================================

Model Name                       Total Functions  Passed  pass@1  avg_pass_ratio
--------------------------------------------------------------------------------
qwen2.5-coder-7b                        17          0       0.0000          0.0310
qwen2.5-coder-7b-sft                    17          1       0.0588          0.3750
qwen2.5-coder-32b                       17          0       0.0000          0.0600
qwen2.5-coder-7b-lora                   17          0       0.0000          0.0420
qwen2.5-coder-7b-detail                 17          1       0.0588          0.3300
qwen2.5-coder-7b-sft-detail             17          3       0.1765          0.4200
qwen2.5-coder-32b-detail                17          4       0.2353          0.3125
qwen2.5-coder-7b-lora-detail            17          2       0.1176          0.3300
================================================================================
```

**CSV File Content:**

The CSV file contains more detailed information, including separate metrics for each test example. You can open it with Excel or other tools.

### Step 4: Organize in Excel

Open the generated CSV file and you'll see:

| model_name | total_functions | total_passed | pass@1 | avg_pass_ratio | prime_pass@1 | PURE_pass@1 | ARES_pass@1 | LUFFY_pass@1 | prime_avg_pass_ratio | ... |
|------------|----------------|--------------|--------|----------------|--------------|-------------|-------------|--------------|---------------------|-----|
| qwen2.5-coder-7b | 17 | 0 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0317 | ... |
| qwen2.5-coder-7b-sft | 17 | 1 | 0.0588 | 0.3750 | 0.0000 | 0.3333 | 0.0000 | 0.3333 | 0.0000 | ... |

You can select the columns you need and rearrange them to your desired format.

## Metrics Explanation

### pass@1 (Single Test Example)
- Number of fully passed functions / Total number of functions in the test example
- Example: prime has 7 functions, 1 fully passed, then pass@1 = 1/7 = 0.14

### pass@1 (Overall)
- Total number of fully passed functions across all test examples / Total number of functions
- Example: 4 test examples with 17 total functions, 3 fully passed, then pass@1 = 3/17 = 0.176

### avg_pass_ratio (Single Test Example)
- Average pass ratio of all functions in the test example (considering partial passes)
- Example: A function has 10 tests, 3 passed, then the function's pass_ratio = 0.3

### avg_pass_ratio (Overall)
- Weighted average of avg_pass_ratio across all test examples by function count
- Formula: `Σ(avg_pass_ratio_i × functions_i) / Σ(functions_i)`

## Common Use Cases

### Use Case 1: Evaluate RL Domain Performance

```bash
python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names qwen2.5-coder-7b qwen2.5-coder-32b \
  --test_examples prime ARES LUFFY PURE \
  --output_csv RL_comparison.csv
```

### Use Case 2: Evaluate TensorRT Optimization Domain Performance

```bash
python batch_aggregate_metrics.py \
  --base_dir data/tensorrt_model_optimizer \
  --model_names qwen2.5-coder-7b qwen2.5-coder-32b \
  --test_examples byteperf FlagScale Nemo \
  --output_csv TensorRT_comparison.csv
```

### Use Case 3: Compare Different Training Methods

```bash
python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names \
    qwen2.5-coder-7b-instruct \
    qwen2.5-coder-7b-sft \
    qwen2.5-coder-7b-lora \
  --test_examples prime ARES LUFFY PURE PACS DAPO critic-rl \
  --output_csv training_method_comparison.csv
```

### Use Case 4: Detailed Analysis of Single Model

```bash
python aggregate_metrics.py \
  --model_dir data/verl/qwen2.5-coder-7b-sft-detail \
  --test_examples prime ARES LUFFY PURE \
  --output data/verl/qwen2.5-coder-7b-sft-detail/RL_analysis.json
```

## Automation Script Example

Create a script to automatically process all models:

```bash
#!/bin/bash
# auto_aggregate_all_models.sh

cd scripts

# Define all models
MODELS=(
  "qwen2.5-coder-7b"
  "qwen2.5-coder-7b-sft"
  "qwen2.5-coder-32b"
  "qwen2.5-coder-7b-lora"
  "qwen2.5-coder-7b-detail"
  "qwen2.5-coder-7b-sft-detail"
  "qwen2.5-coder-32b-detail"
  "qwen2.5-coder-7b-lora-detail"
)

# RL domain test examples
RL_EXAMPLES="prime ARES LUFFY PURE"

# Batch aggregation
python batch_aggregate_metrics.py \
  --base_dir data/verl \
  --model_names "${MODELS[@]}" \
  --test_examples $RL_EXAMPLES \
  --output_csv verl_RL_all_models.csv \
  --output_json verl_RL_all_models.json

echo "✅ Complete! Results saved to:"
echo "  - verl_RL_all_models.csv"
echo "  - verl_RL_all_models.json"
```

## Advanced Tips

### Tip 1: Process CSV with Python

```python
import pandas as pd

# Read CSV
df = pd.read_csv('verl_RL_comparison.csv')

# Sort by pass@1
df_sorted = df.sort_values('pass@1', ascending=False)

# Display only key columns
columns = ['model_name', 'pass@1', 'avg_pass_ratio', 
           'prime_pass@1', 'ARES_pass@1', 'LUFFY_pass@1', 'PURE_pass@1']
print(df_sorted[columns])

# Save sorted results
df_sorted.to_csv('verl_RL_comparison_sorted.csv', index=False)
```

### Tip 2: Generate Markdown Table

```python
import pandas as pd

df = pd.read_csv('verl_RL_comparison.csv')

# Select desired columns
columns = ['model_name', 'pass@1', 'avg_pass_ratio']
df_selected = df[columns]

# Convert to Markdown
markdown = df_selected.to_markdown(index=False)
print(markdown)

# Save as Markdown file
with open('verl_RL_comparison.md', 'w') as f:
    f.write(markdown)
```

### Tip 3: Visualize Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('verl_RL_comparison.csv')

# Plot pass@1 comparison
plt.figure(figsize=(12, 6))
plt.bar(df['model_name'], df['pass@1'])
plt.xlabel('Model')
plt.ylabel('pass@1')
plt.title('Model Comparison - pass@1')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('verl_RL_pass@1_comparison.png')
plt.show()
```

## Troubleshooting

### Issue: Cannot Find Metrics Files

Ensure you have run the evaluation scripts:

```bash
# Run evaluation
bash scripts/run_batch_execution_evaluation_pure.sh
```

### Issue: Model Directory Does Not Exist

Check directory structure:

```bash
ls -la data/verl/
```

### Issue: Incorrect CSV Format

Try reading and reformatting with Python:

```python
import pandas as pd
df = pd.read_csv('output.csv')
df.to_csv('output_fixed.csv', index=False)
```

## More Information

- Detailed Documentation: [AGGREGATE_METRICS_README.md](AGGREGATE_METRICS_README.md)
- Example Scripts: `example_aggregate_usage.sh`
- Main Tools:
  - `aggregate_metrics.py` - Single model aggregation
  - `batch_aggregate_metrics.py` - Batch model aggregation
  - `run_aggregate_metrics.sh` - Shell script wrapper

