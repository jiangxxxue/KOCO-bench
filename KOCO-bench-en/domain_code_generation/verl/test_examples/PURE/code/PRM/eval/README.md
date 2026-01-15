# Evaluating PURE-PRM-7B

## ProcessBench Evaluation

Accelerate evaluation using multi-GPU and batch inference:

```bash
torchrun --nproc_per_node=4 process_bench.py \
  --model jinachris/PURE-PRM-7B \
  --batch_size 24 \
  --num_of_workers 4 \
  --separator "\n"
```

## PRMBench Evaluation

Refer to [our implementation in the official PRMBench repository](https://github.com/ssmisya/PRMBench/commit/3e9a9d10b88c1599e24bec5869bfa5caba2bcf40). Execute with:

```bash
bash mr_eval/scripts/examples/direct_run.sh
```

## Best-of-N (BoN) Evaluation

Run:

```bash
python bon.py
```

**Note:** This script currently lacks multi-GPU/batch inference and may run slowly. We welcome contributions to accelerate it following the approach in `process_bench.py` – PRs are encouraged!