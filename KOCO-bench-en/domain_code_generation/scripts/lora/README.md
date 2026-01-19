# LoRA Fine-tuning Commands

## Build Dataset

```bash
cd ..
python finetune_dataset_builder.py \
  --source-dir /path/to/your/codebase \
  --output-file ./data/your_framework/training_dataset.jsonl
```

## Run Fine-tuning

```bash
cd lora
bash run_finetuning_lora.sh
```

