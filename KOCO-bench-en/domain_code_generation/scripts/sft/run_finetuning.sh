#!/bin/bash
# Fine-tuning script for code repository NTP (Next Token Prediction)
# Compatible with finetuning.py

# Strict mode: exit on error
set -euo pipefail

# ========= Basic Paths =========
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
FRAMEWORK="${FRAMEWORK:-your_framework}"
DATA_PATH="../data/${FRAMEWORK}/${FRAMEWORK}_training_dataset.jsonl"
OUTPUT_DIR="../models/${FRAMEWORK}-sft"

# ========= Training Parameters (NTP Optimized) =========
MAX_SEQ_LENGTH=2048
BATCH_SIZE=2
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-6
NUM_EPOCHS=2
WARMUP_RATIO=0.03
KEEP_FILE_TYPES="python,shell,yaml,markdown"     # Aligned with ModelArguments in finetuning.py
STRIDE_FRACTION=0.125                            # Sliding window overlap ratio (= 1/8 * seq_len)
ADD_FILE_PATH_HEADER="false"                     # Whether to add "# File: path" comment at sample start

# ========= GPU / Environment =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
NUM_GPUS="${NUM_GPUS:-4}"  # Number of GPUs to use

echo "========================================================"
echo "🚀 Starting ${FRAMEWORK} codebase NTP fine-tuning (finetuning.py)"
echo "========================================================"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Sequence Length: ${MAX_SEQ_LENGTH}"
echo "Batch Size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "File Type Whitelist: ${KEEP_FILE_TYPES}"
echo "Stride Fraction: ${STRIDE_FRACTION}"
echo "Add File Path Header: ${ADD_FILE_PATH_HEADER}"
echo "========================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# --------- Multi-GPU Training (DeepSpeed ZeRO-3 Model Sharding) ---------
# Suitable for: Models too large to fit on a single GPU, requiring multi-GPU model loading
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_CONFIG="${SCRIPT_DIR}/ds_config_zero3.json"

deepspeed --num_gpus=${NUM_GPUS} finetuning.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DS_CONFIG}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --val_split_ratio 0.1 \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_grad_norm 1.0 \
  --optim adamw_torch \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 3 \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --use_wandb false \
  --fp16 false \
  --bf16 true \
  --tf32 true \
  --dataloader_num_workers 4 \
  --gradient_checkpointing true \
  --remove_unused_columns false \
  --logging_first_step true \
  --report_to none \
  --keep_file_types "${KEEP_FILE_TYPES}" \
  --stride_fraction "${STRIDE_FRACTION}" \
  --add_file_path_header "${ADD_FILE_PATH_HEADER}"

echo ""
echo "========================================================"
echo "🎉 Training completed! Model saved to: ${OUTPUT_DIR}"
echo "========================================================"
