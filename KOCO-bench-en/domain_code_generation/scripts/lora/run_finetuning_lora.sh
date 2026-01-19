#!/bin/bash
# LoRA fine-tuning script (compatible with finetuning_lora.py)
# Based on the original run_finetuning.sh, adapted for LoRA parameter-efficient fine-tuning

# Strict mode: exit on error
set -euo pipefail

# ========= Basic Paths =========
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
FRAMEWORK="${FRAMEWORK:-your_framework}"
DATA_PATH="../data/${FRAMEWORK}/${FRAMEWORK}_training_dataset.jsonl"
OUTPUT_DIR="../models/${FRAMEWORK}-lora"

# ========= LoRA Parameters =========
LORA_R=16                                        # LoRA rank, recommended 8-64
LORA_ALPHA=32                                    # LoRA alpha, typically 2*r
LORA_DROPOUT=0.05                                # LoRA dropout
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"    # Modules to apply LoRA
USE_RSLORA=false                                 # Whether to use Rank-Stabilized LoRA
USE_DORA=false                                   # Whether to use DoRA

# ========= Training Parameters (Optimized for LoRA) =========
MAX_SEQ_LENGTH=2048
BATCH_SIZE=4                                     # LoRA uses less memory, can increase batch size
GRADIENT_ACCUMULATION=2                          # Reduce gradient accumulation accordingly
LEARNING_RATE=1e-4                               # LoRA typically uses larger learning rate (1e-4 to 3e-4)
NUM_EPOCHS=5                                     # LoRA converges faster, can increase epochs
WARMUP_RATIO=0.03
KEEP_FILE_TYPES="python,shell,yaml,markdown"
STRIDE_FRACTION=0.125
ADD_FILE_PATH_HEADER="false"

# ========= GPU / Environment =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

# Disable flash-attention auto-detection (avoid GLIBC issues)
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export DISABLE_FLASH_ATTN=1

echo "========================================================"
echo "🚀 Starting ${FRAMEWORK} codebase LoRA fine-tuning (finetuning_lora.py)"
echo "========================================================"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Sequence Length: ${MAX_SEQ_LENGTH}"
echo "Batch Size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "========================================================"
echo "LoRA Configuration:"
echo "  - Rank (r): ${LORA_R}"
echo "  - Alpha: ${LORA_ALPHA}"
echo "  - Dropout: ${LORA_DROPOUT}"
echo "  - Target Modules: ${TARGET_MODULES}"
echo "  - Use RSLoRA: ${USE_RSLORA}"
echo "  - Use DoRA: ${USE_DORA}"
echo "========================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Switch to lora directory
cd "$(dirname "$0")"

# --------- LoRA Fine-tuning (Single or Multi-GPU) ---------
python finetuning_lora.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
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
  --save_strategy steps \
  --save_total_limit 3 \
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
  --add_file_path_header "${ADD_FILE_PATH_HEADER}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --target_modules "${TARGET_MODULES}" \
  --use_rslora "${USE_RSLORA}" \
  --use_dora "${USE_DORA}"

echo ""
echo "========================================================"
echo "🎉 LoRA fine-tuning completed! Adapter saved to: ${OUTPUT_DIR}"
echo "========================================================"
echo ""
echo "💡 Usage:"
echo "from peft import PeftModel"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo ""
echo "base_model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}')"
echo "model = PeftModel.from_pretrained(base_model, '${OUTPUT_DIR}')"
echo "tokenizer = AutoTokenizer.from_pretrained('${OUTPUT_DIR}')"
echo "========================================================"

