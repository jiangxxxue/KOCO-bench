#!/bin/bash
# LoRA 微调脚本（适配 finetuning_lora.py）
# 基于原版 run_finetuning.sh，调整为 LoRA 参数高效微调

# 严格模式：遇错立即退出
set -euo pipefail

# ========= 基础路径 =========
MODEL_PATH="/home/shixianjie/models/Qwen2.5-Coder-7B-Instruct"
FRAMEWORK="smolagents"
DATA_PATH="../data/${FRAMEWORK}/${FRAMEWORK}_training_dataset.jsonl"
OUTPUT_DIR="../models/qwen2.5-coder-7b-${FRAMEWORK}-lora"

# ========= LoRA 参数 =========
LORA_R=16                                        # LoRA rank，建议 8-64
LORA_ALPHA=32                                    # LoRA alpha，通常为 2*r
LORA_DROPOUT=0.05                                # LoRA dropout
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"    # 应用 LoRA 的模块
USE_RSLORA=false                                 # 是否使用 Rank-Stabilized LoRA
USE_DORA=false                                   # 是否使用 DoRA

# ========= 训练参数（针对 LoRA 优化）=========
MAX_SEQ_LENGTH=2048
BATCH_SIZE=4                                     # LoRA 显存占用更少，可适当增大
GRADIENT_ACCUMULATION=2                          # 相应减少梯度累积
LEARNING_RATE=1e-4                               # LoRA 通常使用更大的学习率（1e-4 到 3e-4）
NUM_EPOCHS=5                                     # LoRA 收敛快，可适当增加轮数
WARMUP_RATIO=0.03
KEEP_FILE_TYPES="python,shell,yaml,markdown"
STRIDE_FRACTION=0.125
ADD_FILE_PATH_HEADER="false"

# ========= GPU / 环境 =========
export CUDA_VISIBLE_DEVICES=4,5
export TOKENIZERS_PARALLELISM=false

# 禁用 flash-attention 自动检测（避免 GLIBC 问题）
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export DISABLE_FLASH_ATTN=1

echo "========================================================"
echo "🚀 开始 ${FRAMEWORK} 代码库 LoRA 微调（finetuning_lora.py）"
echo "========================================================"
echo "模型: ${MODEL_PATH}"
echo "数据: ${DATA_PATH}"
echo "输出: ${OUTPUT_DIR}"
echo "序列长度: ${MAX_SEQ_LENGTH}"
echo "Batch大小: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "学习率: ${LEARNING_RATE}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "========================================================"
echo "LoRA 配置:"
echo "  - Rank (r): ${LORA_R}"
echo "  - Alpha: ${LORA_ALPHA}"
echo "  - Dropout: ${LORA_DROPOUT}"
echo "  - Target Modules: ${TARGET_MODULES}"
echo "  - Use RSLoRA: ${USE_RSLORA}"
echo "  - Use DoRA: ${USE_DORA}"
echo "========================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# 切换到 lora 目录执行
cd "$(dirname "$0")"

# --------- LoRA 微调（单机多卡或单卡）---------
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
echo "🎉 LoRA 微调完成！Adapter 保存在: ${OUTPUT_DIR}"
echo "========================================================"
echo ""
echo "💡 使用方法："
echo "from peft import PeftModel"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo ""
echo "base_model = AutoModelForCausalLM.from_pretrained('${MODEL_PATH}')"
echo "model = PeftModel.from_pretrained(base_model, '${OUTPUT_DIR}')"
echo "tokenizer = AutoTokenizer.from_pretrained('${OUTPUT_DIR}')"
echo "========================================================"

