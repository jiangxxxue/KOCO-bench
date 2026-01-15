#!/bin/bash
#代码库 NTP 续训脚本（适配 finetuning.py）

# 严格模式：遇错立即退出
set -euo pipefail

# ========= 基础路径 =========
MODEL_PATH="/home/shixianjie/models/Qwen2.5-Coder-7B-Instruct"
FRAMEWORK="raganything"
DATA_PATH="../data/${FRAMEWORK}/${FRAMEWORK}_training_dataset.jsonl"
OUTPUT_DIR="../models/qwen2.5-coder-7b-${FRAMEWORK}-sft"

# ========= 训练参数（NTP 优化）=========
MAX_SEQ_LENGTH=2048
BATCH_SIZE=2
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-6
NUM_EPOCHS=2
WARMUP_RATIO=0.03
KEEP_FILE_TYPES="python,shell,yaml,markdown"     # 与 finetuning.py 的 ModelArguments 对齐
STRIDE_FRACTION=0.125                            # 滑窗重叠比例 (= 1/8 * seq_len)
ADD_FILE_PATH_HEADER="false"                     # 是否在样本前加“# File: path”注释

# ========= GPU / 环境 =========
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
NUM_GPUS=4  # 使用的 GPU 数量

echo "========================================================"
echo "🚀 开始 ${FRAMEWORK} 代码库 NTP 续训（finetuning.py）"
echo "========================================================"
echo "模型: ${MODEL_PATH}"
echo "数据: ${DATA_PATH}"
echo "输出: ${OUTPUT_DIR}"
echo "序列长度: ${MAX_SEQ_LENGTH}"
echo "Batch大小: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "学习率: ${LEARNING_RATE}"
echo "训练轮数: ${NUM_EPOCHS}"
echo "文件类型白名单: ${KEEP_FILE_TYPES}"
echo "滑窗重叠比例: ${STRIDE_FRACTION}"
echo "样本头部注释: ${ADD_FILE_PATH_HEADER}"
echo "========================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"

# --------- 单机多卡训练（DeepSpeed ZeRO-3 模型分片）---------
# 适用于：模型太大，单卡放不下，需要多卡一起加载模型
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
echo "🎉 训练完成！模型保存在: ${OUTPUT_DIR}"
echo "========================================================"
