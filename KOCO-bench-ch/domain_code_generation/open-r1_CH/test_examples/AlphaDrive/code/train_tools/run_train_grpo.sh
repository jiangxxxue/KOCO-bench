#!/bin/bash

# dynamic settings
export ENV_NAME="/path/to/your/python/env"
export MODEL_PATH="/path/to/your/Qwen2-VL-2B-Instruct"
export OUT_NAME="Qwen2-VL-2B-EXP"
export TRAIN_DATA="/path/to/your/train/data"
export EVAL_DATA="/path/to/your/val/data"
export EVAL_SAVE_NAME="eval_result.json"
export OUTDIR="/path/to/your/out_dir"
export WORKING_PATH="path/to/your/AlphaDrive"

cd ${WORKING_PATH}


# setup environments
echo "Setup environments..."
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

mkdir -p ${OUTDIR}


echo "Training Process..."
cd src/r1-v


$ENV_NAME/bin/torchrun --nproc_per_node="8" \
    --nnodes="2" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir $OUTDIR/$OUT_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $TRAIN_DATA \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --reward_funcs "plan_speed_reward" "plan_path_reward" "plan_format_reward" \
    --num_train_epochs 1 \
    --run_name $OUT_NAME \
    --save_steps 1000 \
    --save_only_model true \
    --num_generations 2   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  


echo "Validation Process..."
cd ${WORKING_PATH}
$ENV_NAME/bin/python eval_tools/qwen2vl_plan_cmd_eval_grpo.py \
    --eval_data_path $EVAL_DATA \
    --model_path $OUTDIR/$OUT_NAME \
    --save_path $OUTDIR/$OUT_NAME/$EVAL_SAVE_NAME
