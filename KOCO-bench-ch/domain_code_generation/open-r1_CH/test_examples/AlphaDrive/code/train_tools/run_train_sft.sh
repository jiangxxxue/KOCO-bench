#!/bin/bash

# dynamic settings
export ENV_NAME="/path/to/your/python/env"
export MODEL_PATH="/path/to/your/Qwen2-VL-2B-Instruct"
export OUT_NAME="Qwen2-VL-2B-EXP"
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
$ENV_NAME/bin/accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft.py --config src/r1-v/configs/qwen2vl_sft_config.yaml 


echo "Validation Process..."
$ENV_NAME/bin/python eval_tools/qwen2vl_plan_cmd_eval_sft.py \
    --eval_data_path $EVAL_DATA \
    --model_path $OUTDIR/$OUT_NAME \
    --save_path $OUTDIR/$OUT_NAME/$EVAL_SAVE_NAME
