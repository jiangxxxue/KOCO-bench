export ENV_NAME="/path/to/your/python/env"
export OUTDIR="/path/to/your/output/directory"
export OUT_NAME="Qwen2-VL-2B-EXP"
export EVAL_DATA="/path/to/your/val/data"
export EVAL_SAVE_NAME="eval_result.json"

echo "Validation Process..."
$ENV_NAME/bin/python eval_tools/qwen2vl_plan_cmd_eval_grpo.py \
    --eval_data_path $EVAL_DATA \
    --model_path $OUTDIR/$OUT_NAME \
    --save_path $OUTDIR/$OUT_NAME/$EVAL_SAVE_NAME