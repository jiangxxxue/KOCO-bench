set -x

critic_model="Qwen/Qwen2.5-Coder-32B-Instruct"
data_path="sft_32b.jsonl"
output_path="qwen25-sft-1epoch-32b"

accelerate launch --config_file ctrl/sft/zero3_accelerate.yaml --main_process_port 7788 \
    scripts/run_sft.py \
    --model_name_or_path $critic_model \
    --data_path $data_path \
    --output_dir $output_path \
    --instruction_field "critic_instruction" \
    --output_field "critique" \
    --num_train_epochs 1 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --bf16 True
