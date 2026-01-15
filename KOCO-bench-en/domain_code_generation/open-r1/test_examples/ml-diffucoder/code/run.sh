export E2B_API_KEY=e2b_yourkey_here

nohup python scripts/e2b_router.py > e2b_router.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 8 \
    src/open_r1/grpo.py --config recipes/config_coupled_code.yaml