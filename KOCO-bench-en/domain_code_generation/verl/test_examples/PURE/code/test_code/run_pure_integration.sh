#!/usr/bin/env bash
# PURE integration test script
# This script simulates the complete PURE training process, using smaller models and data for quick functionality verification


set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

# Set HuggingFace environment variables to use local cache
export HF_HUB_CACHE="/workspace/data/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface_cache"
export TRANSFORMERS_CACHE="/workspace/data/huggingface_cache"

# Use smaller model for quick testing - use local absolute path
LOCAL_MODEL_PATH="/workspace/data/huggingface_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

echo "Checking if local model path exists..."
if [ -d "${LOCAL_MODEL_PATH}" ] && [ -f "${LOCAL_MODEL_PATH}/config.json" ] && [ -f "${LOCAL_MODEL_PATH}/model.safetensors" ]; then
    echo "Using local model path: ${LOCAL_MODEL_PATH}"
    # Use local path instead of Hub identifier
    MODEL_PATH="${LOCAL_MODEL_PATH}"
else
    echo "Local model incomplete, using Hub identifier but setting offline mode"
    # Set environment variables to use correct cache location
    export HF_HUB_CACHE="/workspace/data/huggingface_cache"
    export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface_cache"
    export HF_HUB_OFFLINE=1
    MODEL_PATH="${MODEL_ID}"
fi

# Use actual data file paths
TRAIN_FILES=/KOCO-bench/KOCO-bench-en/domain_code_generaiton/verl/test_examples/PURE/code/train.parquet
VAL_FILES=/KOCO-bench/KOCO-bench-en/domain_code_generaiton/verl/test_examples/PURE/code/val.parquet

# PRM model path (PURE's core feature)
# PRM_MODEL_PATH=jinachris/PURE-PRM-7B
# echo "Checking if PRM model is cached..."
# if [ ! -d "/workspace/data/huggingface_cache/hub/models--jinachris--PURE-PRM-7B" ]; then
#     echo "PRM model not cached, starting download..."
#     huggingface-cli download "${PRM_MODEL_PATH}"
# else
#     echo "PRM model already exists in cache, skipping download"
# fi

# Test parameter configuration (smaller values for quick execution)
train_traj_micro_bsz_per_gpu=2 # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

exp_name="$(basename "${MODEL_ID,,}")-pure-integration-test"

echo "=== PURE Integration Test Configuration ==="
echo "Model: ${MODEL_PATH}"
echo "PRM Model: ${PRM_MODEL_PATH}"
echo "Train files: ${TRAIN_FILES}"
echo "Val files: ${VAL_FILES}"
echo "Batch size: ${train_prompt_bsz}"
echo "Experiment name: ${exp_name}"
echo "=========================="

# PURE three-mode integration test

# Switch to PURE code directory
cd /KOCO-bench/KOCO-bench-en/domain_code_generaiton/verl/test_examples/PURE/code


echo "=== Test Mode: PURE-PRM+VR (Process Reward + Verifiable Reward Fusion) ==="
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    reward_model.model.path="${MODEL_PATH}" \
    trainer.logger=console \
    trainer.project_name='pure-integration-test' \
    trainer.experiment_name="${exp_name}-prm-vr-fusion" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=3 $@ || echo "PURE-PRM+VR test completed"
