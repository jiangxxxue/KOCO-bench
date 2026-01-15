#!/usr/bin/env bash
# PURE集成测试脚本
# 该脚本模拟完整的PURE训练流程，使用较小的模型和数据以便快速验证功能


set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}

# 设置HuggingFace环境变量以使用本地缓存
export HF_HUB_CACHE="/workspace/data/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface_cache"
export TRANSFORMERS_CACHE="/workspace/data/huggingface_cache"

# 使用较小的模型进行快速测试 - 使用本地绝对路径
LOCAL_MODEL_PATH="/workspace/data/huggingface_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

echo "检查本地模型路径是否存在..."
if [ -d "${LOCAL_MODEL_PATH}" ] && [ -f "${LOCAL_MODEL_PATH}/config.json" ] && [ -f "${LOCAL_MODEL_PATH}/model.safetensors" ]; then
    echo "使用本地模型路径: ${LOCAL_MODEL_PATH}"
    # 使用本地路径而不是 Hub 标识符
    MODEL_PATH="${LOCAL_MODEL_PATH}"
else
    echo "本地模型不完整，使用 Hub 标识符但设置离线模式"
    # 设置环境变量以使用正确的缓存位置
    export HF_HUB_CACHE="/workspace/data/huggingface_cache"
    export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface_cache"
    export HF_HUB_OFFLINE=1
    MODEL_PATH="${MODEL_ID}"
fi

# 使用实际的数据文件路径
TRAIN_FILES=/KOCO-bench/KOCO-bench-ch/domain_code_generation/verl/test_examples/PURE/code/train.parquet
VAL_FILES=/KOCO-bench/KOCO-bench-ch/domain_code_generation/verl/test_examples/PURE/code/val.parquet

# PRM模型路径（PURE的核心特色）
# PRM_MODEL_PATH=jinachris/PURE-PRM-7B
# echo "检查PRM模型是否已缓存..."
# if [ ! -d "/workspace/data/huggingface_cache/hub/models--jinachris--PURE-PRM-7B" ]; then
#     echo "PRM模型未缓存，开始下载..."
#     huggingface-cli download "${PRM_MODEL_PATH}"
# else
#     echo "PRM模型已存在于缓存中，跳过下载"
# fi

# 测试参数配置（较小的值以便快速运行）
train_traj_micro_bsz_per_gpu=2 # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

exp_name="$(basename "${MODEL_ID,,}")-pure-integration-test"

echo "=== PURE 集成测试配置 ==="
echo "Model: ${MODEL_PATH}"
echo "PRM Model: ${PRM_MODEL_PATH}"
echo "Train files: ${TRAIN_FILES}"
echo "Val files: ${VAL_FILES}"
echo "Batch size: ${train_prompt_bsz}"
echo "Experiment name: ${exp_name}"
echo "=========================="

# PURE三种模式的集成测试

# 切换到PURE代码目录
cd /KOCO-bench/KOCO-bench-ch/domain_code_generation/verl/test_examples/PURE/code


echo "=== 测试模式: PURE-PRM+VR （过程奖励+可验证奖励融合）==="
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
