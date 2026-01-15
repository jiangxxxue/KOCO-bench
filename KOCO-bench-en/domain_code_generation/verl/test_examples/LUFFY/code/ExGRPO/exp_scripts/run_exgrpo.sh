set -x

# NOTE: change to your root dir
ROOT= # Your Root Path

ray stop

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

# NOTE: change to your model dir
export MODEL_PATH= # Your Backbone Model Path

export DATA_DIR=$ROOT/data/
export EXP_NAME=ExGRPO 
export WANDB_PROJECT="exgrpo"

# NOTE: change to your save path
export SAVE_PATH=$ROOT/checkpoints/$WANDB_PROJECT/$EXP_NAME

cd $ROOT/exgrpo/verl/

OMP_NUM_THREADS=32 python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    trainer.experience=True \
    trainer.exp_metric=ent \
    trainer.exp_select_mode=argmin \
    trainer.exp_sample_range=all \
    trainer.exp_bucket_manager=normal \
    trainer.experience_ratio=0.5 \
    trainer.experience_start_sratio=0.35 \
    trainer.exp_is_correct=True \
    trainer.experience_lbound=0 \
    trainer.experience_rbound=7 \
    data.train_files=$DATA_DIR/openr1.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm_exp \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$SAVE_PATH" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    +actor_rollout_ref.actor.skip_grad_bound=500 \
    trainer.total_training_steps=701 "${@:1}" \
