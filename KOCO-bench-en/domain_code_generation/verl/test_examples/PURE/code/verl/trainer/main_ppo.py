# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true', 
                    'NCCL_DEBUG': 'WARN',
                    "NCCL_PXN_DISABLE": "1",
                    "NCCL_ALGO": "^Ring",
                    "NCCL_NET_OVERHEAD": "1000000",
                    "CUDA_LAUNCH_BLOCKING": "1",
                }
            },
            # debug mode
            local_mode=config.trainer.get('debug', False),
        )

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf, open_dict

    from verl.utils.fs import copy_to_local

    # modify chat template to avoid string escaping
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.data.chat_template = '{}\n\nPlease reason step by step with steps separated by "\n\n", and put your final answer within \\boxed{{}}.'
        # debug mode
        if config.trainer.get('debug', False):
            config.trainer.logger = ['console']
            exp_name = config.trainer.experiment_name
            config.trainer.experiment_name = exp_name if exp_name.lower().startswith('debug_') else 'debug_' + exp_name
            config.trainer.val_before_train = False
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.type == "orm":
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
        elif config.reward_model.type == "prm":
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import (
                    ProcessRewardModelWorker as RewardModelWorker,
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == 'blank':
        from verl.workers.reward_manager import BlankRewardManager
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = BlankRewardManager
    else:
        raise NotImplementedError
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=2, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    if reward_manager_name == 'blank':
        val_reward_fn = PrimeRewardManager(tokenizer=tokenizer, num_examine=2, compute_score=compute_score)
    else:
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=2, compute_score=compute_score)
    
    curriculum_learning_fn = None
    if config.curriculum_learning.get('enable', False):
        method = config.curriculum_learning.get('fn_name', 'remove_extreme')
        fn_params = config.curriculum_learning.get('fn_params', None)
        fn_params = dict(fn_params) if fn_params is not None else {}

        if method == 'interquartile':
            from verl.workers.curriculum_learning import InterquartileCurriculumLearning
            curriculum_learning_fn = InterquartileCurriculumLearning(**fn_params)
        elif method == 'remove_extreme':
            from verl.workers.curriculum_learning import RemoveExtremeRewardCurriculumLearning
            curriculum_learning_fn = RemoveExtremeRewardCurriculumLearning()
        else:
            raise NotImplementedError

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            curriculum_learning_fn=curriculum_learning_fn,)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
