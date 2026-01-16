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
Using FSDPTrainer to run PPO code algorithm
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"

import hydra
import ray

from ctrl.rl.critic_fsdp_workers import ActorRolloutRefWorker, CriticWorker
from ctrl.rl.critic_rm import RewardModelForCritic
from ctrl.rl.critic_ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@hydra.main(config_path="rl_config", config_name="trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # only support fsdp for now
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
    from verl.single_controller.ray import RayWorkerGroup

    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    reward_fn = RewardModelForCritic(
        file=config.data.train_files,
        tokenizer=tokenizer,
        num_examine=0,
        run_all_cases=False,
    )
    val_reward_fn = RewardModelForCritic(
        file=config.data.val_files,
        tokenizer=tokenizer,
        num_examine=1,
        run_all_cases=False,
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
