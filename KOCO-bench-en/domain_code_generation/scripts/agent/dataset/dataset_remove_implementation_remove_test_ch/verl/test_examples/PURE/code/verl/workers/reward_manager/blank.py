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

from verl import DataProto
import torch


class BlankRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine=1, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        response_ids = data.batch['responses']
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['answer'] for data_item in data]
        default_data_sources = ['numina_aops_forum'] * len(sequences_str)  # tricky: force to use prime_math.compute_score
        data_sources = data.batch.get('data_sources', default_data_sources)

        rm_scores = data.batch.get('rm_scores', None)
        if rm_scores is not None:
            # orm score > 0 and vr = 1
            # or orm score <= 0 and vr = 0
            # then orm_match_vr is True
            outcome_reward = rm_scores.sum(-1)

        example_idx = 1
        for i in range(len(data)):
            output_str = f"{'Rollout Example ' + str(example_idx):#^{50}}\n"
            data_source = data_sources[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                
                output_str += f"Question:\n{repr(prompt_str[i])}\n"  # Unescaped prompts to check chat_template
                steps = sequences_str[i].split('\n\n')
                output_str += f"Rollout:\n{steps}\n"
                output_str += f"Num of steps: {len(steps)}\n"
                output_str += f"Ground-truth: {ground_truth[i]}\n"
                if rm_scores is not None:
                    output_str += f"Outcome reward: {outcome_reward[i].item()}\n"
                print(output_str.rstrip())
                example_idx += 1
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        output = DataProto.from_dict({
            "verifiable_rewards": reward_tensor.sum(-1),
            "reward_fn_scores": reward_tensor,
            })

        return output