# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


def evaluation_func_wrapper(evaluation_func, task, completion, reference, timeout):
    class TimeoutException(Exception):
        pass

    def alarm_handler(*args, **kwargs):
        raise TimeoutException("Function execution exceeded timeout.")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(int(timeout))
    try:
        return evaluation_func(task, completion, reference)
    finally:
        signal.alarm(0)


async def single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func_wrapper, evaluation_func, task, completion, reference, timeout)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['answer'] for data_item in data]
        default_data_sources = ['numina_aops_forum'] * len(sequences_str)  # tricky: force to use prime_math.compute_score
        data_sources = data.batch.get('data_sources', default_data_sources)

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             sequences_str,
                                             ground_truth,
                                             data_sources,
                                             num_processes=64))
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0. for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_str))]
        
        scores = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)

        # repeat punishment: if repeatness > 0.2, get 0 VR
        repeat_mask = data.batch.get('repeat_mask', None)
        if data.meta_info.get('repeatness_punishment', False) and repeat_mask is not None:
            repeat_mask = repeat_mask.to(device=scores.device, dtype=torch.bool)
            scores[repeat_mask] = 0
        
        # check whether outcome-level reward from RM matches GT-level VR
        rm_scores = data.batch.get('rm_scores', None)
        orm_match_vr = None
        if rm_scores is not None:
            # orm score > 0 and vr = 1
            # or orm score <= 0 and vr = 0
            # then orm_match_vr is True
            outcome_reward = rm_scores.sum(-1)
            predictions = outcome_reward.sign()
            predictions[predictions == -1] = 0
            orm_match_vr = predictions == scores
        
        example_idx = 1
        for i in range(len(data)):
            output_str = f"{'Rollout Example ' + str(example_idx):#^{50}}\n"
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                
                output_str += f"Question:\n{repr(prompt_str[i])}\n"  # Unescaped prompts to check chat_template
                steps = sequences_str[i].split('\n\n')
                output_str += f"Rollout:\n{steps}\n"
                output_str += f"Num of steps: {len(steps)}\n"
                output_str += f"Ground-truth: {ground_truth[i]}\n"
                output_str += f"Verifiable reward: {scores[i].item()}\n"
                if repeat_mask is not None:
                    output_str += f"Repeatness: {repeat_mask[i].item()}\n"
                if orm_match_vr is not None:
                    output_str += f"Outcome reward: {outcome_reward[i].item()}\n"
                    output_str += f"ORM match VR? {orm_match_vr[i].item()}\n"
                print(output_str.rstrip())
                example_idx += 1
        
        output = {
            "verifiable_rewards": scores,
            "reward_fn_scores": reward_tensor,
        }
        if orm_match_vr is not None:
            output["orm_match_vr"] = orm_match_vr.to(dtype=torch.float32)
        output = DataProto.from_dict(output)
        return output
