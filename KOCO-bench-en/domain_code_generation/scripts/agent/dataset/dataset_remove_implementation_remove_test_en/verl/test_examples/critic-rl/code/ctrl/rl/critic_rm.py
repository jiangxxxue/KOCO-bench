# Copyright (2025) critic-rl Authors 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.
"""
A function reward model for CodeContests (critic training)
"""
import ast
import asyncio
import re
from typing import Any, List

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from transformers import PreTrainedTokenizer
from verl import DataProto

from ctrl.gen.prompt import extract_critique_from_xml, get_prompter
from ctrl.eval.sandbox_utils import RunCodeRequest, SubmitRequest, TestConfig, submit_to_sandbox

RUN_TIMEOUT = 10
MAX_REQUESTS = 256








class RewardModelForCritic(object):
    def __init__(
        self,
        file,  # train_file or test_file
        tokenizer: PreTrainedTokenizer,
        num_examine=0,
        run_all_cases=True,
    ):
        self.tokenizer = tokenizer

        # load train file or test file
        self.file = file

        self.num_examine = num_examine

        if isinstance(self.file, str):
            self.dataset = pd.read_parquet(self.file)
        else:
            self.dataset = pd.concat([pd.read_parquet(f) for f in self.file])

        self.dataset["proxy_id"] = self.dataset["id"]
        self.id_to_infos = self.dataset.set_index("proxy_id").to_dict(orient="index")
        self.run_all_cases = run_all_cases

        # implement code hash so that we can reduce sandbox usage
        self.code_to_reward = {}

    def parse_query(self, ids: List[str]) -> List[pd.Series]:
        init_messages, rows = [], []

        for id in ids:
            row = self.id_to_infos[id]
            prompter = get_prompter(row["prompter_type"])

            init_message = [
                {
                    "role": "user",
                    "content": prompter.get_gen_prompt(row["problem"]),
                },
                {
                    "role": "assistant",
                    "content": row["solution"],
                },
            ]

            init_messages.append(init_message)
            rows.append(row)
        return init_messages, rows


    def parse_response(self, data: DataProto) -> DataProto:
        ids = []
        generations = []
        valid_response_lengths = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids)
            # remove <eos> token
            response_str = response_str.replace(self.tokenizer.eos_token, "")

            if i < self.num_examine:
                print(response_str)

            # get test case
            problem_id = data_item.non_tensor_batch["problem_id"]
            ids.append(problem_id)
            generations.append(response_str)
            valid_response_lengths.append(valid_response_length)

        return ids, generations, valid_response_lengths

    async def reward_revision(
        self, critique: str, revision: str, sample: dict, semaphore: asyncio.Semaphore
    ) -> float:
        if ("Overall judgment: Correct" not in critique) and (
            "Overall judgment: Incorrect" not in critique
        ):
            return 0.0

        revised_code = sanitize(revision)
        success = await self.get_sandbox_response(revised_code, sample, semaphore)
        reward = success

        return reward


    async def get_sandbox_response(
        self, revision: str, sample: dict, semaphore: asyncio.Semaphore
    ):
        # check if the code is already in the cache
        code_str = desanitize(revision).strip()
        normalized_code_str = normalize_code(code_str)
        uts = sample["selected_uts"]
        run_timeout = sample.get("time_limit", RUN_TIMEOUT)
        if np.isnan(run_timeout):
            run_timeout = RUN_TIMEOUT
        prompter_type = sample["prompter_type"]

        if (normalized_code_str, str(uts)) in self.code_to_reward:
            return self.code_to_reward[(normalized_code_str, str(uts))]

        if prompter_type == "mbppplus":
            if not isinstance(uts, str):  # apps
                ut_list = list(uts)

                if self.run_all_cases:
                    reqs = [
                        RunCodeRequest(
                            **{
                                "code": code_str + "\n" + ut,
                                "language": "python",
                                "run_timeout": run_timeout,
                            }
                        )
                        for ut in ut_list
                    ]
                else:
                    reqs = [
                        RunCodeRequest(
                            **{
                                "code": code_str + "\n" + "\n".join(ut_list),
                                "language": "python",
                                "run_timeout": run_timeout,
                            }
                        )
                    ]
            else:  # mbppplus
                reqs = [
                    RunCodeRequest(
                        **{
                            "code": code_str + "\n" + uts,
                            "language": "python",
                            "run_timeout": run_timeout,
                        }
                    )
                ]
            all_res = await asyncio.gather(
                *[submit_to_sandbox(req, semaphore) for req in reqs]
            )
            reward = np.mean(
                [float(res.dict()["status"] == "Success") for res in all_res]
            )
        elif prompter_type == "livecodebench":
            provided_data = {k: sample[k] for k in ["id", "content", "labels", "test"]}
            req = SubmitRequest(
                dataset="dataset_id",
                id=0,
                config=TestConfig(
                    dataset_type="LiveCodeBenchDataset",
                    provided_data=provided_data,
                    run_timeout=run_timeout,
                    extra={"run_all_cases": self.run_all_cases},
                ),
                completion=revision,
            )
            res = await submit_to_sandbox(req, semaphore)
            reward = np.mean([t["passed"] for t in res.dict()["tests"]])  # float: 0-1
        elif prompter_type == "code_contests":  # oj
            provided_data = {
                "test": uts,
            }

            req = SubmitRequest(
                dataset="code_contests",
                id=0,
                config=TestConfig(
                    language="python",
                    dataset_type="CommonOJDataset",
                    provided_data=provided_data,
                    extra={"run_all_cases": self.run_all_cases},
                    run_timeout=run_timeout,
                ),
                completion=revision,
            )
            res = await submit_to_sandbox(req, semaphore)
            reward = (
                np.mean([t["passed"] for t in res.dict()["tests"]])
                if self.run_all_cases
                else float(res.accepted)
            )
        else:
            raise ValueError(f"Unknown prompter type: {prompter_type}")

        # update cache
        self.code_to_reward[(normalized_code_str, str(uts))] = reward

        return reward
