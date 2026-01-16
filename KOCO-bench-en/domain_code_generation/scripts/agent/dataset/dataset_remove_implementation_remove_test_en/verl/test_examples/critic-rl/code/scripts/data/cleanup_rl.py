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
"""Convert data to prompt data for verl"""

import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ctrl.gen.prompt import get_prompter


def get_fields(prompter_type):
    if prompter_type == "code_contests":
        return dict(
            problem_field="problem",
            id_field="task_id",
            test_field="all_uts",
        )
    if prompter_type == "livecodebench":
        return dict(
            id_field="id",
            problem_field="problem",
            test_field="test",
        )
    if prompter_type == "mbppplus":
        return dict(
            problem_field="problem",
            test_field="test",
            id_field="task_id",
        )
    if prompter_type == "assert" or prompter_type == "io":
        return dict(
            problem_field="problem",
            test_field="all_uts",
            id_field="task_id",
        )


def main(args):
    input_df = pd.read_json(args.input, lines=True).dropna(subset=["solution"])
    dataset_df = pd.read_json(args.dataset_path, lines=True)

    # join two with `task_id`
    fields = get_fields(args.prompter_type)
    df = input_df.merge(dataset_df, left_on="task_id", right_on=fields["id_field"])

    if args.prompter_type == "assert":
        args.prompter_type = "mbppplus"
    if args.prompter_type == "io":
        args.prompter_type = "code_contests"
    df["prompter_type"] = args.prompter_type

    # format context messages
    prompter = get_prompter(args.prompter_type)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def format_prompt(row):
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": prompter.get_critique_prompt(
                    row[fields["problem_field"]], row["sanitized_solution"]
                ),
            }
        )
        if args.keep_chat:
            return messages
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    df["prompt"] = df.apply(format_prompt, axis=1)
    try:
        df["success_float"] = df.metadata.apply(
            lambda x: np.mean([t["passed"] for t in x["result"]["tests"]])
        )
    except:
        df["success_float"] = df.success

    # filter long prompts
    if args.max_prompt_len is not None:
        df = df[
            df.prompt.apply(lambda x: len(tokenizer.tokenize(x)) < args.max_prompt_len)
        ]

    df["selected_uts"] = df[fields["test_field"]]
    df["problem_id"] = [f"{args.input}/{i}" for i in np.arange(len(df))]
    df["id"] = df["problem_id"].apply(str)

    df = df.drop(["critique", "metadata", fields["test_field"]], axis=1)

    df.to_parquet(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file name")
    parser.add_argument("output", type=str, help="output file name")
    parser.add_argument("--split_name", type=str, required=True, help="split name")
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")

    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument("--max_prompt_len", type=int, default=None)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    parser.add_argument("--prompter_type", type=str, default="code_contests")
    parser.add_argument("--keep_chat", action="store_true")
    args = parser.parse_args()

    main(args)
