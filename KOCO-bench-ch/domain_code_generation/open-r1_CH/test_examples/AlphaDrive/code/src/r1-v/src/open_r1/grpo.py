# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'plan_speed_reward', 'plan_path_reward', 'plan_format_reward'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["plan_speed_reward", "plan_path_reward", "plan_format_reward"],
        metadata={"help": "List of reward functions. Possible values: 'plan_speed_reward', 'plan_path_reward', 'plan_format_reward'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def plan_speed_reward(completions,
                      solution,
                      diversity_weight=0.4,
                      complexity_weights=None,
                      **kwargs):
    """
    planning speed reward function.
    """
    if complexity_weights is None:
        complexity_weights = {
            "ACCELERATE": 0.9, "DECELERATE": 1.0, "STOP": 1.0, "KEEP": 0.8,
        }
    
    rewards = []
    global_decision_count = Counter()

    for completion, sol in zip(completions, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        if not sol_match:
            rewards.append(0)
            continue
        ground_truth_words = set(sol_match.group(1).strip().split(', '))
        ground_truth_words = {word for word in ground_truth_words if word in complexity_weights}

        match = re.search(r"<answer>(.*?)</answer>", completion[0]["content"])
        if match:
            content = match.group(1).strip()
        else:
            content = completion[0]["content"].replace('<answer>', '').replace('</answer>', '')

        content_word_list = [re.sub(r'[^\w]', '', word) for word in content.split(', ') if word in complexity_weights]
        content_words = set(content_word_list)
        global_decision_count.update(content_words)

        true_positives = len(content_words & ground_truth_words)
        false_positives = len(content_words - ground_truth_words)
        false_negatives = len(ground_truth_words - content_words)

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        if true_positives == 0 and false_positives == 0 and false_negatives == 0:
            f1_score = 0  # no match
        else:
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        complexity_factor = sum(complexity_weights[word] for word in content_words) / (len(content_words) + 1e-6)
        
        diversity_factor = [True if global_decision_count[word] == 1 else False for word in content_words] 
        diversity_factor = True if all(diversity_factor) else False
        diversity_factor = diversity_weight if diversity_factor else -diversity_weight

        reward = f1_score * complexity_factor + diversity_factor

        rewards.append(reward)

    return rewards


def plan_path_reward(completions,
                     solution,
                     diversity_weight=0.4,
                     complexity_weights=None,
                     **kwargs):
    """
    planning path reward function.
    """
    if complexity_weights is None:
        complexity_weights = {
            "LEFT_TURN": 1.0, "RIGHT_TURN": 1.0,
            "LEFT_CHANGE": 1.0, "RIGHT_CHANGE": 1.0, "STRAIGHT": 0.8
        }
    
    rewards = []
    global_decision_count = Counter()

    for completion, sol in zip(completions, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        if not sol_match:
            rewards.append(0)
            continue
        ground_truth_words = set(sol_match.group(1).strip().split(', '))
        ground_truth_words = {word for word in ground_truth_words if word in complexity_weights}

        match = re.search(r"<answer>(.*?)</answer>", completion[0]["content"])
        if match:
            content = match.group(1).strip()
        else:
            content = completion[0]["content"].replace('<answer>', '').replace('</answer>', '')

        content_word_list = [re.sub(r'[^\w]', '', word) for word in content.split(', ') if word in complexity_weights]
        content_words = set(content_word_list)
        global_decision_count.update(content_words)

        true_positives = len(content_words & ground_truth_words)
        false_positives = len(content_words - ground_truth_words)
        false_negatives = len(ground_truth_words - content_words)

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        if true_positives == 0 and false_positives == 0 and false_negatives == 0:
            f1_score = 0  # no match
        else:
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        complexity_factor = sum(complexity_weights[word] for word in content_words) / (len(content_words) + 1e-6)

        diversity_factor = [True if global_decision_count[word] == 1 else False for word in content_words] 
        diversity_factor = True if all(diversity_factor) else False
        diversity_factor = diversity_weight if diversity_factor else -diversity_weight

        reward = f1_score * complexity_factor + diversity_factor

        rewards.append(reward)

    return rewards


def plan_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # check if answer format is <think>xxx</think>\n<answer>xxx</answer>
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    return [1.0 if match else 0.0 for match in matches]



reward_funcs_registry = {
    "plan_format_reward": plan_format_reward,
    "plan_speed_reward": plan_speed_reward,
    "plan_path_reward": plan_path_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    # QUESTION_TEMPLATE = "{Question}  Output the final answer in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        # {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
