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
import pathlib
import random

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json

from open_r1.vlm_modules import *

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()    



tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    question_template: Optional[str] = field(
        default="scoring",
        metadata={
            "help": "Choose scoring or comparing question"
        },
    )


def extract_first_number(model_answer):
    match = re.search(r'-?\d+(\.\d+)?', model_answer)
    if match:
        return float(match.group())
    else:
        return random.randint(1, 5)








reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.question_template)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    training_args.max_completion_length = 256
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
    
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder, accu_reward_method in zip(data_files, image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    if isinstance(item['image'], str):
                        # Store image path instead of loading the image
                        item['image_path'] = [os.path.join(image_folder, item['image'])]
                        del item['image'] # remove the image column so that it can be loaded later
                    elif isinstance(item['image'], list):
                        # if the image is a list, then it is a list of images (for multi-image input)
                        item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                        del item['image'] # remove the image column so that it can be loaded later
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                # Remove immediate image loading
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                # Handle solution that could be a float or string
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    # If it's a float or other non-string type, keep it as is
                    item['solution'] = str(solution_value)
                
                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'dataset_name': example['dataset_name'],
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'dataset_name': example['dataset_name'],
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
