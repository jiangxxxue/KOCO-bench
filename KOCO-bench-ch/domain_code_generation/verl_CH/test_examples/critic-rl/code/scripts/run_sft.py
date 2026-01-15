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
import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed
import transformers
from datasets import load_dataset
from transformers import Trainer

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"


def build_instruction_prompt(instruction: str):
    return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(
        instruction.strip()
    ).lstrip()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-Coder-7B-Instruct")


@dataclass
class DataArguments:
    data_path: List[str] = field(
        default_factory=list, metadata={"help": "Paths to the training data."}
    )
    instruction_field: str = field(
        default="instruction", metadata={"help": "The field name for the instruction"}
    )
    output_field: str = field(
        default="output", metadata={"help": "The field name for the output"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(  # NOTE: ignore this
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # max_length=tokenizer.model_max_length,
            # truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank == 0:
        print("=" * 100)
        print(training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    def train_tokenize_function(examples, tokenizer):
        sources = [
            build_instruction_prompt(instruction)
            for instruction in examples[data_args.instruction_field]
        ]
        targets = [
            f"{output.strip()}{tokenizer.eos_token}"
            for output in examples[data_args.output_field]
        ]
        data_dict = preprocess(sources, targets, tokenizer)
        return data_dict

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16
    )

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))

    raw_train_datasets = load_dataset(
        "json",
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir,
    )
    if training_args.local_rank > 0:
        torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer},
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()

    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(
                f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}."
            )
            print(
                f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}."
            )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
