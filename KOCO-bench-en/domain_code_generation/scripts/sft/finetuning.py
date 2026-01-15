#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetuning.py — VERL 代码库的继续预训练（NTP）
目标：让模型学习该仓库的语料分布、API习惯与编码风格，提升补全/读写稳定性。

要点：
- 直接用代码文本做 next-token prediction（无对话模板）
- 长文本：return_overflowing_tokens=True + stride，并进行 packing/打包到固定 block_size
- 训练稳定：bf16（优先）、梯度裁剪、可选 Deepspeed、gradient checkpointing
- 数据：JSONL 每行来自你的构建器，至少包含 "content"；若有 "file_type"/"file_path" 会用于筛选/可选注释

示例运行（单机）：
python finetuning.py \
  --model_name_or_path /home/you/models/Qwen2.5-Coder-7B-Instruct \\
  --dataset_path ../data/verl/verl_training_dataset.jsonl \\
  --output_dir ../models/qwen25-coder7b-verl-ntp \\
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true --fp16 false --tf32 true \
  --optim adamw_torch \
  --gradient_checkpointing true \
  --remove_unused_columns false

若使用 Deepspeed：
  …… 同上再加： --deepspeed ds_config.json
"""

import os
import sys
import json
import math
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import wandb

# -------------------- 日志 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verl-ntp")

# -------------------- 参数 --------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        metadata={"help": "预训练模型名称或本地路径"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "训练序列长度（打包后 block_size）"},
    )
    add_file_path_header: bool = field(
        default=False,
        metadata={"help": "是否在样本首行添加文件路径注释（默认 False）"},
    )
    keep_file_types: str = field(
        default="python,shell,yaml,markdown",
        metadata={"help": "保留的 file_type 白名单，逗号分隔；为空表示不过滤"},
    )

@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "训练数据集 JSONL 路径"})
    val_split_ratio: float = field(default=0.1, metadata={"help": "验证集比例（0~1）"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "最多读取的样本数（调试用）"})
    stride_fraction: float = field(
        default=0.125,
        metadata={"help": "滑窗重叠比例（max_seq_length * stride_fraction）"},
    )

@dataclass
class TrainingArguments2(TrainingArguments):
    use_wandb: bool = field(default=False, metadata={"help": "是否启用 wandb"})
    project_name: str = field(default="verl-ntp", metadata={"help": "wandb 项目名"})

# -------------------- 数据处理 --------------------
class VERLDataProcessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 1024,
        add_file_path_header: bool = False,
        keep_file_types: Optional[List[str]] = None,
        stride_tokens: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.block_size = max_seq_length
        self.add_file_path_header = add_file_path_header
        self.keep_file_types = set(t.strip() for t in keep_file_types) if keep_file_types else None
        self.stride_tokens = stride_tokens if stride_tokens is not None else max(1, max_seq_length // 8)

    def _comment_prefix(self, file_type: str) -> str:
        if file_type in {"python", "shell", "yaml", "toml"}:
            return "# "
        elif file_type in {"javascript", "typescript", "cpp", "java", "c", "go", "rust"}:
            return "// "
        return "# "

    def _make_text(self, sample: Dict[str, Any]) -> Optional[str]:
        # 读取 content
        content = sample.get("content", None)
        if not content or not str(content).strip():
            return None

        # 过滤 file_type（若提供）
        if self.keep_file_types and "file_type" in sample:
            if sample["file_type"] not in self.keep_file_types:
                return None

        if not self.add_file_path_header:
            return content

        file_path = sample.get("file_path", "unknown")
        file_type = sample.get("file_type", "unknown")
        prefix = self._comment_prefix(file_type)
        header = f"{prefix}File: {file_path}\n"
        return header + content

    def load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> Dataset:
        logger.info(f"Loading JSONL: {dataset_path}")
        samples: List[Dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = self._make_text(obj)
                    if text is None:
                        continue
                    samples.append({"text": text})
                    if max_samples and len(samples) >= max_samples:
                        break
                except Exception as e:
                    logger.warning(f"Skip line {ln}: {e}")
                    continue
        if not samples:
            raise ValueError("Empty dataset after filtering.")
        logger.info(f"Loaded {len(samples)} usable samples.")
        return Dataset.from_list(samples)

    # 先分词（允许溢出为多窗）
    def tokenize_fn(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return self.tokenizer(
            examples["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=self.block_size,
            padding=False,
            return_overflowing_tokens=True,
            stride=self.stride_tokens,
        )

    # 再打包为固定长度 block
    def group_texts(self, examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        # 把一个 batch 的若干段拼接为长序列
        concatenated: List[int] = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)
        total_length = (len(concatenated) // self.block_size) * self.block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        input_ids = [concatenated[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
        labels = [ids[:] for ids in input_ids]
        return {"input_ids": input_ids, "labels": labels}

# -------------------- 训练器 --------------------
class VERLFineTuner:
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments2):
        self.margs = model_args
        self.dargs = data_args
        self.targs = training_args
        self.tokenizer = None
        self.model = None

    def setup_wandb(self):
        if self.targs.use_wandb:
            wandb.init(
                project=self.targs.project_name,
                name=f"verl-ntp-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={**vars(self.margs), **vars(self.dargs), **self.targs.to_dict()},
            )

    def load_model_and_tokenizer(self):
        logger.info(f"Loading tokenizer & model from: {self.margs.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.margs.model_name_or_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # torch dtype 决策：优先 bf16（若开启 & 支持），否则 fp16（若开启 & 支持），否则 fp32
        def decide_dtype() -> torch.dtype:
            if self.targs.bf16 and torch.cuda.is_available():
                return torch.bfloat16
            if self.targs.fp16 and torch.cuda.is_available():
                return torch.float16
            return torch.float32

        torch_dtype = decide_dtype()

        # 检测是否使用 DeepSpeed（通过 --deepspeed 参数启动）
        use_deepspeed = self.targs.deepspeed is not None
        
        if use_deepspeed:
            # DeepSpeed ZeRO-3：不使用 device_map，让 DeepSpeed 管理模型分片
            self.model = AutoModelForCausalLM.from_pretrained(
                self.margs.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        else:
            # 非 DeepSpeed 模式：使用 device_map="auto" 自动分配设备
            self.model = AutoModelForCausalLM.from_pretrained(
                self.margs.model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        # 可选：梯度检查点
        if self.targs.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def prepare_datasets(self):
        keep_types = [t for t in self.margs.keep_file_types.split(",") if t.strip()] if self.margs.keep_file_types else None
        stride_tokens = max(1, int(self.margs.max_seq_length * self.dargs.stride_fraction))

        processor = VERLDataProcessor(
            tokenizer=self.tokenizer,
            max_seq_length=self.margs.max_seq_length,
            add_file_path_header=self.margs.add_file_path_header,
            keep_file_types=keep_types,
            stride_tokens=stride_tokens,
        )

        full = processor.load_dataset(self.dargs.dataset_path, self.dargs.max_samples)

        if 0.0 < self.dargs.val_split_ratio < 1.0:
            split = full.train_test_split(test_size=self.dargs.val_split_ratio, seed=42)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            train_ds, eval_ds = full, None

        # 分词 + 打包
        logger.info("Tokenizing (sliding windows) …")
        train_tok = train_ds.map(
            processor.tokenize_fn,
            batched=True,
            remove_columns=train_ds.column_names,
            desc="Tokenize train",
        )
        if eval_ds is not None:
            eval_tok = eval_ds.map(
                processor.tokenize_fn,
                batched=True,
                remove_columns=eval_ds.column_names,
                desc="Tokenize eval",
            )
        else:
            eval_tok = None

        logger.info("Packing fixed-length blocks …")
        train_blk = train_tok.map(
            processor.group_texts,
            batched=True,
            remove_columns=train_tok.column_names,  # 移除旧的 attention_mask 等列
            desc="Pack train",
        )
        if eval_tok is not None:
            eval_blk = eval_tok.map(
                processor.group_texts,
                batched=True,
                remove_columns=eval_tok.column_names,  # 移除旧的 attention_mask 等列
                desc="Pack eval",
            )
        else:
            eval_blk = None

        return train_blk, eval_blk

    def train(self):
        self.setup_wandb()
        self.load_model_and_tokenizer()
        train_ds, eval_ds = self.prepare_datasets()

        # collator：Causal LM（会对 pad 位置置 -100）
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=self.model,
            args=self.targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )

        train_result = trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.targs.output_dir)
        if self.targs.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        logger.info(f"Training done. final_loss={getattr(train_result, 'training_loss', None)}")
        return train_result

# -------------------- main --------------------
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments2))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        margs, dargs, targs = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        margs, dargs, targs = parser.parse_args_into_dataclasses()

    # 输出目录兜底
    if not targs.output_dir:
        targs.output_dir = f"./verl_ntp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(targs.output_dir, exist_ok=True)

    # TF32（如果传了）
    if hasattr(targs, "tf32") and targs.tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    logger.info(
        f"cfg: model={margs.model_name_or_path}, "
        f"seq_len={margs.max_seq_length}, stride≈{int(margs.max_seq_length * dargs.stride_fraction)}, "
        f"add_header={margs.add_file_path_header}, keep_types={margs.keep_file_types}"
    )

    try:
        runner = VERLFineTuner(margs, dargs, targs)
        runner.train()
        # 额外保存配置
        cfg = {
            "model_args": vars(margs),
            "data_args": vars(dargs),
            "training_args": targs.to_dict(),
            "saved_at": datetime.now().isoformat(),
        }
        with open(os.path.join(targs.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info(f"Model & tokenizer saved to: {targs.output_dir}")
        return 0
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
