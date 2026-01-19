#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetuning_lora.py — LoRA fine-tuning for code repositories (NTP)
Based on finetuning.py, using PEFT LoRA for parameter-efficient fine-tuning

Key points:
- Uses PEFT library's LoRA implementation, training only a small number of parameters
- Maintains the same data processing pipeline as the original (tokenization, sliding windows, packing)
- Supports custom LoRA parameters: r, lora_alpha, lora_dropout, target_modules
- Lower memory usage and faster training

Example usage:
python finetuning_lora.py \
  --model_name_or_path /path/to/your/model \
  --dataset_path ./data/your_framework/training_dataset.jsonl \
  --output_dir ./models/your_framework-lora \
  --max_seq_length 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,v_proj,k_proj,o_proj
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

# PEFT library for LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import wandb

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("code-lora")

# -------------------- Arguments --------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        metadata={"help": "Pretrained model name or local path"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Training sequence length (block_size after packing)"},
    )
    add_file_path_header: bool = field(
        default=False,
        metadata={"help": "Whether to add file path comment at the beginning of sample (default False)"},
    )
    keep_file_types: str = field(
        default="python,shell,yaml,markdown",
        metadata={"help": "Whitelist of file_type to keep, comma-separated; empty means no filtering"},
    )

@dataclass
class LoraArguments:
    """LoRA related arguments"""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter for scaling"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"},
    )
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj",
        metadata={"help": "Module names to apply LoRA, comma-separated"},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use Rank-Stabilized LoRA"},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation)"},
    )

@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Training dataset JSONL path"})
    val_split_ratio: float = field(default=0.1, metadata={"help": "Validation set ratio (0~1)"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of samples to read (for debugging)"})
    stride_fraction: float = field(
        default=0.125,
        metadata={"help": "Sliding window overlap ratio (max_seq_length * stride_fraction)"},
    )

@dataclass
class TrainingArguments2(TrainingArguments):
    use_wandb: bool = field(default=False, metadata={"help": "Whether to enable wandb"})
    project_name: str = field(default="code-lora", metadata={"help": "wandb project name"})

# -------------------- Data Processing --------------------
class CodeDataProcessor:
    """Reuse original data processing logic"""
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
        # Read content
        content = sample.get("content", None)
        if not content or not str(content).strip():
            return None

        # Filter file_type (if provided)
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

    # Tokenize first (allow overflow to multiple windows)
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

    # Then pack into fixed-length blocks
    def group_texts(self, examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        # Concatenate multiple segments in a batch into a long sequence
        concatenated: List[int] = []
        for seq in examples["input_ids"]:
            concatenated.extend(seq)
        total_length = (len(concatenated) // self.block_size) * self.block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        input_ids = [concatenated[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
        labels = [ids[:] for ids in input_ids]
        return {"input_ids": input_ids, "labels": labels}

# -------------------- LoRA Trainer --------------------
class CodeLoraFineTuner:
    def __init__(
        self,
        model_args: ModelArguments,
        lora_args: LoraArguments,
        data_args: DataArguments,
        training_args: TrainingArguments2,
    ):
        self.margs = model_args
        self.largs = lora_args
        self.dargs = data_args
        self.targs = training_args
        self.tokenizer = None
        self.model = None

    def setup_wandb(self):
        if self.targs.use_wandb:
            wandb.init(
                project=self.targs.project_name,
                name=f"code-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    **vars(self.margs),
                    **vars(self.largs),
                    **vars(self.dargs),
                    **self.targs.to_dict(),
                },
            )

    def load_model_and_tokenizer(self):
        logger.info(f"Loading tokenizer & model from: {self.margs.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.margs.model_name_or_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # torch dtype decision
        def decide_dtype() -> torch.dtype:
            if self.targs.bf16 and torch.cuda.is_available():
                return torch.bfloat16
            if self.targs.fp16 and torch.cuda.is_available():
                return torch.float16
            return torch.float32

        torch_dtype = decide_dtype()

        # Load base model
        logger.info(f"Loading base model with dtype={torch_dtype}")
        # Note: Do not use device_map for LoRA training, let Trainer handle multi-GPU automatically
        # Using device_map="auto" will cause device inconsistency after PEFT adds LoRA layers
        self.model = AutoModelForCausalLM.from_pretrained(
            self.margs.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # Use standard attention to avoid flash-attention GLIBC issues
        )

        # Prepare LoRA configuration
        target_modules = [m.strip() for m in self.largs.target_modules.split(",") if m.strip()]
        logger.info(f"LoRA config: r={self.largs.lora_r}, alpha={self.largs.lora_alpha}, "
                   f"dropout={self.largs.lora_dropout}, target_modules={target_modules}")
        
        lora_config = LoraConfig(
            r=self.largs.lora_r,
            lora_alpha=self.largs.lora_alpha,
            lora_dropout=self.largs.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=self.largs.use_rslora,
            use_dora=self.largs.use_dora,
        )

        # Apply LoRA
        logger.info("Applying LoRA to model...")
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters statistics
        self.model.print_trainable_parameters()

        # Optional: gradient checkpointing
        if self.targs.gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

    def prepare_datasets(self):
        keep_types = [t for t in self.margs.keep_file_types.split(",") if t.strip()] if self.margs.keep_file_types else None
        stride_tokens = max(1, int(self.margs.max_seq_length * self.dargs.stride_fraction))

        processor = CodeDataProcessor(
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

        # Tokenization + packing
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
            remove_columns=train_tok.column_names,
            desc="Pack train",
        )
        if eval_tok is not None:
            eval_blk = eval_tok.map(
                processor.group_texts,
                batched=True,
                remove_columns=eval_tok.column_names,
                desc="Pack eval",
            )
        else:
            eval_blk = None

        # Print dataset statistics
        logger.info(f"Train dataset size: {len(train_blk)} samples")
        if eval_blk is not None:
            logger.info(f"Eval dataset size: {len(eval_blk)} samples")
            if len(eval_blk) == 0:
                logger.warning("⚠️  Eval dataset is empty after packing! Disabling evaluation.")
                eval_blk = None
        
        return train_blk, eval_blk

    def train(self):
        self.setup_wandb()
        self.load_model_and_tokenizer()
        train_ds, eval_ds = self.prepare_datasets()

        # Intelligently set evaluation strategy
        if eval_ds is None or len(eval_ds) == 0:
            logger.warning("⚠️  No eval dataset, disabling evaluation")
            # Compatible with different versions of transformers
            if hasattr(self.targs, 'evaluation_strategy'):
                self.targs.evaluation_strategy = "no"
            if hasattr(self.targs, 'eval_strategy'):
                self.targs.eval_strategy = "no"
            self.targs.load_best_model_at_end = False
            eval_ds = None
        else:
            logger.info(f"✓ Using eval dataset with {len(eval_ds)} samples")
            # Enable evaluation and save best model
            # Compatible with different versions of transformers
            eval_strat = getattr(self.targs, 'evaluation_strategy', None) or getattr(self.targs, 'eval_strategy', None)
            if eval_strat == "no" or eval_strat is None:
                if hasattr(self.targs, 'evaluation_strategy'):
                    self.targs.evaluation_strategy = "steps"
                if hasattr(self.targs, 'eval_strategy'):
                    self.targs.eval_strategy = "steps"
            
            if not hasattr(self.targs, 'eval_steps') or self.targs.eval_steps is None:
                self.targs.eval_steps = self.targs.save_steps if self.targs.save_steps else 200
            self.targs.load_best_model_at_end = True
            self.targs.metric_for_best_model = "eval_loss"
            self.targs.greater_is_better = False
            logger.info(f"✓ Will evaluate every {self.targs.eval_steps} steps and save best model")

        # collator: Causal LM (will set pad positions to -100)
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
        
        # Save LoRA adapter
        logger.info(f"Saving LoRA adapter to: {self.targs.output_dir}")
        self.model.save_pretrained(self.targs.output_dir)
        self.tokenizer.save_pretrained(self.targs.output_dir)
        
        # Save training configuration
        lora_config = {
            "model_args": vars(self.margs),
            "lora_args": vars(self.largs),
            "data_args": vars(self.dargs),
            "training_args": self.targs.to_dict(),
            "saved_at": datetime.now().isoformat(),
            "adapter_type": "lora",
        }
        with open(os.path.join(self.targs.output_dir, "lora_training_config.json"), "w", encoding="utf-8") as f:
            json.dump(lora_config, f, ensure_ascii=False, indent=2)
        
        if self.targs.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
                
        logger.info(f"Training done. final_loss={getattr(train_result, 'training_loss', None)}")
        logger.info(f"LoRA adapter saved to: {self.targs.output_dir}")
        return train_result

# -------------------- Main --------------------
def main():
    parser = transformers.HfArgumentParser((ModelArguments, LoraArguments, DataArguments, TrainingArguments2))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        margs, largs, dargs, targs = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        margs, largs, dargs, targs = parser.parse_args_into_dataclasses()

    # Fallback for output directory
    if not targs.output_dir:
        targs.output_dir = f"./code_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(targs.output_dir, exist_ok=True)

    # TF32 (if provided)
    if hasattr(targs, "tf32") and targs.tf32 and torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    logger.info(
        f"cfg: model={margs.model_name_or_path}, "
        f"seq_len={margs.max_seq_length}, stride≈{int(margs.max_seq_length * dargs.stride_fraction)}, "
        f"lora_r={largs.lora_r}, lora_alpha={largs.lora_alpha}, "
        f"target_modules={largs.target_modules}"
    )

    try:
        runner = CodeLoraFineTuner(margs, largs, dargs, targs)
        runner.train()
        logger.info(f"✅ LoRA adapter saved successfully to: {targs.output_dir}")
        return 0
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

