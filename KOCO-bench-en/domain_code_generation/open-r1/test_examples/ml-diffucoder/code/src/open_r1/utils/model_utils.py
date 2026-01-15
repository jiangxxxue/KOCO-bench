#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# Adapted from huggingface/open-r1: https://github.com/huggingface/open-r1/blob/6a0cd5c8ad031fc75118a4ce7f42a4860c3d8dea/src/open_r1/utils/model_utils.py


import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModel:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
