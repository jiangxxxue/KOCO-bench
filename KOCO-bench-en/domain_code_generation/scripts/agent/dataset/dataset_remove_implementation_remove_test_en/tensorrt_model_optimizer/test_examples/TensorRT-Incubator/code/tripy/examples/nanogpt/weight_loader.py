#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from transformers import GPT2LMHeadModel

import nvtripy as tp


def load_weights_from_hf(model, model_type, dtype):
    print(f"Loading weights from pretrained model: '{model_type}'")

    tripy_state_dict = model.state_dict()
    # attention biases are initialized in the model based on block size.
    tripy_keys = {key for key in tripy_state_dict.keys() if not key.endswith(".attn.bias")}

    # Load huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    hf_state_dict = model_hf.state_dict()
    # We ignore some of the keys in the HF checkpoint:
    hf_keys = {
        key for key in hf_state_dict.keys() if not key.endswith(".attn.masked_bias") and not key.endswith(".attn.bias")
    }
    assert hf_keys == tripy_keys, (
        f"Mismatched keys. Note:\n"
        f"`hf_keys` extra keys: {hf_keys - tripy_keys}\n"
        f"`tripy_keys` extra keys: {tripy_keys - hf_keys}"
    )

    # See https://paperswithcode.com/method/weight-tying for details on why we do this:
    hf_state_dict["transformer.wte.weight"] = hf_state_dict["lm_head.weight"]

    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    torch_dtype = getattr(torch, dtype.name)
    for key in hf_keys:
        weight = hf_state_dict[key]
        if any(key.endswith(w) for w in transposed):
            with torch.no_grad():
                weight = hf_state_dict[key].t().contiguous()
        weight = weight.to(torch_dtype)
        param = tp.Tensor(weight)
        tripy_state_dict[key] = param

    model.load_state_dict(tripy_state_dict)


