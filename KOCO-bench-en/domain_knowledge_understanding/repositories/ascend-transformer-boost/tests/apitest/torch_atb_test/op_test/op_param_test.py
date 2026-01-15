#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_atb
import unittest

def run_test():
    layer_norm_param = torch_atb.LayerNormParam()
    layer_norm_param.layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM
    norm_param = torch_atb.LayerNormParam.NormParam()
    norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    norm_param.epsilon = 1e7
    norm_param.begin_norm_axis = 1
    norm_param.begin_params_axis = 2
    norm_param.dynamic_quant_type = torch_atb.DynamicQuantType.DYNAMIC_QUANT_UNDEFINED
    layer_norm_param.norm_param = norm_param
    pre_norm_param = torch_atb.LayerNormParam.PreNormParam()
    pre_norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    pre_norm_param.epsilon = 1.0
    pre_norm_param.op_mode = 3
    pre_norm_param.zoom_scale_value = 1.0
    layer_norm_param.pre_norm_param = pre_norm_param
    post_norm_param = torch_atb.LayerNormParam.PostNormParam()
    post_norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    post_norm_param.epsilon = 2.2
    post_norm_param.op_mode = 2
    post_norm_param.zoom_scale_value = 1.0
    layer_norm_param.post_norm_param = post_norm_param
    print(layer_norm_param)

    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_LOGICAL_NOT
    quant_param = torch_atb.ElewiseParam.QuantParam()
    quant_param.input_scale = 1e7
    quant_param.asymmetric = True
    quant_param.input_offset = 3
    elewise_param.quant_param = quant_param
    muls_param = torch_atb.ElewiseParam.MulsParam()
    muls_param.var_attr = 1e7
    elewise_param.muls_param = muls_param
    elewise_param.out_tensor_type = torch_atb.AclDataType.ACL_UINT16
    print(elewise_param)

    linear_param = torch_atb.LinearParam()
    linear_param.transpose_a = True 
    linear_param.transpose_b = True 
    linear_param.has_bias = True
    linear_param.out_data_type = torch_atb.AclDataType.ACL_FLOAT
    linear_param.en_accum = True
    print(linear_param)

    softmax_param = torch_atb.SoftmaxParam()
    softmax_param.axes = [1,2,3]
    print(softmax_param)

    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.quant_type = torch_atb.SelfAttentionParam.QuantType.TYPE_QUANT_UNQUANT
    self_attention_param.out_data_type = torch_atb.AclDataType.ACL_DT_UNDEFINED
    self_attention_param.head_num = 2
    self_attention_param.kv_head_num = 3
    self_attention_param.q_scale = 1e7
    self_attention_param.qk_scale = 1e7
    self_attention_param.batch_run_status_enable = False
    self_attention_param.is_triu_mask = 0
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.UNDEFINED
    self_attention_param.kernel_type = torch_atb.SelfAttentionParam.KernelType.KERNELTYPE_DEFAULT
    self_attention_param.clamp_type = torch_atb.SelfAttentionParam.ClampType.CLAMP_TYPE_UNDEFINED
    self_attention_param.clamp_min =  0.0
    self_attention_param.clamp_max =  0.0
    self_attention_param.mask_type =  torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_UNDEFINED
    self_attention_param.kvcache_cfg =  torch_atb.SelfAttentionParam.KvCacheCfg.K_CACHE_V_CACHE
    self_attention_param.scale_type =  torch_atb.SelfAttentionParam.ScaleType.SCALE_TYPE_TOR
    self_attention_param.input_layout =  torch_atb.InputLayout.TYPE_BSND
    self_attention_param.mla_v_head_size = 8
    self_attention_param.cache_type = torch_atb.SelfAttentionParam.CacheType.CACHE_TYPE_NORM
    self_attention_param.window_size = 1
    print(self_attention_param)

    rope_param = torch_atb.RopeParam()
    rope_param.rotary_coeff = 1
    rope_param.cos_format = 2
    print(rope_param)

    split_param = torch_atb.SplitParam()
    split_param.split_dim = 1
    split_param.split_num = 2
    split_param.split_sizes = [1,2,3]
    print(split_param)

    gather_param = torch_atb.GatherParam()
    gather_param.axis = 1
    gather_param.batch_dims = 2
    print(gather_param)

    activation_param = torch_atb.ActivationParam()
    activation_param.activation_type = torch_atb.ActivationType.ACTIVATION_GELU
    activation_param.scale = 1e7
    activation_param.dim = 3
    activation_param.gelu_mode = torch_atb.ActivationParam.GeLUMode.TANH_MODE
    print(activation_param)

    rms_norm_param = torch_atb.RmsNormParam()
    rms_norm_param.layer_type = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_UNDEFINED
    norm_param = torch_atb.RmsNormParam.NormParam()
    norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    norm_param.epsilon = 1e7
    norm_param.layer_norm_eps = 1e7
    norm_param.rstd = False
    norm_param.precision_mode = torch_atb.RmsNormParam.PrecisionMode.HIGH_PRECISION_MODE
    norm_param.model_type = torch_atb.RmsNormParam.ModelType.LLAMA_MODEL
    norm_param.dynamic_quant_type = torch_atb.DynamicQuantType.DYNAMIC_QUANT_UNDEFINED
    rms_norm_param.norm_param = norm_param
    pre_norm_param = torch_atb.RmsNormParam.PreNormParam()
    pre_norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    pre_norm_param.epsilon = 1.2
    pre_norm_param.has_bias = True
    rms_norm_param.pre_norm_param = pre_norm_param
    post_norm_param = torch_atb.RmsNormParam.PostNormParam()
    post_norm_param.quant_type = torch_atb.QuantType.QUANT_UNQUANT
    post_norm_param.epsilon = 1.2
    post_norm_param.has_bias = True
    rms_norm_param.post_norm_param = post_norm_param
    print(rms_norm_param)

if __name__ == "__main__":
    run_test()