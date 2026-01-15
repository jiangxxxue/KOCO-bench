#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

from enum import Enum
import torch
import logging

class OpTypes(Enum):
    NA = 0 # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9

def get_precision_and_eb_threshold(op_type, dtype):
    precision_threshold = 0
    eb_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT]:
        if dtype in [torch.int8]:
            precision_threshold = 1
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-7)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-11)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.COMPUTE_FLOAT_HIGH_PRECISION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-11)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-14)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.VECTOR_FUSION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-9)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-12)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.CV_FUSION]:
        precision_threshold = 522 #最大相对误差5/平均相对误差2/均方根误差2
        if dtype in [torch.float16]:
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            eb_threshold = 2**(-14)
    logging.debug(f"op_type: {op_type}, dtype: {dtype}, precision_threshold: {precision_threshold}, eb_threshold: {eb_threshold}")
    return precision_threshold, eb_threshold

def precision_performance_analysis(op_type,golden_output_tensor_list,output_tensor_list):
    for i in range(len(golden_output_tensor_list)):
        actual_output = output_tensor_list[i].cpu()
        golden_output = golden_output_tensor_list[i]
        precision_threshold, eb_threshold = get_precision_and_eb_threshold(op_type, actual_output.dtype)
        precision, eb = cal_precision_eb_percent(op_type,i, actual_output, golden_output, precision_threshold, eb_threshold)
    if precision == 100 and eb == 0:
        return True 
    else:
        print(f"precision: {precision}, eb: {eb}")
        return False
    

def cal_precision_eb_percent(op_type,i, actual_output, golden_output, precision_threshold, eb_threshold):
    actual_output = actual_output if actual_output.dtype != torch.bool else actual_output.long()
    golden_output = golden_output if golden_output.dtype != torch.bool else golden_output.long()
    if op_type in [OpTypes.COMPUTE_FLOAT, OpTypes.COMPUTE_FLOAT_HIGH_PRECISION, OpTypes.VECTOR_FUSION] and actual_output.dtype in [torch.float16, torch.bfloat16]:
        actual_output = actual_output.to(torch.float32)
        golden_output = golden_output.to(torch.float32)
    #对于输出中出现的NAN以及INF全部替换成0
    actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
    actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
    golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
    golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
    if op_type == OpTypes.RAND:
        alpha = 0.01
        t_statistic, p_value = scipy.stats.ks_2samp(actual_output, golden_output)
        precision_percent = 100 if p_value > alpha else 0
        eb_percent = 0
        return precision_percent, eb_percent
    diff = torch.subtract(actual_output, golden_output)
    tensor_max = torch.maximum(torch.ones(golden_output.shape, dtype=golden_output.dtype), torch.abs(golden_output))
    if precision_threshold == 1:
        tolerance = torch.subtract(torch.abs(diff), torch.ones(diff.shape, dtype=diff.dtype))
    else:
        tolerance = torch.subtract(torch.abs(diff), precision_threshold * tensor_max)
    # eb 统计误差偏移情况
    eb = eb_threshold
    if eb_threshold != 0:
        eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
    precision_percent = torch.sum(tolerance <= 0).numpy() / torch.numel(tolerance) * 100
    eb_percent = 0 if eb == 0 else torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100
    return precision_percent, eb_percent