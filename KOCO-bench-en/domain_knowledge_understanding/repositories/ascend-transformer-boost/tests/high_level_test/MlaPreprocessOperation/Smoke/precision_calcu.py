# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import unittest
import logging
import json
import re
import numpy
import torch
import torch_npu
import math
import sys
import shutil
from enum import Enum

MIN_ERR = 1e-7
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

dtype_dict = {"float": torch.float32, "float16": torch.float16, "int8": torch.int8, "int32": torch.int32, "uint8": torch.uint8,
              "int16": torch.int16, "uint16": torch.int16, "uint32": torch.int32, "int64": torch.int64, "uint64": torch.int64,
              "double": torch.double, "bool": torch.bool, "complex64": torch.complex64, "complex128": torch.complex128, "bf16": torch.bfloat16}

def get_eb_threshold(dtype:torch.dtype):
    eb_threshold = 0
    if dtype in [torch.bfloat16]:
        eb_threshold = 2**(-7)
    if dtype in [torch.float16]:
        eb_threshold = 2**(-10)
    if dtype in [torch.float32]:
        eb_threshold = 2**(-14)
    return eb_threshold

def get_err_threshold(op_type:OpTypes, dtype:torch.dtype):
    err_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-7)
        if dtype in [torch.float16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float32]:
            err_threshold = 2**(-11)
    if op_type in [OpTypes.CV_FUSION]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float16]:
            err_threshold = 2**(-11)
        if dtype in [torch.float32]:
            err_threshold = 2**(-14)
    return err_threshold


#误差均衡性（EB）
def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    return EB

#单标杆、浮点比对方法|actual - expected| <= err × max(1, | expected |)
def ref_compare(golden:torch.Tensor, actual:torch.Tensor, err):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= err * golden_nmax).all()
    logging.info(f"new golden result:{result}")
    return result


#最大相对误差：max relative error，MARE
def get_mare(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare

#平均相对误差：mean relative error，MERE
def get_mere(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere

#均方根误差:Root Mean Squared Error，RMSE
def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse

def compare_cv(golden:torch.Tensor, gpu:torch.Tensor, actual:torch.Tensor):
    op_type = OpTypes.CV_FUSION
    judge_threshold = 522
    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)
    logging.info(f"err_threshold:{err_threshold} eb_threshold:{eb_threshold}")
    mare_npu = get_mare(golden, actual)
    mare_gpu = get_mare(golden, gpu)

    mere_npu = get_mere(golden, actual)
    mere_gpu = get_mere(golden, gpu)

    rmse_npu = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, gpu)

    mare_rate = mare_npu / max(mare_gpu, err_threshold)
    mere_rate = mere_npu / max(mere_gpu, err_threshold)
    rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)

    EB = get_eb(gpu, actual)
    result = (mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2) and (EB < eb_threshold)

    print(f"mare_npu:{mare_npu} mare_gpu:{mare_gpu}")
    print(f"mere_npu:{mere_npu} mere_gpu:{mere_gpu}")
    print(f"rmse_npu:{rmse_npu} rmse_gpu:{rmse_gpu}")
    print(f"MARE:{mare_rate} MERE:{mere_rate} RMSE:{rmse_rate} EB:{EB}")
    print(f"new golden cv result:{result}")
    return result

if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
   gloden = torch.rand((128,128), dtype=torch.float32)
   actual = gloden.to(torch.float16)
   gpu = actual
   compare_cv(gloden, gpu, actual)
   