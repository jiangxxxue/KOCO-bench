#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import math
import json
import unittest
import torch
import torch_npu
import numpy as np
import random
import logging


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "BlockCopyOperation"
PARAM = {}

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2] # 最後兩維nd->nz
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a /a0), a0, b0]

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) # x原始需要對齊，才能reshape

def generate_dstBlks(srcBlks, blockCount, dstCount):
    # 创建包含所有可能整数的集合
    all_possible_integers = set(range(blockCount))

    # 从集合中移除数组 a 中的所有元素
    integers_not_in_src = all_possible_integers - set(srcBlks)

    # 确保剩余的整数数量足够
    if len(integers_not_in_src) < dstCount:
        raise ValueError("Not enough unique integers available")

    # 从剩余的集合中随机选择 x 个整数
    selected_integers = np.random.choice(list(integers_not_in_src), size=dstCount, replace=False)

    return selected_integers

def generate_cumSum(srcCount, dstCount):
    if srcCount > dstCount:
        raise ValueError("srcCount cannot be greater than dstCount")

    # 生成包含1到count的所有整数的列表
    full_range = list(range(1, dstCount))

    # 从full_range中随机选择length个不重复的整数
    random_sequence = random.sample(full_range, srcCount - 1)
    cumSum = np.append(random_sequence, dstCount)
    cumSum.sort()

    return cumSum


def create_input_nz(blockCount, blockSize, numHead, headSize, srcCount, dstCount, dtypeStr, format='nd'):
    kCache = np.random.rand(blockCount,blockSize,numHead,headSize).astype(dtypeStr)
    vCache = np.random.rand(blockCount,blockSize,numHead,headSize).astype(dtypeStr)
    srcBlks = np.random.randint(blockCount, size=srcCount).astype(np.int32)
    dstBlks = generate_dstBlks(srcBlks, blockCount, dstCount).astype(np.int32)
    cumSum = generate_cumSum(srcCount, dstCount).astype(np.int32)
    
    input0 = torch.from_numpy(kCache).npu()
    input1 = torch.from_numpy(vCache).npu()
    input2 = torch.from_numpy(srcBlks).int().npu()
    input3 = torch.from_numpy(dstBlks).int().npu()
    input4 = torch.from_numpy(cumSum).int().npu()
    
    return input0, input1, input2, input3, input4


class TestConcatOperation(operation_test.OperationTest):

    def golden_calc(self, in_tensors):
        kCacheGolden = self.kCache.copy()
        vCacheGolden = self.vCache.copy()
        srcBlks = in_tensors[2].cpu().numpy()
        dstBlks = in_tensors[3].cpu().numpy()
        cumSum = in_tensors[4].cpu().numpy()
        startIdx = 0
        endIdx = 0
        for i in range(len(srcBlks)):
            srcBlk = srcBlks[i]
            endIdx = cumSum[i]
            for j in range(startIdx, endIdx):
                dstBlk = dstBlks[j]
                kCacheGolden[dstBlk] = kCacheGolden[srcBlk].copy()
                vCacheGolden[dstBlk] = vCacheGolden[srcBlk].copy()
            startIdx = endIdx
        blockCount = kCacheGolden.shape[0]
        blockSize = kCacheGolden.shape[1]

        kCacheGolden = kCacheGolden.reshape(blockCount, blockSize, -1)
        vCacheGolden = vCacheGolden.reshape(blockCount, blockSize, -1)
        kCacheGolden = convert_nd_to_nz(kCacheGolden)
        vCacheGolden = convert_nd_to_nz(vCacheGolden)
        kCacheGolden = kCacheGolden.reshape(blockCount, -1, blockSize, 16)
        vCacheGolden = vCacheGolden.reshape(blockCount, -1, blockSize, 16)
        return [kCacheGolden, vCacheGolden]

    def golden_compare(self, out_tensors, golden_out_tensors):
        # 算子执行结果
        kCache = out_tensors[0].cpu().numpy().astype(np.float16)
        vCache = out_tensors[1].cpu().numpy().astype(np.float16)
        # 真值   numpy.array类型
        kCacheGolden = golden_out_tensors[0]
        vCacheGolden = golden_out_tensors[1]
        return np.array_equal(kCache, kCacheGolden) and np.array_equal(vCache, vCacheGolden)

    def check_310p_and_execute(self, *args, **kwargs):
        if not operation_test.get_soc_version() == 'Ascend310P':
            logging.info("this testcase supports Ascend310P only")
            return
        self.execute_inplace(*args, **kwargs)

    def test_blockcopy_float16_case_small_nz(self):
        blockCount = 32
        blockSize = 16
        numHead = 32
        headSize = 128
        srcCount = 3
        dstCount = 10
        dtypeStr = "float16"
        input0, input1, input2, input3, input4 =  create_input_nz(blockCount, blockSize, numHead, headSize, srcCount, dstCount, dtypeStr)

        kCache = input0.cpu().numpy().copy()
        vCache = input1.cpu().numpy().copy()
        kCache = kCache.reshape(blockCount, blockSize, -1)
        vCache = vCache.reshape(blockCount, blockSize, -1)
        kCache = convert_nd_to_nz(kCache)
        kCache = kCache.reshape(blockCount, -1, blockSize, 16).astype(dtypeStr)
        kCache = np.ascontiguousarray(kCache)
        vCache = convert_nd_to_nz(vCache)
        vCache = vCache.reshape(blockCount, -1, blockSize, 16).astype(dtypeStr)
        vCache = np.ascontiguousarray(vCache)

        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        kCache = torch.from_numpy(kCache).npu()
        vCache = torch.from_numpy(vCache).npu()
        kCache = torch_npu.npu_format_cast(kCache, 29)
        vCache = torch_npu.npu_format_cast(vCache, 29)
        self.blockCount = blockCount
        self.dstBlks = input3.cpu().numpy().copy()
        self.check_310p_and_execute(OP_NAME, PARAM, [kCache, vCache, input2, input3, input4], [0, 1]) 
    
    def test_blockcopy_float16_case_nz(self):
        blockCount = 587
        blockSize = 128
        numHead = 32
        headSize = 128
        srcCount = 3
        dstCount = 10
        dtypeStr = "float16"
        input0, input1, input2, input3, input4 =  create_input_nz(blockCount, blockSize, numHead, headSize, srcCount, dstCount, dtypeStr)

        kCache = input0.cpu().numpy().copy()
        vCache = input1.cpu().numpy().copy()
        kCache = kCache.reshape(blockCount, blockSize, -1)
        vCache = vCache.reshape(blockCount, blockSize, -1)
        kCache = convert_nd_to_nz(kCache)
        kCache = kCache.reshape(blockCount, -1, blockSize, 16).astype(dtypeStr)
        kCache = np.ascontiguousarray(kCache)
        vCache = convert_nd_to_nz(vCache)
        vCache = vCache.reshape(blockCount, -1, blockSize, 16).astype(dtypeStr)
        vCache = np.ascontiguousarray(vCache)

        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        kCache = torch.from_numpy(kCache).npu()
        vCache = torch.from_numpy(vCache).npu()
        kCache = torch_npu.npu_format_cast(kCache, 29)
        vCache = torch_npu.npu_format_cast(vCache, 29)
        self.blockCount = blockCount
        self.dstBlks = input3.cpu().numpy().copy()
        self.check_310p_and_execute(OP_NAME, PARAM, [kCache, vCache, input2, input3, input4], [0, 1])

if __name__ == '__main__':
    if ("Ascend310B" != torch.npu.get_device_name()[len("Ascend310B")]) :
        unittest.main()