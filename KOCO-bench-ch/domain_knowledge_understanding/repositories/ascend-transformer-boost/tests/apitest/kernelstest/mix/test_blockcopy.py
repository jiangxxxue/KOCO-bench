
import os
import unittest

import numpy as np
import torch
import sys
import op_test

np.random.seed(1)

OP_NAME = "BlockCopyOperation"
OP_PARAM0 = {}

import numpy as np
import random

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

def create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, dtypeStr):
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

class TestBlockCopy(op_test.OpTest):

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
        return [kCacheGolden, vCacheGolden]
    
    def golden_compare(self, out_tensors, golden_out_tensors):
        # 算子执行结果
        kCache = out_tensors[0].cpu()
        vCache = out_tensors[1].cpu()
        if kCache.dtype == torch.bfloat16:
            kCache = kCache.to(torch.float16)
            vCache = vCache.to(torch.float16)
        kCache = kCache.numpy()
        vCache = vCache.numpy()
        # 真值   numpy.array类型
        kCacheGolden = golden_out_tensors[0]
        vCacheGolden = golden_out_tensors[1]
        return np.array_equal(kCache, kCacheGolden) and np.array_equal(vCache, vCacheGolden)
    
    @op_test.only_910b
    def test_blockcopy_float16_case_small(self):
        blockCount = 15
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 3
        dstCount = 10
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_float16_case_src_equal_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5000
        dstCount = 5000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_float16_case_big(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 1000
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_float16_case_typical(self):
        blockCount = 1000
        blockSize = 128
        numHead = 1
        headSize = 128
        srcCount = 100
        dstCount = 500
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_float16_case_src_far_less_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_float16_case_aligned(self):
        blockCount = 200
        blockSize = 8
        numHead = 16
        headSize = 16
        srcCount = 30
        dstCount = 60
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_small(self):
        blockCount = 15
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 3
        dstCount = 10
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_src_equal_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5000
        dstCount = 5000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_big(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 1000
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_typical(self):
        blockCount = 1000
        blockSize = 128
        numHead = 1
        headSize = 128
        srcCount = 100
        dstCount = 500
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_src_far_less_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_bfloat16_case_aligned(self):
        blockCount = 200
        blockSize = 8
        numHead = 16
        headSize = 16
        srcCount = 30
        dstCount = 60
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        input0 = input0.to(torch.bfloat16)
        input1 = input1.to(torch.bfloat16)
        self.kCache = input0.to(torch.float16).cpu().numpy().copy()
        self.vCache = input1.to(torch.float16).cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_int8_case_small(self):
        blockCount = 15
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 3
        dstCount = 10
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_int8_case_src_equal_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5000
        dstCount = 5000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_int8_case_big(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 1000
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_int8_case_src_far_less_dst(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 2
        srcCount = 5
        dstCount = 6000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])
    
    @op_test.only_910b
    def test_blockcopy_int8_case_aligned(self):
        blockCount = 200
        blockSize = 8
        numHead = 16
        headSize = 16
        srcCount = 30
        dstCount = 60
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])

    @op_test.only_310p
    def test_blockcopy_float16_case_small_310(self):
        blockCount = 15
        blockSize = 2
        numHead = 2
        headSize = 16
        srcCount = 3
        dstCount = 10
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])

    @op_test.only_310p
    def test_blockcopy_float16_case_big_310(self):
        blockCount = 20000
        blockSize = 2
        numHead = 2
        headSize = 16
        srcCount = 2000
        dstCount = 2000
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "float16")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])

    @op_test.only_310p
    def test_blockcopy_int8_case_aligned_310(self):
        blockCount = 200
        blockSize = 8
        numHead = 16
        headSize = 16
        srcCount = 30
        dstCount = 60
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])


    @op_test.only_310p
    def test_blockcopy_int8_case_aligned_big_310(self):
        blockCount = 20000
        blockSize = 8
        numHead = 16
        headSize = 32
        srcCount = 2000
        dstCount = 2001
        self.set_param(OP_NAME, OP_PARAM0)
        # 构造随机输入和占位输出
        input0, input1, input2, input3, input4 =  create_input(blockCount, blockSize, numHead, headSize, srcCount, dstCount, "int8")
        self.kCache = input0.cpu().numpy().copy()
        self.vCache = input1.cpu().numpy().copy()
        self.blockCount = blockCount
        self.execute([input0, input1, input2, input3, input4], [0, 1])


if __name__ == '__main__':
    unittest.main()
