# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import unittest
import torch
import sys, os
import logging
import numpy as np
 
sys.path.append("../")
sys.path.append("../..")
import op_test
from typing import List, Tuple
 
BLOCK_SIZE = 16
DRANGE = (-5, 5)
MATMUL_ACCUM_ATOMIC = 2
 
 
def ceil_div(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return int("inf")
    return -(dividend // -divisor)
 
 
def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align
 
 
def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    eb = torch.mean(actual_error / golden_nmax)
    result = torch.abs(eb) <= 2 ** (-7)
    return result
 
 
def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result
 
 
class TestGmmAdd(op_test.OpTest):
    def __gen_test_data(self, shape: tuple, groupList, in_dtype=torch.float16) -> None:
        groupNum, msize, ksize, nsize = shape
        self.x = torch.rand((msize, ksize)).to(in_dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
        self.weight = torch.rand((ksize, nsize)).to(in_dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
        self.y = torch.rand(size=(groupNum*msize, nsize)).to(torch.float32) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
        self.yRef = torch.zeros_like(self.y)
        for i in range(groupNum):
            startIdx = 0 if i == 0 else groupList[i - 1]
            endIdx = groupList[i]
            startNum = i * msize
            endNum = (i + 1) * msize
            self.yRef[startNum:endNum, :] = (
                torch.mm(self.x[:, startIdx:endIdx].float(), self.weight[startIdx:endIdx, :].float()) + self.y[startNum:endNum, :]
            )
        if self.op_desc["specificParam"]["transposeA"]:
            self.x.transpose_(1, 0)
        if self.op_desc["specificParam"]["transposeB"]:
            self.weight.transpose_(1, 0)
        return
 
    def golden_calc(self, in_tensors: List[torch.Tensor]):
        return [self.yRef]
 
    def golden_compare(self, out_tensors: List[torch.Tensor], golden_out_tensors: List[torch.Tensor]):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        eb = get_eb(golden_out_tensors[0], out_tensors[0])
        err_thod = 2**-8
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], err_thod)
        return eb and cmp
 
    @op_test.only_910b
    def common_testcase_910b(self, shape, group_list, transA, transB, data_type):
        self.set_param(
            "GmmAddOperation",
            {
                "transposeA": transA,
                "transposeB": transB,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data(shape, group_list, data_type)
        self.execute(
            [
                self.x.to(data_type),
                self.weight.to(data_type),
                torch.tensor(group_list, dtype=torch.int64),
                self.y.float(),
            ],
            [3],
        )
        pass
 
 
    @op_test.only_910b
    def testcase_gmm_add_bf16_tn0(self):
        shape = (2, 345, 1280, 567)
        group_list = [256, 1280]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)

    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_gn1(self):
        shape = (1, 345, 1280, 567)
        group_list = [1280]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)

    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_gn3(self):
        shape = (3, 345, 1560, 567)
        group_list = [256, 1280, 1560]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)

    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_gn4(self):
        shape = (4, 345, 2560, 567)
        group_list = [345, 1280, 1920, 2560]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)

    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_gn5(self):
        shape = (5, 345, 2048, 567)
        group_list = [256, 512, 1024, 1536, 2048]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16) 


    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_case0(self):
        shape = (8, 9216, 2048, 9216)
        group_list = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)

    @op_test.only_910b
    def testcase_gmm_add_bf16_tn_case1(self):
        shape = (8, 4608, 2048, 9216)
        group_list = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
        transA, transB = True, False
        self.common_testcase_910b(shape, group_list, transA, transB, torch.bfloat16)
 
if __name__ == "__main__":
    unittest.main()