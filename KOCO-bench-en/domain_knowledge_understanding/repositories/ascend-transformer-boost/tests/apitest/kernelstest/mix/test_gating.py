# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import unittest

import time
import numpy as np
import torch
import torch_npu
import sys
import op_test

np.random.seed(1)
generalize = [1,4,15,16,17,32,64,65,128,256,257]

OP_NAME = "GatingOperation"

# 计算真值
def calc_golden(input_topk, input_index, device_expert_index, expert_size, topk_size, token_size, is_cumsum_int64=False):
    used_expert_num = expert_size
    if expert_size > 0 and len(device_expert_index) > 0:
        used_expert_num = len(device_expert_index)
        device_expert_index = np.sort(device_expert_index)

    input_topk_view = input_topk.view()

    inputLen = topk_size * token_size
    token_index_golden = np.zeros(inputLen, dtype=np.int32)
    cumsum_pre = np.zeros(expert_size, dtype=np.int32)

    if expert_size > 0:
        unique, counts = np.unique(input_topk_view, return_counts=True)
        cumsum_pre[unique] = counts
        cumsum_full = np.cumsum(cumsum_pre, dtype=np.int32)

        if len(device_expert_index) > 0:
            cumsum_golden = np.cumsum(cumsum_pre[device_expert_index],
                                    dtype=np.int64 if is_cumsum_int64 else np.int32)
        else:
            cumsum_golden = np.cumsum(cumsum_pre,
                                    dtype=np.int64 if is_cumsum_int64 else np.int32)
    else:
        cumsum_golden = np.zeros(used_expert_num, dtype=np.int64 if is_cumsum_int64 else np.int32)

    sort_indices = np.argsort(input_topk_view, kind='mergesort')
    input_topk_sorted = input_topk_view[sort_indices]
    original_index_golden = input_index[sort_indices]

    if expert_size > 0:
        token_index_golden = np.floor_divide(original_index_golden, topk_size, dtype=np.int32)

    if expert_size > 0 and len(device_expert_index) > 0:
        valid_index_golden = np.zeros(1, dtype=np.int32)
        valid_index_golden[0] = np.sum(cumsum_pre[device_expert_index])
        p = 0

        for _, expert_idx in enumerate(device_expert_index):
            left = cumsum_full[expert_idx-1] if expert_idx > 0 else 0
            right = cumsum_full[expert_idx]

            slice_length = right - left
            original_index_golden[p:p+slice_length] = original_index_golden[left:right]
            token_index_golden[p:p+slice_length] = token_index_golden[left:right]
            p += slice_length
    else:
        valid_index_golden = np.array([]).astype(np.int32)

    return token_index_golden, cumsum_golden, original_index_golden, input_topk_sorted, valid_index_golden

# 构造随机输入和零值输出
def create_input_and_output(token_size, topk_size, device_expert_index, expert_size, min_expert_size, max_expert_size, is_cumsum_int64=False):
    used_expert_num = expert_size
    valid_index = np.array([], dtype=np.int32)

    if expert_size > 0 and len(device_expert_index) > 0:
        used_expert_num = len(device_expert_index)
        valid_index = np.zeros(1, dtype=np.int32)

    total_size = token_size * topk_size
    input_topk = np.zeros(total_size, dtype=np.int32)

    for i in range(token_size):
        input_topk[i * topk_size:(i + 1) * topk_size] = np.random.choice(
            np.arange(min_expert_size, max_expert_size, dtype=np.int32),
            size=topk_size, replace=False)

    input_index = np.arange(total_size, dtype=np.int32)

    token_index = np.zeros(total_size, dtype=np.int32)
    cumsum = np.zeros(used_expert_num, dtype=np.int64 if is_cumsum_int64 else np.int32)
    original_index = np.zeros(total_size, dtype=np.int32)

    input0 = torch.from_numpy(input_topk).int().npu()
    input1 = torch.from_numpy(input_index).int().npu()
    output0 = torch.from_numpy(token_index).int().npu()
    output1 = torch.from_numpy(cumsum).long().npu() if is_cumsum_int64 else torch.from_numpy(cumsum).int().npu()
    output2 = torch.from_numpy(original_index).int().npu()
    output3 = torch.from_numpy(valid_index).int().npu()

    return input0, input1, output0, output1, output2, output3

class TestGating(op_test.OpTest):
    input_topk = None
    input_idx_arr = None
    input_topk_copy = None  # 排序后的topk序列

    def golden_calc(self, in_tensors):
        input_topk = in_tensors[0].cpu().numpy()
        input_idx_arr = in_tensors[1].cpu().numpy()
        token_index_golden, cum_sum_golden, original_index_golden, input_topk_copy, valid_index_golden = calc_golden(input_topk, input_idx_arr, self.device_expert, self.EXPERT_NUM, self.TOPK_NUM, self.TOKEN_NUM, self.cumSumInt64)

        return [token_index_golden, cum_sum_golden, original_index_golden, valid_index_golden]

    def golden_compare(self, out_tensors, golden_out_tensors):
        # 算子执行结果
        token_index = out_tensors[0].cpu().numpy().astype(np.int32)
        cum_sum = out_tensors[1].cpu().numpy().astype(np.int32)
        original_index = out_tensors[2].cpu().numpy().astype(np.int32)
        valid_index = out_tensors[3].cpu().numpy().astype(np.int32)

        # 真值   numpy.array类型
        token_index_golden = golden_out_tensors[0]
        cum_sum_golden = golden_out_tensors[1]
        original_index_golden = golden_out_tensors[2]
        valid_index_golden = golden_out_tensors[3]

        if (len(valid_index_golden) == 0):
            valid_index_num = len(token_index)
        else: 
            valid_index_num = valid_index_golden[0]

        return np.array_equal(token_index[0:valid_index_num], token_index_golden[0:valid_index_num]) \
            and np.array_equal(cum_sum, cum_sum_golden) \
            and np.array_equal(original_index[0:valid_index_num], original_index_golden[0:valid_index_num]) \
            and np.array_equal(valid_index, valid_index_golden)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_unaligned_case0(self):
        self.EXPERT_NUM = 0
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = 64
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_unaligned_case1(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 1711
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_aligned_case0(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_aligned_case1(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 2048
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_boundary_case0(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 1
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a    
    def test_gating_boundary_case1(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 7
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_boundary_case2(self):
        self.EXPERT_NUM = 200
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_unaligned_case0(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = [1,2,3]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_unaligned_case1(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 1711
        self.device_expert = [1,3,4]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_unaligned_case2(self):
        self.EXPERT_NUM = 0
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = 64
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_aligned_case0(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = [6,2,5]#2,3,4
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_aligned_case1(self):
        self.EXPERT_NUM = 64
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 2048
        self.device_expert = [30,40,41,50]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_boundary_case0(self):
        self.EXPERT_NUM = 200
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 1
        self.device_expert = [10,11,12]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_boundary_case1(self):
        self.EXPERT_NUM = 200
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 7
        self.device_expert = [150,160,199]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_ep_boundary_case2(self):
        self.EXPERT_NUM = 200
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 512
        self.device_expert = [199,198,100,105]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_cumSum_int64_case1(self):
        for token_num in generalize:
            self.EXPERT_NUM = 200
            self.TOPK_NUM = 7
            self.TOKEN_NUM = token_num
            self.device_expert = []
            self.cumSumInt64 = True
            min_value = 0
            max_value = self.EXPERT_NUM
            OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert, "cumSumInt64":self.cumSumInt64}
            self.set_param(OP_NAME, OP_PARAM)
            # 构造随机输入和占位输出
            input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
            self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_cumSum_int64_ep_case0(self):
        for token_num in generalize:
            self.EXPERT_NUM = 200
            self.TOPK_NUM = 3
            self.TOKEN_NUM = token_num
            self.device_expert = [3, 5, 9, 8]
            self.cumSumInt64 = True
            min_value = 0
            max_value = self.EXPERT_NUM
            OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert, "cumSumInt64":self.cumSumInt64}
            self.set_param(OP_NAME, OP_PARAM)
            # 构造随机输入和占位输出
            input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
            self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 7
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case1(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case2(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case3(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 7
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case4(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 1024
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case5(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 32
        self.TOKEN_NUM = 4096
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_boundary_case6(self):
        self.EXPERT_NUM = 1
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 1024
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_aligned_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 2048
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_aligned_case1(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_aligned_case2(self):
        self.EXPERT_NUM = 512
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_unaligned_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = 64
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_unaligned_case1(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 1711
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_unaligned_case2(self):
        self.EXPERT_NUM = 512
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 1711
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_cumSum_int64_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 7
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = True
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "cumSumInt64":self.cumSumInt64}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_cumSum_int64_case1(self):
        for token_num in generalize:
            self.EXPERT_NUM = 1024
            self.TOPK_NUM = 7
            self.TOKEN_NUM = token_num
            self.device_expert = []
            self.cumSumInt64 = True
            min_value = 0
            max_value = self.EXPERT_NUM
            OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "cumSumInt64":self.cumSumInt64}
            self.set_param(OP_NAME, OP_PARAM)
            # 构造随机输入和占位输出
            input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
            self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_1024_cumSum_int64_case2(self):
        for token_num in generalize:
            self.EXPERT_NUM = 100
            self.TOPK_NUM = 7
            self.TOKEN_NUM = token_num
            self.device_expert = []
            self.cumSumInt64 = True
            min_value = 0
            max_value = self.EXPERT_NUM
            OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "cumSumInt64":self.cumSumInt64}
            self.set_param(OP_NAME, OP_PARAM)
            # 构造随机输入和占位输出
            input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
            self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_boundary_test_big_topk_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1024
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_boundary_test_big_topk_case1(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 512
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_boundary_test_big_topk_case2(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 256
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_gating_expert_equals_topk_case0(self):
        self.EXPERT_NUM = 4
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = []
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_unaligned_case0(self):
        self.EXPERT_NUM = 256
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = [1,2,3]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_unaligned_case1(self):
        self.EXPERT_NUM = 512
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 1711
        self.device_expert = [1,133,134]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_unaligned_case2(self):
        self.EXPERT_NUM = 400
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 9
        self.device_expert = [200, 300]
        self.cumSumInt64 = False
        min_value = 0
        max_value = 64
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_aligned_case0(self):
        self.EXPERT_NUM = 512
        self.TOPK_NUM = 4
        self.TOKEN_NUM = 512
        self.device_expert = [6,2,5]#2,3,4
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_aligned_case1(self):
        self.EXPERT_NUM = 256
        self.TOPK_NUM = 2
        self.TOKEN_NUM = 3 * 2048
        self.device_expert = [30,40,41,50]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_boundary_case0(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 1
        self.device_expert = [100,211,712]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_boundary_case1(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 7
        self.device_expert = [150,260,599]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

    @op_test.only_910b
    def test_gating_1024_ep_boundary_case2(self):
        self.EXPERT_NUM = 1024
        self.TOPK_NUM = 1
        self.TOKEN_NUM = 512
        self.device_expert = [1023,1022,600,305]
        self.cumSumInt64 = False
        min_value = 0
        max_value = self.EXPERT_NUM
        OP_PARAM = {"headNum": self.TOPK_NUM, "headSize": self.EXPERT_NUM, "deviceExpert": self.device_expert}
        self.set_param(OP_NAME, OP_PARAM)
        # 构造随机输入和占位输出
        input0, input1, output0, output1, output2, output3 =  create_input_and_output(self.TOKEN_NUM, self.TOPK_NUM, self.device_expert, self.EXPERT_NUM, min_value, max_value, self.cumSumInt64)
        self.execute([input0, input1], [output0, output1, output2, output3])

if __name__ == '__main__':
    unittest.main()
