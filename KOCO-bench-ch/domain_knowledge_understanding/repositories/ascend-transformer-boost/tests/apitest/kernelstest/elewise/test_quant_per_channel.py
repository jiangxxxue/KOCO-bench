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

import numpy as np
import torch
import op_test


OP_NAME = "ElewiseOperation"


class TestQuantPerChannel(op_test.OpTest):
    def golden_calc(self, in_tensors):
        int8_lower_bound = -128
        if "ASDOPS_QUANT_MIN_NEG_127" in self.op_desc["specificParam"]:
            if self.op_desc["specificParam"]["ASDOPS_QUANT_MIN_NEG_127"] == "1":
                int8_lower_bound = -127
        if in_tensors[0].dtype == torch.bfloat16:
            input_x = in_tensors[0].float().numpy()
            input_scale = in_tensors[1].float().numpy()
            input_offset = in_tensors[2].float().numpy()
        else:
            input_x = in_tensors[0].numpy()
            input_scale = in_tensors[1].numpy()
            input_offset = in_tensors[2].numpy()
        if len(input_offset) == 0:
            out = np.clip((np.round((input_x / input_scale))), int8_lower_bound, 127)
        else:
            out = np.clip((np.round((input_x / input_scale)) + input_offset), int8_lower_bound, 127)

        return [torch.from_numpy(out).to(torch.int8)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], atol=1)

    def test_quant_per_channel_10_8192(self):
        op_param = {"elewiseType": 18}
        shape = (10,8192)
        shape1 = (8192)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_10_8192_bf16(self):
        op_param = {"elewiseType": 18}
        shape = (10,8192)
        shape1 = (8192)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_10_32768(self):
        op_param = {"elewiseType": 18}
        shape = (10,32768)
        shape1 = (32768)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_1_8222_not_align(self):
        op_param = {"elewiseType": 18}
        shape = (1,8222)
        shape1 = (8222)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_4096_40_not_align_dimbloop_larger_than_1(self):
        op_param = {"elewiseType": 18}
        shape = (4096,40)
        shape1 = (40)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_4096_40_scalar_not_align_dimbloop_larger_than_1(self):
        op_param = {"elewiseType": 18}
        shape = (4096,40)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_1_8222_not_align_bf16(self):
        op_param = {"elewiseType": 18}
        shape = (1,8222)
        shape1 = (8222)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_4096_40_not_align_bf16_dimbloop_larger_than_1(self):
        op_param = {"elewiseType": 18}
        shape = (4096,40)
        shape1 = (40)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_4096_40_scalar_not_align_bf16_dimbloop_larger_than_1(self):
        op_param = {"elewiseType": 18}
        shape = (4096,40)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_2_40961_cut_n_not_align(self):
        op_param = {"elewiseType": 18}
        shape = (2,40961)
        shape1 = (40961)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_2_40961_cut_n_not_align_bf16(self):
        op_param = {"elewiseType": 18}
        shape = (2,40961)
        shape1 = (40961)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_99_31_smaller_than_32(self):
        op_param = {"elewiseType": 18}
        shape = (99,31)
        shape1 = (31)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_2_2_small_shape(self):
        op_param = {"elewiseType": 18}
        shape = (2,2)
        shape1 = (2)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_2_1_small_shape_scalar(self):
        op_param = {"elewiseType": 18}
        shape = (1,1)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_99_32_small_shape_align(self):
        op_param = {"elewiseType": 18}
        shape = (99,32)
        shape1 = (99,32)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_10_20_1024_5(self):
        op_param = {"elewiseType": 18}
        shape = (10,20,1024,5)
        shape1 = (1024,5)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_10_20_1024_1024(self):
        op_param = {"elewiseType": 18}
        shape = (10,20,1024,1024)
        shape1 = (1024,1024)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_1024_1024_lower_bound(self):
        op_param = {"elewiseType": 18, "ASDOPS_QUANT_MIN_NEG_127": "0"}
        shape = (1024,1024)
        shape1 = (1024,1024)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)],
                     {"ASDOPS_QUANT_MIN_NEG_127": "0"})

    def test_quant_per_channel_1024_1024_scalar(self):
        op_param = {"elewiseType": 18}
        shape = (1024,1024)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_quant_per_channel_1024_1024_scalar_bf16(self):
        op_param = {"elewiseType": 18}
        shape = (1024,1024)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16(), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_1024_1024_scalar_lower_bound(self):
        op_param = {"elewiseType": 18, "ASDOPS_QUANT_MIN_NEG_127": "1"}
        shape = (1024,1024)
        shape1 = (1)
        input0 = np.random.uniform(low=-1000, high=-1000, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=5, high=5, size=shape1).astype(np.float16)
        input2 = np.random.uniform(low=5, high=5, size=shape1).astype(np.int8)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.from_numpy(input2)],
                     [torch.zeros(shape, dtype=torch.int8)],
                     {"ASDOPS_QUANT_MIN_NEG_127": "1"})

    def test_quant_per_channel_1_1024_1024_offset_not_exist(self):
        op_param = {"elewiseType": 18}
        shape = (1,1024,1024)
        shape1 = (1024,1024)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.Tensor()],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_1024_1024_scalar_offset_not_exist(self):
        op_param = {"elewiseType": 18}
        shape = (1024,1024)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.Tensor()],
                     [torch.zeros(shape, dtype=torch.int8)])

    def test_quant_per_channel_1024_1024_scalar_offset_not_exist_lower_bound(self):
        op_param = {"elewiseType": 18, "ASDOPS_QUANT_MIN_NEG_127": "1"}
        shape = (1024,1024)
        shape1 = (1)
        input0 = np.random.uniform(low=-1000, high=-1000, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=5, high=5, size=shape1).astype(np.float16)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.Tensor()],
                     [torch.zeros(shape, dtype=torch.int8)],
                     {"ASDOPS_QUANT_MIN_NEG_127": "1"})

    # 长序列用例，CI机器无法支撑，本地调试使用
    def atest_quant_per_channel_long_sequence_scalar(self):
        op_param = {"elewiseType": 18, "ASDOPS_QUANT_MIN_NEG_127": "1"}
        shape = (1048576,4096)
        shape1 = (1)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape1).astype(np.float16)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1), torch.Tensor()],
                     [torch.zeros(shape, dtype=torch.int8)],
                     {"ASDOPS_QUANT_MIN_NEG_127": "1"})

if __name__ == '__main__':
    unittest.main()