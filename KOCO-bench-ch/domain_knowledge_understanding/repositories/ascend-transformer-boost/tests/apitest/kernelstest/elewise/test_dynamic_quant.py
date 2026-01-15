#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import op_test


OP_NAME = "ElewiseOperation"
PARAM = {"elewiseType" : 20, "asymmetric" : False}


class TestDynamicQuant(op_test.OpTest):

    def golden_calc(self, in_tensors):
        input_x = in_tensors[0].to(torch.float).cpu().numpy()
        if PARAM["asymmetric"]:
            row_max = np.max(input_x, axis=-1, keepdims=True)
            row_min = np.min(input_x, axis=-1, keepdims=True)
            row_max = row_max.astype(np.float32)
            row_min = row_min.astype(np.float32)
            out_scale = (row_max - row_min) / 255
            out_offset = - (row_max + row_min) / (2 * out_scale)

            input_x = input_x.astype(np.float32)
            input_x = input_x / out_scale
            input_x = input_x + out_offset
            input_x = np.clip(input_x, -128, 127)
            out_x = np.round(input_x)
            return [torch.from_numpy(out_x).to(torch.int8),
                    torch.from_numpy(out_scale.squeeze(axis=-1)).to(torch.float32),
                    torch.from_numpy(out_offset.squeeze(axis=-1)).to(torch.float32)]
        else:
            input_abs = np.abs(input_x)
            scale = np.max(input_abs, axis=-1, keepdims=True)
            scale = scale.astype(np.float32)
            out_scale = scale / 127
            input_x = input_x * 127
            input_x = input_x / scale
            out_x = np.round(input_x)
            return [torch.from_numpy(out_x).to(torch.int8),
                    torch.from_numpy(out_scale.squeeze(axis=-1)).to(torch.float32)]

    def golden_compare(self, out_tensor, golden_out_tensor):
        
        diff = torch.abs(torch.subtract(out_tensor[0], golden_out_tensor[0]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output0 accuracy failed")
            return False

        diff = torch.abs(torch.subtract(out_tensor[1], golden_out_tensor[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensor[1].shape, dtype=golden_out_tensor[1].dtype),
                                   torch.abs(golden_out_tensor[1]))

        if torch.any(torch.greater(diff, 2**(-8) * tensor_max)):
            print("[new standards] output1 accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), 2**(-10))):
            print("[new standards] output1 eb failed")
            return False

        if len(golden_out_tensor) > 2:
            diff = torch.abs(torch.subtract(out_tensor[2], golden_out_tensor[2]))
            tensor_max = torch.maximum(torch.ones(golden_out_tensor[2].shape, dtype=golden_out_tensor[1].dtype),
                                    torch.abs(golden_out_tensor[2]))

            if torch.any(torch.greater(diff, 2**(-8) * tensor_max)):
                print("[new standards] output2 accuracy failed")
                return False

            if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), 2**(-10))):
                print("[new standards] output2 eb failed")
                return False
        return True

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case1(self):
        shape0 = (2, 32, 32)
        shape1 = (2, 32)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case2(self):
        shape0 = (2, 32, 32)
        shape1 = (2, 32)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = True
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case3(self):
        shape0 = (200, 3, 1024)
        shape1 = (200, 3)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case4(self):
        shape0 = (300, 2, 2048)
        shape1 = (300, 2)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = True
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case5(self):
        shape0 = (200, 1024)
        shape1 = (200)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case6(self):
        shape0 = (300, 2048)
        shape1 = (300)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = True
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    # when dim=4、5, pta bug, test case failed
    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case7(self):
        shape0 = (2, 5, 10, 3, 3, 1024)
        shape1 = (2, 5, 10, 3, 3)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_dynamic_quant_fp16_case8(self):
        shape0 = (2, 5, 2, 5, 3, 2, 2048)
        shape1 = (2, 5, 2, 5, 3, 2)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        PARAM["asymmetric"] = True
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])
        
    @op_test.only_910b
    def test_dynamic_quant_bf16_case9(self):
        shape0 = (3072, 5120)
        shape1 = (3072)
        input0 = np.random.uniform(low=-5, high=5, size=shape0).astype(np.float32)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])
        
    @op_test.only_910b
    def test_dynamic_quant_bf16_case10(self):
        shape0 = (352, 1536)
        shape1 = (352)
        input0 = np.random.uniform(low=-5, high=5, size=shape0).astype(np.float32)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])
        
    @op_test.only_910b
    def test_dynamic_quant_bf16_case11(self):
        shape0 = (1024, 1536)
        shape1 = (1024)
        input0 = np.random.uniform(low=-5, high=5, size=shape0).astype(np.float32)
        PARAM["asymmetric"] = False
        self.set_param(OP_NAME, PARAM)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape0, dtype=torch.int8),
                      torch.zeros(shape1, dtype=torch.float32),
                      torch.zeros(shape1, dtype=torch.float32)])
        
if __name__ == '__main__':
    unittest.main()