#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "ElewiseOperation"
PARAM = {"elewiseType" : 19, "quantParam": {"asymmetric":False}}


class TestElewiseQuantOperation(operation_test.OperationTest):
    def execute(self, op_name, op_param, in_tensors):
        print(f"———————— {op_name} test start ————————")
        self.op_param = op_param
        operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            operation.set_param(op_param)
        out_tensors = operation.execute(in_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.golden_compare_all(out_tensors, golden_out_tensors)

    def golden_compare_all(self, out_tensors, golden_out_tensors):
        self.assertEqual(len(out_tensors), len(golden_out_tensors))
        tensor_count = len(golden_out_tensors)
        for i in range(tensor_count):
            self.assertTrue(self.golden_compare(
                out_tensors[i], golden_out_tensors[i]))

    def golden_calc(self, in_tensors):
        input_x = in_tensors[0].cpu().numpy()
        if PARAM["quantParam"]["asymmetric"]:
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

            input_x = input_x.astype(np.float32)
            input_x = input_x * 127
            input_x = input_x / scale
            out_x = np.round(input_x)
            # make sure 2 golden tensor when asymmetric is false
            return [torch.from_numpy(out_x).to(torch.int8),
                    torch.from_numpy(out_scale.squeeze(axis=-1)).to(torch.float32)]

    def golden_compare(self, out_tensor, golden_out_tensor):
        torch.set_printoptions(profile="full")
        if out_tensor.dtype == torch.int8:
            result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), atol=1)
            if not result:
                print('int8 tensor check error')
        else:
            result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(),
                                    rtol=0.001, atol=0.001)
        if not result:
            print('float tensor check error')
            print("out_tensor.shape", out_tensor.shape,
                "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
            print("out_tensor:", out_tensor,
                ", \ngolden_out_tensor:", golden_out_tensor)

        return result
    def test_dynamic_quant_fp16_case1(self):
        if operation_test.get_soc_version() == 'Ascend910A' or \
           operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        x = torch.rand(2, 32, 32, dtype=torch.float16).npu()
        x = torch.clamp(x, -5, 20)
        PARAM["quantParam"]["asymmetric"] = False
        self.execute(OP_NAME, PARAM, [x])

    # def test_dynamic_quant_fp16_case2(self):
    #     x = torch.rand(2, 32, 32, dtype=torch.float16).npu()
    #     x = torch.clamp(x, -5, 20)
    #     PARAM["quantParam"]["asymmetric"] = True
    #     self.execute(OP_NAME, PARAM, [x])
    def test_dynamic_quant_fp16_case3(self):
        if operation_test.get_soc_version() == 'Ascend910A' or \
           operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        x = torch.rand(19, 23, 2048, dtype=torch.float16).npu()
        x = torch.clamp(x, -5, 20)
        PARAM["quantParam"]["asymmetric"] = False
        self.execute(OP_NAME, PARAM, [x])

    # def test_dynamic_quant_fp16_case4(self):
    #     x = torch.rand(19, 23, 2048, dtype=torch.float16).npu()
    #     x = torch.clamp(x, -5, 20)
    #     PARAM["quantParam"]["asymmetric"] = True
    #     self.execute(OP_NAME, PARAM, [x])
    def test_dynamic_quant_fp16_case5(self):
        if operation_test.get_soc_version() == 'Ascend910A' or \
           operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        x = torch.rand(200, 2048, dtype=torch.float16).npu()
        x = torch.clamp(x, -5, 20)
        PARAM["quantParam"]["asymmetric"] = False
        self.execute(OP_NAME, PARAM, [x])

    # def test_dynamic_quant_fp16_case6(self):
    #     x = torch.rand(19, 2, 3, 2048, dtype=torch.float16).npu()
    #     x = torch.clamp(x, -5, 20)
    #     PARAM["quantParam"]["asymmetric"] = True
    #     self.execute(OP_NAME, PARAM, [x])
    def test_dynamic_quant_fp16_case7(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            logging.info("this testcase only supports Ascend910B")
            return True
        x = torch.rand(19, 23, 3000, dtype=torch.float16).npu()
        x = torch.clamp(x, -5, 20)
        PARAM["quantParam"]["asymmetric"] = False
        self.execute(OP_NAME, PARAM, [x])

    # def test_dynamic_quant_fp16_case8(self):
    #     if not operation_test.get_soc_version() == 'Ascend910B':
    #         print("this testcase only supports Ascend910B")
    #         return True
    #     x = torch.rand(19, 23, 3000, dtype=torch.float16).npu()
    #     x = torch.clamp(x, -5, 20)
    #     PARAM["quantParam"]["asymmetric"] = True
    #     self.execute(OP_NAME, PARAM, [x])

if __name__ == '__main__':
    unittest.main()
