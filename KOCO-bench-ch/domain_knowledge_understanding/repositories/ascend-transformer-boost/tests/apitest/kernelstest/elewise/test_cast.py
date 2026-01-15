#
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
import sys
import logging
import op_test

OP_NAME = "ElewiseOperation"
DEFAULT_SHAPE = (10000,)

TORCH_TO_ACL_DTYPE = {
    torch.float32: 0,
    torch.half: 1,
    torch.int32: 3,
    torch.int64: 9,
    torch.bfloat16: 27,
    torch.int8: 2
}
ACL_TO_TORCH_DTYPE = {
    0: torch.float32,
    1: torch.half,
    2: torch.int8,
    3: torch.int32,
    9: torch.int64,
    27:torch.bfloat16
}


class TestElewise(op_test.OpTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0].type(ACL_TO_TORCH_DTYPE[self.op_desc["specificParam"]["outTensorType"]])]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        aclOutTensorType = self.op_desc["specificParam"]["outTensorType"]
        if aclOutTensorType in [0, 1, 27]:
            logging.debug("float16/bfloat16/float32 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                logging.debug(
                    f"actual_output is : {actual_output.type(torch.float32)}")
                logging.debug(
                    f"golden_output is : {golden_output.type(torch.float32)}")
                print(torch.sum(torch.sub(actual_output.type(
                    torch.float32), golden_output.type(torch.float32))))
                result.append(torch.equal(actual_output.type(
                    torch.float32), golden_output.type(torch.float32)))
        elif aclOutTensorType in [2, 3, 9]:
            logging.debug("int32 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                out_dtype=ACL_TO_TORCH_DTYPE[aclOutTensorType]
                result.append(torch.equal(actual_output.type(
                    out_dtype), golden_output.type(out_dtype)))
        else:
            logging.debug("Unsupport dtype:%s golden compare",
                          actual_output.dtype)
            result.append(False)

        logging.debug(f"result is : {all(result)}")
        return all(result)

    def __test_cast(self, shape: tuple[int], input_dtype: torch.dtype, output_dtype: torch.dtype):
        input_ndarray = np.random.uniform(low=-5, high=5, size=shape)
        input_tensor = torch.from_numpy(input_ndarray).to(input_dtype)
        output_tensor = torch.zeros(shape).to(output_dtype)
        self.set_param(OP_NAME,
                       {"elewiseType": 1,
                        "outTensorType": TORCH_TO_ACL_DTYPE[output_dtype]}
                       )
        self.execute(
            [input_tensor], [output_tensor]
        )

    def test_cast_f16f32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float16, torch.float32)

    def test_cast_f32f16(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float32, torch.float16)

    def test_cast_i32i64(self):
        self.__test_cast(DEFAULT_SHAPE, torch.int32, torch.int64)

    def test_cast_i64i32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.int64, torch.int32)

    def test_cast_f16i32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float16, torch.int32)

    def test_cast_i32f16(self):
        self.__test_cast(DEFAULT_SHAPE, torch.int32, torch.float16)

    @op_test.only_910b
    def test_cast_bf16f32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.bfloat16, torch.float32)

    @op_test.only_910b
    def test_cast_f32bf16(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float32, torch.bfloat16)

    def test_cast_i32f32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.int32, torch.float32)

    def test_cast_f32i32(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float32, torch.int32)

    def test_cast_i8f16(self):
        self.__test_cast(DEFAULT_SHAPE, torch.int8, torch.float16)

    def test_cast_f16i8(self):
        self.__test_cast(DEFAULT_SHAPE, torch.float16, torch.int8)


if __name__ == '__main__':
    unittest.main()
