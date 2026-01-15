#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import logging
import unittest
import op_test
import torch

OP_NAME = "SortOperation"

class TopkFP32Test(op_test.OpTest):
    def golden_calc(self, in_tensors):
        num = self.op_desc["specificParam"]["num"]
        res = list(torch.topk(in_tensors[0].to(torch.float32), k=num[0]))
        return [res[0].to(torch.float32), res[1].to(torch.int32)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.debug(out_tensors[0])
        logging.debug(golden_out_tensors[0])
        result0 = torch.equal(out_tensors[0], golden_out_tensors[0])
        # Indices (out_tensors[1]) are unstable due to unstable sorting.
        index_gather = torch.gather(self.in_tensors[0], -1, out_tensors[1].to(torch.int64))
        result1 = torch.equal(index_gather, golden_out_tensors[0])
        return result0 and result1

    @op_test.only_910b
    def test_TopKDescFP32_8x6_1(self):
        shape = (8, 6)
        num = [1]
        self.in_tensors = [torch.empty(shape, dtype=torch.float32).uniform_(-2, 2)]
        self.set_param("SortOperation", {"num": num})
        self.execute(self.in_tensors,
                     [torch.zeros((shape[0], num[0])).to(torch.float32), torch.zeros((shape[0], num[0])).to(torch.int32)])

    @op_test.only_910b
    def test_TopKDescFP32_8x6_3(self):
        shape = (8, 6)
        num = [3]
        self.in_tensors = [torch.empty(shape, dtype=torch.float32).uniform_(-2, 2)]
        self.set_param("SortOperation", {"num": num})
        self.execute(self.in_tensors,
                     [torch.zeros((shape[0], num[0])).to(torch.float32), torch.zeros((shape[0], num[0])).to(torch.int32)])

    @op_test.only_910b
    def test_TopKDescFP32_128x55296_1(self):
        # typical case from LLM model
        shape = (128, 55296)
        num = [1]
        self.in_tensors = [torch.empty(shape, dtype=torch.float32).uniform_(-100, 100)]
        self.set_param("SortOperation", {"num": num})
        self.execute(self.in_tensors,
                     [torch.zeros((shape[0], num[0])).to(torch.float32), torch.zeros((shape[0], num[0])).to(torch.int32)])


if __name__ == '__main__':
    unittest.main()
