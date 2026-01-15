#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
import sys
import os
import unittest
import torch
import logging
from itertools import product

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "FaUpdateOperation"
faUpdateType = 0


class TestDecodeUpdate(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        all_lse = self.input0
        all_out = self.input1
        sp = all_out.shape[0]
        hd = all_out.shape[-1]

        all_lse = all_lse.transpose(0, 1)
        all_out = all_out.permute(1,2,0).reshape(-1, sp*hd)

        # (b * s * hc, sp)
        lse_exp = torch.exp(all_lse)
        # (b * s * hc, 1)
        sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True)
        # (b * s * hc, sp)
        sum_lse_exp = sum_lse_exp.repeat(1, sp)
        lse_exp = lse_exp/sum_lse_exp

        # oi = lse_exp*oi (b * s * hc, hd, sp)*(b*s*hc, hd, sp)
        lse_exp = lse_exp.unsqueeze(1)
        lse_exp = lse_exp.repeat(1, hd, 1)
        all_out = all_out.reshape(-1, hd, sp)
        all_out = all_out * lse_exp

        # o = sum(oi) (b*s*hc, hd)
        all_out = torch.sum(all_out, dim=-1, keepdim=True)
        return [all_out]

    def golden_compare(self, out_tensors, golden_out_tensors):
        rtol = 1e-5
        atol = 1e-5
        etol = 1e-3
        output = out_tensors.to(torch.float32).reshape((-1))
        golden = golden_out_tensors.to(torch.float32).reshape((-1))
        if output.dtype == torch.bfloat16:
            rtol = 1e-3
            atol = 1e-3
        elif golden.dtype == torch.float32:
            rtol = 1e-6
            atol = 1e-6
        different_element_results = torch.isclose(output.cpu(), golden, rtol=rtol, atol=atol, equal_nan=True)
        different_element_indexes = torch.nonzero(different_element_results == False)

        for index in range(len(different_element_indexes)):
            real_index = different_element_indexes[index]
            golden_data = golden[real_index]
            output_data = output[real_index]
            print("data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
                  (real_index, golden_data, output_data, abs(output_data - golden_data) / golden_data))
        error_ratio = float(different_element_indexes.nelement()) / golden.nelement()
        print("Accuracy is: %.4f, tolrence: %.4f \n" % (1-error_ratio, etol))
        return error_ratio <= etol

    def execute_faupdate(self, hDim, b, s, sp, hc):
        shape0 = (sp, b*s*hc)
        shape1 = (sp, b*s*hc, hDim)
        out_shape = (b*s*hc, hDim)

        #input0: lse   input1: go
        self.input0 = torch.rand(shape0, dtype=torch.float32)
        self.input1 = torch.rand(shape1, dtype=torch.float32)
        output0 = torch.zeros(b*s*hc*hDim, dtype=torch.float32).reshape(out_shape).npu()

        OP_PARAM = {"faUpdateType": faUpdateType, "sp": sp}
        self.execute_out(OP_NAME, OP_PARAM, [self.input0.npu(), self.input1.npu()], [output0])

    def test_fp32_case1_sp2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case1 sp = 2 starting")
        hDim = 8
        b = 11
        s = 1
        sp = 2
        hc = 13
        self.execute_faupdate(hDim, b, s, sp, hc)

    def test_fp32_case2_sp4(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case2 sp = 4 starting")
        hDim = 16
        b = 7
        s = 1
        sp = 4
        hc = 6
        self.execute_faupdate(hDim, b, s, sp, hc)

    def test_fp32_case3_sp8(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case3 sp = 8 starting")
        hDim = 32
        b = 31
        s = 1
        sp = 8
        hc = 32
        self.execute_faupdate(hDim, b, s, sp, hc)

    def test_fp32_case4_largeBs(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case4 large BS starting")
        hDim = 128
        b = 1000
        s = 1
        sp = 3
        hc = 13
        self.execute_faupdate(hDim, b, s, sp, hc)

    def test_fp32_case5_smallBs(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case5 small BS starting")
        hDim = 8
        b = 2
        s = 1
        sp = 1
        hc = 1
        self.execute_faupdate(hDim, b, s, sp, hc)

    def test_fp32_case6_Prefill(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case6 Prefill starting")
        hDim = 128
        b = 2
        s = 4000
        sp = 2
        hc = 32
        self.execute_faupdate(hDim, b, s, sp, hc)
        
    def test_fp32_case7_Prefill(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print("TEST case6 Prefill starting")
        hDims = [16, 256, 512]
        batchs = [1, 2, 10]
        seqlens = [1, 16, 64]
        sps = [1, 5, 10]
        hcs = [1, 10, 16, 32]
        for hDim, batch, seqlen, sp, hc in product(hDims, batchs, seqlens, sps, hcs):
            self.execute_faupdate(hDim, batch, seqlen, sp, hc)


if __name__ == '__main__':
    unittest.main()
