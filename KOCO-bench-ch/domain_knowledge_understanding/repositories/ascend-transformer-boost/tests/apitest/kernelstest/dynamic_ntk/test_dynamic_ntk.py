#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import torch
import op_test
import logging


class TestDynamicNTk(op_test.OpTest):
    def __gen_golden(self, position_ids, inv_freqs, seq_lens, out_type = 0):
        off = 0
        num_tokens = position_ids.shape[0]
        dim = inv_freqs.shape[1] * 2
        batch_num = seq_lens.shape[0]
        otype = torch.float16 if out_type == 0 else torch.bfloat16
        sinOut = torch.zeros([num_tokens, dim], dtype=torch.float32)
        cosOut = torch.zeros([num_tokens, dim], dtype=torch.float32)
        for batch_id in range(batch_num):
            pos_len = seq_lens[batch_id]
            freqs = torch.einsum('i,j->ij', position_ids[off:off + pos_len].to(torch.float32), inv_freqs[batch_id])
            emb = torch.cat((freqs, freqs), dim = -1)
            cosOut[off:off + pos_len, :] = emb.cos()
            sinOut[off:off + pos_len, :] = emb.sin()
            off += pos_len
        return sinOut.to(otype), cosOut.to(otype)

    def __gen_test_data(self, batch, num_tokens, dim, max_seq_len, out_type=0):
        aux_array = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        batch_base = torch.randint(10000, 50000, [batch], dtype=torch.float32)
        position_ids = torch.randint(0, max_seq_len, [num_tokens], dtype=torch.int32)
        inv_freqs = torch.zeros([batch, int(dim / 2)], dtype=torch.float32)
        for i in range(batch):
            inv_freqs[i, :] = 1.0 / batch_base[i] ** aux_array

        avg_seq_len = int(num_tokens / batch)
        seq_lens = torch.ones([batch], dtype=torch.int32) * avg_seq_len
        seq_lens[0] += num_tokens - avg_seq_len * batch
        self.golden_sin, self.golden_cos = self.__gen_golden(position_ids, inv_freqs, seq_lens, out_type)
        self.out_type = int(out_type)
        return position_ids, inv_freqs, seq_lens

    def golden_calc(self, in_tensors):
        return [self.golden_sin, self.golden_cos]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        rtol = 2 ** (-7) if self.out_type == torch.bfloat16 else 2 ** (-8)
        atol = rtol
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=rtol, atol=atol)

    @op_test.skip_310b
    @op_test.skip_910a
    def testcase_2_8_32_0(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(3, 8, 32, 128000, 0)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).half(), torch.zeros(self.golden_cos.shape).half()],
        )

    @op_test.skip_310b
    @op_test.skip_910a
    def testcase_3_8_32_0(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(3, 8, 32, 128000, 0)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).half(), torch.zeros(self.golden_cos.shape).half()],
        )

    @op_test.skip_310b
    @op_test.skip_910a
    def testcase_3_129_32_0(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(3, 129, 32, 128000, 0)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).half(), torch.zeros(self.golden_cos.shape).half()],
        )

    @op_test.skip_310b
    @op_test.skip_910a
    def testcase_3_5568_128_0(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(3, 5568, 128, 128000, 0)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).half(), torch.zeros(self.golden_cos.shape).half()],
        )

    @op_test.skip_310b
    @op_test.skip_910a
    def testcase_16_5568_128_0(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(16, 5568, 128, 128000, 0)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).half(), torch.zeros(self.golden_cos.shape).half()],
        )

    @op_test.only_910b
    def testcase_3_129_512_1(self):
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(3, 129, 512, 128000, 1)
        self.set_param(
            "DynamicNTKOperation",
            {"outType": self.out_type},
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd, self.format_nd])
        self.execute(
            [position_ids, inv_freqs, seq_lens],
            [torch.zeros(self.golden_sin.shape).to(torch.bfloat16), torch.zeros(self.golden_cos.shape).to(torch.bfloat16)],
        )

if __name__ == "__main__":
    unittest.main()
