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
import math
import unittest
import numpy as np
import sys, os
import op_test
import torch


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def align_up(val: int, align: int) -> int:
    if (align == 0):
        return 0
    return ((val + align - 1) // align) * align


def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0) * a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0, 2])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m, n0 = x_shape

    # reshape为3维，对最后2维进行pad
    pad_num1 = align_up(x.shape[-1], 16) - x.shape[-1]
    pad_num2 = align_up(x.shape[-2], 16) - x.shape[-2]
    x = x.reshape([-1] + list(x.shape[-2:]))
    x = np.pad(x, ((0, 0), (0, pad_num2), (0, pad_num1)))

    x = x.reshape(x_shape[:-3] + [m, n1, n0]).transpose(*array_trans)
    x = np.ascontiguousarray(x)
    return x


class TestKvcache(op_test.OpTest):
    def case_param_gen(self, case_params_list):
        self.format = case_params_list['format']
        self.layer = case_params_list['layer']
        self.layer_id = case_params_list['layer_id']
        self.batch = case_params_list['batch']
        self.max_seqlen = case_params_list['max_seqlen']
        self.hidden_size = case_params_list['hidden_size']

    def golden_calc(self, in_tensors):
        newkv = self.newkv
        token_offset = self.token_offset
        seqlen = self.seqlen

        cache_out = np.zeros(shape=(self.layer, self.batch, self.max_seqlen, self.hidden_size)).astype(newkv.dtype)
        prefix_ntokens = 0
        for i in range(self.batch):
            for j in range(seqlen[i]):
                cache_out[self.layer_id][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
        if self.format == self.format_nz:
            cache_out = convert_nd_to_nz(cache_out)
        return [torch.tensor(cache_out)]

    def golden_compare(self, out_tensors, golden_tensors):
        if (out_tensors[0].dtype == torch.bfloat16):
            return torch.allclose(out_tensors[0].bfloat16(), golden_tensors[0].bfloat16(), rtol=0.001, atol=0.001)
        return torch.allclose(out_tensors[0], golden_tensors[0], rtol=0.001, atol=0.001)
    # kvcache nd
    @op_test.only_910b
    def test_kvcache_case0(self):

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nz 对齐
    @op_test.only_310p
    def test_kvcache_case1(self):

        layer = 1
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 3}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i] + 4
        ntokens = np.sum(seqlen)
        ntokens_ceil = np.sum((seqlen + 15) // 16 * 16)
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nz 非对齐
    @op_test.only_310p
    def test_kvcache_case2(self):

        layer = 1
        layer_id = 0
        batch = 12
        max_seqlen = 1028
        hidden_size = 2047

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 3}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, 1024, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i] + 4
        ntokens = np.sum(seqlen)
        ntokens_ceil = np.sum((seqlen + 15) // 16 * 16)
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nd
    @op_test.only_910b
    def test_kvcache_case3(self):

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).bfloat16(), torch.tensor(layer_id), torch.tensor(cache_in).bfloat16(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    @op_test.only_910b
    def test_kvcache_nd_int8(self):

        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.int8)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.int8)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        return self.execute([torch.tensor(newkv), torch.tensor(layer_id), torch.tensor(cache_in),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    # kvcache nz params对齐
    @op_test.only_310p
    def test_kvcache_case4(self):
        layer = 1
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i] + 4
        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 9, "seqLen": seqlen.tolist(), "tokenOffset": token_offset.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        
        ntokens = np.sum(seqlen)
        ntokens_ceil = np.sum((seqlen + 15) // 16 * 16)
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half()],
                            [2])

    # kvcache nd params
    @op_test.only_910b
    def test_kvcache_case5(self):
        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 7, "seqLen": seqlen.tolist(), "tokenOffset": token_offset.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 3)
        self.set_output_formats([self.format_nd])

        case_params_list = {'format' : self.format_nd, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)
        
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half()],
                            [2])

    # kvcache nz 对齐
    @op_test.only_310p
    def test_kvcache_case6(self):
        layer = 5
        layer_id = 4
        batch = 158
        max_seqlen = 2048
        hidden_size = 52*64

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 3}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        case_params_list = {'format' : self.format_nz, 'layer': layer, 'layer_id' : layer_id, 'batch': batch,
                            'max_seqlen': max_seqlen, 'hidden_size': hidden_size}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = np.zeros(shape=(batch)).astype(np.int32)
        for i in range(batch):
            token_offset[i] = seqlen[i]
        ntokens = np.sum(seqlen)
        ntokens_ceil = np.sum((seqlen + 15) // 16 * 16)
        newkv = np.random.uniform(-5, 5, size=(ntokens_ceil, hidden_size)).astype(np.float16)
        for i in range(ntokens, ntokens_ceil):
            newkv[i][:] = [0] * hidden_size
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        # 用于生成golden数据
        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        newkv = newkv.reshape([1] + list(newkv.shape[-2:]))
        newkv = convert_nd_to_nz(newkv)
        cache_in = convert_nd_to_nz(cache_in)
        cache_out = np.zeros_like(cache_in)

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])


if __name__ == '__main__':
    unittest.main()
