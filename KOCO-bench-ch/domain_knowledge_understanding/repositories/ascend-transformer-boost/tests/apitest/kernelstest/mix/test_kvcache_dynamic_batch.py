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
import numpy as np
import sys, os
import op_test
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TestKvcache(op_test.OpTest):
    def case_param_gen(self, case_params_list):
        self.batch = case_params_list['batch']
        self.max_seqlen = case_params_list['max_seqlen']
        self.hidden_size = case_params_list['hidden_size']
        self.batch_run_status = case_params_list['batch_run_status']

    def golden_calc(self, in_tensors):
        newkv = self.newkv
        token_offset = self.token_offset
        seqlen = self.seqlen

        cache_out = np.zeros(shape=(self.batch, self.max_seqlen, self.hidden_size)).astype(np.float16)
        prefix_ntokens = 0
        for i in range(self.batch):
            if self.batch_run_status[i] == 0:
                prefix_ntokens += seqlen[i]
                continue
            for j in range(seqlen[i]):
                cache_out[i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
        return [torch.tensor(cache_out).half()]

    def golden_compare(self, out_tensors, golden_tensors):
        return torch.allclose(out_tensors[0], golden_tensors[0], rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_kvcache_case0(self):

        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048
        batch_run_status = [1] * batch

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 5, "batchRunStatus" : batch_run_status}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'batch': batch, 'max_seqlen': max_seqlen, 'hidden_size': hidden_size, "batch_run_status" : batch_run_status}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    @op_test.only_910b
    def test_kvcache_case1(self):

        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048
        batch_run_status = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 5, "batchRunStatus" : batch_run_status}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'batch': batch, 'max_seqlen': max_seqlen, 'hidden_size': hidden_size, "batch_run_status" : batch_run_status}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    @op_test.only_910b
    def test_kvcache_case2(self):

        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024
        batch_run_status = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]

        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 5, "batchRunStatus" : batch_run_status}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd])

        case_params_list = {'batch': batch, 'max_seqlen': max_seqlen, 'hidden_size': hidden_size, "batch_run_status" : batch_run_status}
        self.case_param_gen(case_params_list)

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        cache_out = np.zeros_like(cache_in)
        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half(),
                             torch.tensor(token_offset), torch.tensor(seqlen)],
                            [2])

    @op_test.only_910b
    def test_kvcache_case3(self):
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 2048
        batch_run_status = [1] * batch
        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        OP_NAME = "KVCacheOperation"
        OP_PARAM = {"type": 11, "batchRunStatus" : batch_run_status, "seqLen": seqlen.tolist(), "tokenOffset": token_offset.tolist()}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 3)
        self.set_output_formats([self.format_nd])

        case_params_list = {'batch': batch, 'max_seqlen': max_seqlen, 'hidden_size': hidden_size, "batch_run_status" : batch_run_status}
        self.case_param_gen(case_params_list)

        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)

        self.newkv = newkv
        self.token_offset = token_offset
        self.seqlen = seqlen

        return self.execute([torch.tensor(newkv).half(), torch.tensor(layer_id), torch.tensor(cache_in).half()],
                            [2])

if __name__ == '__main__':
    unittest.main()
