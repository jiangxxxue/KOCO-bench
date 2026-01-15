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
import sys
import unittest
import math
import numpy as np
import os
import op_test
import torch
import random
import sys
import numpy as np
import math
from precision_compare import data_compare
import csv
rand_seed = 123
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)
MAX_SEQ_LEN = 1024
atten_mask_value = -65536
scale_value = 0.08838834764831843
keep_prob = 1.0
input_layout, pre_tokens, next_tokens = "BNSD", 65536, 1
OP_NAME = "LaserAttentionGradOperation"

class TestLaserAttentionGrad(op_test.OpTest):    
    def attention_ori(self, query, key, value, atten_mask, scale_value, is_triangle, window_len):
        """
        (b, n, s, d)
        """
        query = query.to(torch.float)
        key = key.to(torch.float)
        value = value.to(torch.float)
        batch_size, head_num, seq_size, head_dim = query.shape

        bmm1_res = self.torch_batch_dot(query, torch.transpose(key, 2, 3)).to(torch.float)
        if is_triangle:
            sparse_mask = torch.from_numpy(np.tri(seq_size - window_len, k=-1)).to(torch.float)
            bmm1_res = bmm1_res * scale_value + (atten_mask * atten_mask_value)
            if window_len < seq_size:
                bmm1_res[:,:, window_len : , 0 : seq_size - window_len] = bmm1_res[:,:, window_len : , 0 : seq_size - window_len] + sparse_mask * -65536
        else:
            bmm1_res = bmm1_res * scale_value
        softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
        softmax_sub = bmm1_res - softmax_max
        softmax_exp = torch.exp(softmax_sub)
        softmax_sum = torch.from_numpy(np.sum(softmax_exp.numpy(), -1, keepdims=True))
        softmax_out = softmax_exp / softmax_sum
        softmax_out = softmax_out.to(torch.float)
        attention_out = self.torch_batch_dot(softmax_out, value)
        softmax_log_max_sum = softmax_max + torch.log(softmax_sum)

        return softmax_max, softmax_sum, softmax_log_max_sum, attention_out

    def attention_ori_GQA(self, query_all, key_all, value_all, atten_mask, scale_value, is_triangle, window_len):
        """
        (b, n, s, d)
        """
        nG = query_all.shape[1] // key_all.shape[1]
        G = key_all.shape[1]
        batch_size, head_num, seq_size, head_dim = query_all.shape

        for i in range(query_all.shape[1]):
            g = i // nG
            query = query_all[:, i:i+1, :, :].to(torch.float)
            key = key_all[:, g:g+1, :, :].to(torch.float)
            value = value_all[:, g:g+1, :, :].to(torch.float)

            bmm1_res = self.torch_batch_dot(query, torch.transpose(key, 2, 3)).to(torch.float)
            if not is_triangle:
                bmm1_res = bmm1_res * scale_value
            else:
                sparse_mask = torch.from_numpy(np.tri(seq_size - window_len, k=-1)).to(torch.float)
                bmm1_res = bmm1_res * scale_value + (atten_mask * atten_mask_value)
                if window_len < seq_size:
                    bmm1_res[:,:,window_len : , 0 : seq_size - window_len] = bmm1_res[:,:,window_len : , 0 : seq_size - window_len] + sparse_mask * -65536
            softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
            softmax_sub = bmm1_res - softmax_max
            softmax_exp = torch.exp(softmax_sub)
            softmax_sum = torch.from_numpy(np.sum(softmax_exp.numpy(), -1, keepdims=True))
            softmax_out = softmax_exp / softmax_sum

            softmax_out = softmax_out.to(torch.float)
            attention_out = self.torch_batch_dot(softmax_out, value)

            softmax_log_max_sum = softmax_max + torch.log(softmax_sum)

            if i == 0:
                total_softmax_max = softmax_max
                total_softmax_sum = softmax_sum
                total_softmax_log_max_sum = softmax_log_max_sum
                total_attention_out = attention_out

            else:
                total_softmax_max =  torch.cat((total_softmax_max, softmax_max), dim=1)
                total_softmax_sum =  torch.cat((total_softmax_sum, softmax_sum), dim=1)
                total_softmax_log_max_sum = torch.cat((total_softmax_log_max_sum, softmax_log_max_sum), dim=1)
                total_attention_out = torch.cat((total_attention_out, attention_out), dim=1)

        return total_softmax_max, total_softmax_sum, total_softmax_log_max_sum, total_attention_out

    def attention_grad_ori(self, query, key, value, atten_mask, softmax_log_max_sum, attention_score, attention_score_grad, scale_value, is_triangle, window_len, dtype):
        """
        (b, n, s,  d)
        """
        batch_size, head_num, seq_size, head_dim = query.shape

        query = query.to(torch.float)
        key = key.to(torch.float)
        value = value.to(torch.float)
        attention_score = attention_score.to(dtype)
        attention_score_grad = attention_score_grad.to(dtype)
        attention_score = attention_score.to(torch.float)
        attention_score_grad = attention_score_grad.to(torch.float)

        softmax_log_max_sum = softmax_log_max_sum.view((batch_size, head_num, seq_size, 1))
        softmax_out_sum = torch.from_numpy(
            (attention_score.to(torch.float) * attention_score_grad.to(torch.float)).numpy().sum(-1)).reshape(
            (batch_size, head_num, seq_size, 1))

        # intermediate calculation as fp32
        query_key_mul = self.torch_batch_dot(query, torch.transpose(key, 2, 3)).to(torch.float)
        softmax_grad = self.torch_batch_dot(attention_score_grad, torch.transpose(value, 2, 3)).to(torch.float)
        print(is_triangle)
        if is_triangle:
            sparse_mask = torch.from_numpy(np.tri(seq_size - window_len, k=-1)).to(torch.float)
            bmm1_res_drop = query_key_mul * scale_value + (atten_mask * atten_mask_value)
            if window_len < seq_size:
                bmm1_res_drop[:, :, window_len:, 0: seq_size - window_len] = bmm1_res_drop[:, :, window_len:, 0: seq_size - window_len] + sparse_mask * atten_mask_value
        else:
            bmm1_res_drop = query_key_mul * scale_value

        dS_TMP = bmm1_res_drop - softmax_log_max_sum
        softmax_out = torch.exp(bmm1_res_drop - softmax_log_max_sum)

        bmm1_res_drop_grad_flash = (softmax_grad - softmax_out_sum) * softmax_out
        bmm1_res_grad = bmm1_res_drop_grad_flash * scale_value

        print("old generate dV dQ dK")
        value_grad = self.torch_batch_dot(torch.transpose(softmax_out, 2, 3).to(torch.float), attention_score_grad)
        key_grad = self.torch_batch_dot(torch.transpose(bmm1_res_grad, 2, 3).to(torch.float), query)
        query_grad = self.torch_batch_dot(bmm1_res_grad.to(torch.float), key)

        # print("new generate dV dQ dK")
        # value_grad = self.torch_batch_dot(torch.transpose(softmax_out, 2, 3).to(torch.float), attention_score_grad)
        # key_grad = self.torch_batch_dot(torch.transpose(softmax_grad, 2, 3).to(torch.float), query)
        # query_grad = self.torch_batch_dot(softmax_grad.to(torch.float), key)
        
        return query_grad, key_grad, value_grad


    def attention_grad_ori_GQA(self, query_all, key_all, value_all, atten_mask, softmax_log_max_sum_all, attention_score_all, attention_score_grad_all,
                            scale_value, is_triangle, window_len):
        """
        (b, n, s,  d)
        """
        # value_grad, key_grad, query_grad
        nG = query_all.shape[1] // key_all.shape[1]
        G = key_all.shape[1]
        batch_size, head_num, seq_size, head_dim = query_all.shape

        query_grad = torch.zeros_like(query_all)
        key_grad = torch.zeros_like(key_all)
        value_grad = torch.zeros_like(value_all)

        for i in range(query_all.shape[1]):
            g = i // nG
            attention_score = attention_score_all[:, i:i + 1, :, :].to(torch.float)
            attention_score_grad = attention_score_grad_all[:, i:i + 1, :, :].to(torch.float)
            softmax_log_max_sum = softmax_log_max_sum_all[:, i:i + 1, :].to(torch.float)
            query = query_all[:, i:i + 1, :, :].to(torch.float)

            key = key_all[:, g:g + 1, :, :].to(torch.float)
            value = value_all[:, g:g + 1, :, :].to(torch.float)

            batch_size, head_num, qSeqlen, head_dim = query.shape
            batch_size, head_num, kSeqlen, head_dim = key.shape

            softmax_log_max_sum = softmax_log_max_sum.view((batch_size, head_num, qSeqlen, 1))
            softmax_out_sum = torch.from_numpy(
                (attention_score.to(torch.float) * attention_score_grad.to(torch.float)).numpy().sum(-1)).reshape(
                (batch_size, head_num, qSeqlen, 1))

            # intermediate calculation as fp32
            query_key_mul = self.torch_batch_dot(query, torch.transpose(key, 2, 3)).to(torch.float)
            softmax_grad = self.torch_batch_dot(attention_score_grad, torch.transpose(value, 2, 3)).to(torch.float)

            if not is_triangle:
                bmm1_res_drop = query_key_mul * scale_value
            else:
                sparse_mask = torch.from_numpy(np.tri(seq_size - window_len, k=-1)).to(torch.float)
                bmm1_res_drop = query_key_mul * scale_value + (atten_mask * atten_mask_value)
                if window_len < seq_size:
                    bmm1_res_drop[:, :, window_len:, 0: seq_size - window_len] = bmm1_res_drop[:, :, window_len:, 0: seq_size - window_len] + sparse_mask * atten_mask_value
            softmax_out = torch.exp(bmm1_res_drop - softmax_log_max_sum)

            bmm1_res_drop_grad_flash = (softmax_grad - softmax_out_sum) * softmax_out
            bmm1_res_grad = bmm1_res_drop_grad_flash * scale_value

            query_grad[:, i:i + 1, :, :] = self.torch_batch_dot(bmm1_res_grad.to(torch.float), key)

            key_grad[:, g:g + 1, :, :] += self.torch_batch_dot(torch.transpose(bmm1_res_grad, 2, 3).to(torch.float), query)
            value_grad[:, g:g + 1, :, :] += self.torch_batch_dot(torch.transpose(softmax_out, 2, 3).to(torch.float), attention_score_grad)

        return query_grad, key_grad, value_grad

    def golden_calc(self, in_tensors):
        query = in_tensors[0]
        key = in_tensors[1]
        value = in_tensors[2]
        attention_score_grad = in_tensors[3]
        atten_mask = in_tensors[6]
        softmax_max = in_tensors[7]
        softmax_sum = in_tensors[8]
        softmax_log_max_sum = softmax_max + torch.log(softmax_sum)
        attention_score = in_tensors[9]

        q_head_num = query.shape[1]
        k_head_num = key.shape[1]
        window_len = self.windowLen
        dtype = torch.bfloat16

        if q_head_num == k_head_num:
            query_grad, key_grad, value_grad = self.attention_grad_ori(query, key, value, atten_mask, softmax_log_max_sum, attention_score, attention_score_grad, scale_value, self.is_triangle == 1, window_len, dtype)
        else:
            query_grad, key_grad, value_grad = self.attention_grad_ori_GQA(query, key, value, atten_mask, softmax_log_max_sum, attention_score, attention_score_grad, scale_value, self.is_triangle == 1, window_len)

        return query_grad, key_grad, value_grad

    def golden_compare(self, out_tensors, golden_tensors):
        data_compare(out_tensors[0].float().numpy(), golden_tensors[0].float().numpy())
        data_compare(out_tensors[1].float().numpy(), golden_tensors[1].float().numpy())
        data_compare(out_tensors[2].float().numpy(), golden_tensors[2].float().numpy())
        result_1 = torch.allclose(out_tensors[0].float(), golden_tensors[0].float(), rtol=0.005, atol=0.005)
        result_2 = torch.allclose(out_tensors[1].float(), golden_tensors[1].float(), rtol=0.005, atol=0.005)
        result_3 = torch.allclose(out_tensors[2].float(), golden_tensors[2].float(), rtol=0.005, atol=0.005)
        return result_1, result_2, result_3

    def torch_batch_dot(self, a, b):
        batch, n, h, _ = a.shape
        _, _, _, w = b.shape
        a = a.to(torch.float)
        b = b.to(torch.float)
        res = torch.zeros((batch, n, h, w), dtype=torch.float)
        for i in range(batch):
            for j in range(n):
                res[i, j] = torch.matmul(a[i, j], b[i, j])
        return res

    def gen_data(self, input_shape ,seq_size, dtype, data_range, is_triangle):
        data_range = 100
        if data_range == 100:
            mean = np.random.uniform(-100, 100)
            std_dev = np.random.uniform(1, 25)
            B, N, S, D = input_shape[0]
            query = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[0])).to(dtype)
            key = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[1])).to(dtype)
            value = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[1])).to(dtype)
            attention_score_grad = (torch.rand(input_shape[0]).to(dtype) - 0.5)
        else:
            query = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[0])).type(dtype)
            key = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[1])).type(dtype)
            value = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[1])).type(dtype)
            attention_score_grad = (torch.rand(input_shape[0]).to(dtype) - 0.5)
        atten_mask = torch.from_numpy(1 - np.tri(seq_size)).to(torch.float16)
        batch_size, head_num, seq_size, head_dim = input_shape[0]
        q_head_num = query.shape[1]
        k_head_num = key.shape[1]
        window_len = query.shape[2]
        if q_head_num == k_head_num:
            softmax_max, softmax_sum, softmax_log_max_sum, attention_score = self.attention_ori(query, key, value, atten_mask, scale_value, is_triangle, window_len)
        else:
            softmax_max, softmax_sum, softmax_log_max_sum, attention_score = self.attention_ori_GQA(query, key, value, atten_mask, scale_value, is_triangle, window_len)
        return query, key, value, softmax_max, softmax_sum, attention_score.to(dtype), attention_score_grad, atten_mask

    def atest_backward(self):
        self.set_support_910b_only()
        ori_shape = [(1, 1, 256, 128), (1, 1, 256, 128), 1, 256]
        dtype = torch.bfloat16
        dtype_string = "bfloat16"
        batch_size, head_num, seq_size, head_dim = ori_shape[0]
        self.is_triangle = ori_shape[2]
        self.windowLen = ori_shape[3]
        data_range = 100
        query, key, value, softmax_max, softmax_sum, attention_score, attention_score_grad, atten_mask = self.gen_data(ori_shape, seq_size, dtype, data_range, self.is_triangle)
        
        self.windowLen = ori_shape[3]
        pre_tokens = ori_shape[3]

        OP_PARAM = {"type":0, "scale":scale_value, 
            "headNum":head_num, "ioLayout": input_layout, 
            "keepProb": keep_prob, "preTokens": pre_tokens, 
            "nextTokens": next_tokens, "isHighPrecision": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 10)
        self.set_output_formats([self.format_nd] * 3)

        queryGrad = torch.zeros_like(query).to(torch.float32)
        keyGrad = torch.zeros_like(key).to(torch.float32)
        valueGrad = torch.zeros_like(value).to(torch.float32)
        
        self.execute(
                [
                    query, key, value,
                    attention_score_grad, torch.Tensor().to(torch.bfloat16), torch.Tensor().to(torch.int8), 
                    atten_mask if self.is_triangle == 1 else torch.Tensor().to(atten_mask.dtype), softmax_max.squeeze(dim=3), softmax_sum.squeeze(dim=3), attention_score
                ],
                [
                    queryGrad,
                    keyGrad,
                    valueGrad
                ]
        )

if __name__ == '__main__':
    unittest.main()
