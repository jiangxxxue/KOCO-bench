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
import csv
from precision_compare import data_compare
np.random.seed(123)
random.seed(123)
MAX_SEQ_LEN = 1024
atten_mask_value = -65536
scale_value = 0.08838834764831843
keep_prob = 1.0
input_layout, pre_tokens, next_tokens = "BNSD", 65536, 1
OP_NAME = "LaserAttentionOperation"

class TestLaserAttention(op_test.OpTest):
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

    def golden_calc(self, in_tensors):
        query = in_tensors[0]
        key = in_tensors[1]
        value = in_tensors[2]
        atten_mask = in_tensors[5]

        q_head_num = query.shape[1]
        k_head_num = key.shape[1]
        window_len = self.windowLen
        if q_head_num == k_head_num:
            softmax_max, softmax_sum, softmax_log_max_sum, attention_score = self.attention_ori(query, key, value, atten_mask, scale_value, self.is_triangle == 1, window_len)
        else:
            softmax_max, softmax_sum, softmax_log_max_sum, attention_score = self.attention_ori_GQA(query, key, value, atten_mask, scale_value, self.is_triangle == 1, window_len)
        return softmax_max, softmax_sum, attention_score

    def data_compare_np(self, npu_output, cpu_output, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1, is_display=True,
                    is_simple=True, name=None):
        max_error_idx = 10000000
        real_data = npu_output.flatten()
        data_compe = cpu_output.flatten()
        if real_data.size == 0 and real_data.size == data_compe.size:
            print('The npu_output is [],and it is same as bm_output, the result of data_compare is \"Pass\"')
            return "Pass", 100.0, 0, 0
        start = 0
        end = real_data.size - 1
        if end < start:
            end = start
        max_error = 0
        result = "Failed"
        if real_data.size != data_compe.size:
            print(
                'Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
            return result, 0.0, 0, 0

        comp_overflows_count = data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
        real_overflows_count = real_data[np.isinf(real_data)].size + real_data[np.isnan(real_data)].size
        split_count = int(end - start + 1) if end != start else 1

        try:
            diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))  # 绝对误差
        except MemoryError:
            return result, 0.0, 0, 0
        diff_index = np.where(diff_abs > 0)  # 所有绝对差值不为0的位置
        rdiff, diff_relate = self.cal_relative_diff_np(real_data[diff_index].astype(np.float32), data_compe[diff_index].astype(np.float32),
                                                diff_thd)
        max_diff_relate = 0 if diff_relate.size == 0 else max(diff_relate)
        err_diff = rdiff[rdiff > diff_thd]  # 差值大于阈值的结果
        fulfill_percent = float(split_count - err_diff.size) / float(split_count) * 100.0  # （所有数目-超过阈值的数目）/所有数目

        pct_thd = (1 - pct_thd) * 100.0
        result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
        fulfill_percent = float('%.4f' % fulfill_percent)

        if len(err_diff) > 0:
            max_error = max(err_diff[0:max_error_idx])
            # 关掉最大误差超过0.1，当前版本不要求
            '''
            if max_error >= max_diff_hd:
                result = "Failed"
            '''
        if comp_overflows_count:
            print("标杆数据存在inf/nan")
        if real_overflows_count and not comp_overflows_count:
            print("Npu结果存在inf/nan")
            max_diff_relate = np.max(npu_output) if max_diff_relate == 0 else max_diff_relate
            result = 'Failed'
        if is_display:
            print("[{:2}] 相对误差小于[{:.3f}]的值占比[{:7.3f}] 最大相对误差[{:9.7f}] 最大绝对误差[{:9.7f}] [{}]".format(name, diff_thd, fulfill_percent, max_diff_relate, max(diff_abs), result))

        return [result, fulfill_percent, max(diff_abs), max_diff_relate]

    def cal_relative_diff_np(self, real_data, expect_data, diff_thd):
        a = np.abs(np.subtract(real_data, expect_data))
        b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
        b2 = float((1.0 / (1 << 14)) / diff_thd)
        b = np.add(np.maximum(b1, b2), 10e-10)
        result = np.where(a < diff_thd, a, a / b)  # 如果绝对误差<阈值返回绝对误差，否则返回相对误差
        return result, a / b
        
    def golden_compare(self, out_tensors, golden_tensors):
        data_compare(out_tensors[0].float().numpy(), golden_tensors[0].float().numpy())
        data_compare(out_tensors[1].float().numpy(), golden_tensors[1].float().numpy())
        data_compare(out_tensors[2].float().numpy(), golden_tensors[2].float().numpy())
        return True, True, True

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

    def gen_data(self, input_shape ,seq_size, dtype, data_range):
        if data_range == 100:
            mean = np.random.uniform(-100, 100)
            std_dev = np.random.uniform(1, 25)
            query = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[0])).to(dtype)
            key = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[1])).to(dtype)
            value = torch.from_numpy(np.random.normal(mean, std_dev, input_shape[1])).to(dtype)
        else:
            query = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[0])).type(dtype)
            key = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[1])).type(dtype)
            value = torch.from_numpy(np.random.uniform(-data_range, data_range, input_shape[1])).type(dtype)
        atten_mask = torch.from_numpy(1 - np.tri(seq_size)).to(torch.bfloat16)
        
        return query, key, value, atten_mask

    def atest_forward(self):
        self.set_support_910b_only()
        dtype = torch.bfloat16
        dtype_string = "bfloat16"
        data_range = 10
        ori_shape = [(1, 1, 256, 128), (1, 1, 256, 128), 1, 256]
        batch_size, head_num, seq_size, head_dim = ori_shape[0]
        self.is_triangle = ori_shape[2]
        self.windowLen = ori_shape[3]
        query, key, value, atten_mask = self.gen_data(ori_shape, seq_size, dtype, data_range)
        pre_tokens = ori_shape[3]
        
        OP_PARAM = {"type":0, "scale":scale_value, 
            "headNum":head_num, "ioLayout": input_layout, 
            "keepProb": keep_prob, "preTokens": pre_tokens, 
            "nextTokens": next_tokens, "isHighPrecision": 1, "dtype": dtype_string}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 6)
        self.set_output_formats([self.format_nd] * 3)

        softmaxMax = torch.zeros([batch_size, head_num, seq_size]).to(torch.float)
        softmaxSum = torch.zeros([batch_size, head_num, seq_size]).to(torch.float)
        attentionOut = torch.zeros([batch_size, head_num, seq_size, head_dim]).to(torch.float)
        self.execute(
                [
                    query, key, value, torch.Tensor().to(torch.bfloat16), torch.Tensor().to(torch.int8), atten_mask if self.is_triangle == 1 else torch.Tensor().to(atten_mask.dtype)
                ],
                [
                    softmaxMax,
                    softmaxSum,
                    attentionOut
                ]
        )


if __name__ == '__main__':
    unittest.main()
