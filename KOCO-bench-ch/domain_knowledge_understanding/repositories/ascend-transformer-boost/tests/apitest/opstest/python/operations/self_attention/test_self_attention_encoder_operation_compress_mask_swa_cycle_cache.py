#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import sys
import unittest
import random
import numpy as np
import torch
import torch_npu

np.random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


def gen_seq_len(batch, max_seq, variate_seq=False):
    if variate_seq:
        num = max_seq // 16
        seqlen_aligned_arange = np.arange(1, num) * 16
        if batch > num:
            seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
            seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
            seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
        else:
            seqlen_aligned = seqlen_aligned_arange
        sp_list = np.random.randint(0, 15, size=(num - 1))
        seqlen = seqlen_aligned - sp_list
        seqlen = seqlen[-batch:]
        seqlen_aligned = seqlen_aligned[-batch:]
        print(seqlen)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
        sp_list = sp_list.astype(np.int32)
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)

    ntokens = seqlen.sum()
    print("ntokens:", ntokens)
    return seqlen, seqlen_aligned, ntokens


def group_matmul(heads, group_num, A, B):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    print(score.shape)
    return score

def gen_swa_cmp(window_size, embeddim):
    swa_mask = np.ones(shape=(1, 512, 512)) * -10000.0
    pp_n = 128 if embeddim <= 128 else 64
    # pp_n = 128
    if window_size <= pp_n * 3:
        true_size = window_size
    else:
        if window_size % pp_n == 0:
            true_size = pp_n * 3
        else:
            true_size = pp_n * 2 + window_size % pp_n
    triu_mask = np.triu(swa_mask, 1)
    tril_mask = np.tril(swa_mask, -true_size)
    swa_mask = triu_mask + tril_mask
    # swa_mask = torch.from_numpy(swa_mask).to(torch.float16)
    swa_mask = swa_mask.reshape(512,512)
    return swa_mask

def calc_expect_func(batch, seqlen, heads, embed, window_size, mask_type, group_num=32):
    is_mask = True
    variate_seq = False
    is_decoder = False
    max_seq = 2048
    src_type = 'float16'
    fp32 = True
    print(f"group_num: {group_num}")
    print("q_seq is:")
    if is_decoder:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, 1, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, seqlen, variate_seq)
    else:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, seqlen, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens   # crossattention时，q_seqlen != k_seqlen

    max_s = np.max(q_seqlen)
    ntokens2 = (q_seqlen * kv_seqlen).sum()
    embed_v = embed

    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v)).astype(np.float16)

    mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
    mask_u = np.triu(mask, 1)
    mask_l = np.tril(mask, -window_size)
    mask = mask_u + mask_l
    mask *= -10000.0

    # print(mask)

    q_offset = 0
    k_offset = 0
    v_offset = 0

    s = None
    _p = None
    out = None

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]
        q_slice = q[q_offset:q_offset + q_s][:]
        q_slice = q_slice.reshape(q_s, heads, embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
        k_slice = k[k_offset:k_offset + kv_s][:]
        k_slice = k_slice.reshape(kv_s, group_num, embed)
        k_slice = np.transpose(k_slice, (1, 0, 2))
        k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
        v_slice = v[v_offset:v_offset + kv_s][:]
        v_slice = v_slice.reshape(kv_s, group_num, embed_v)
        v_slice = np.transpose(v_slice, (1, 0, 2))
        score = group_matmul(heads, group_num, q_slice, k_slice_t)
        if s is None:
            s = score.reshape([-1, ])
        else:
            s = np.concatenate((s, score.reshape([-1, ])), 0)

        tor = np.float16(1.0 / math.sqrt(1.0 * embed))
        score = score * tor
        if is_mask:
            score = score + mask[:, :q_s, :kv_s]
        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((heads, q_s, 1))
        score_exp = np.exp(score.astype(np.float32))
        if not fp32:
            score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
            if _p is None:
                _p = score_exp.astype(np.float16).reshape([-1, ])
            else:
                _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
            p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
            out_sub = group_matmul(heads, group_num, p, v_slice)
        else:
            score_sum = np.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.astype(np.float16).reshape([-1, ])
            else:
                _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
            p = score_exp.astype(np.float16)
            out_sub = group_matmul(heads, group_num, p, v_slice)
            out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

        out_sub = out_sub.reshape(heads, q_s, embed_v)
        out_sub = np.transpose(out_sub, (1, 0, 2))
        out_sub = np.ascontiguousarray(out_sub)
        if out is None:
            out = out_sub
        else:
            out = np.concatenate((out, out_sub), 0)

        q_offset += q_s
        k_offset += kv_s
        v_offset += kv_s

    print("==> data generate finished!")

    q = q.astype(src_type).reshape(-1, heads, embed)
    k = k.astype(src_type).reshape(-1, group_num, embed)
    v = v.astype(src_type).reshape(-1, group_num, embed_v)
    # mask = mask.astype(src_type).reshape(max_s, max_s)
    mask = gen_swa_cmp(window_size, embed).astype(src_type)
    q_len = q_seqlen.astype(np.int32)
    out = out.astype(src_type).reshape(-1, heads, embed_v)
    ret_data = q, k, v, mask, q_len, out
    return ret_data

if operation_test.get_soc_version() == 'Ascend910B':
    kv_head = 2
    window_size = 32
    mask_type = 8
    data = calc_expect_func(2, 1024, 2, 128, window_size, mask_type, group_num=kv_head)
    param_seqlen = data[4].tolist()
    in_tensors = [torch.from_numpy(tensor) for tensor in data]
    in_tensors = [tensor.npu() for tensor in in_tensors]
    a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

    OP_NAME = "SelfAttentionOperation"
    PARAM = json.dumps({"headNum": kv_head, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head, \
        "maskType": mask_type, "calcType": 3, "windowSize": 32, "cacheType": 1})
    RUN_PARAM = json.dumps({"seqLen": param_seqlen})
    print(PARAM, RUN_PARAM)


class TestFlashAttentionEncoderOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[5]]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        # print(out_tensor.cpu())
    #     return torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), rtol=0.001, atol=0.001)
        out = out_tensor
        golden = golden_out_tensor
        ratios = [0.001, 0.001, 0.003, 0.003, 0.005, 0.005]
        embeddim = 128
        max_seq = 1024
        error_count = 0
        strict_error_count = 0
        alibi_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        out = out.flatten()
        golden = golden.flatten()
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        # max_diff = diff.max().item()
        # print("maxDiff: " , max_diff)
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)


        limit_error = torch.maximum(
            torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(
            torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(
            diff, strict_limit_error).sum().item()
        print("1/1000 Accuracy is: ",
                        1 - float(error_count) / out_len)
        print("3/1000 Accuracy is ",  1 -
                        float(strict_error_count) / out_len)
        print("accuracy is correct: ", (float(strict_error_count) / out_len) <= ratios[0])  
        # 新精度标准fp16 参考精度标准v0.3浮点计算单标杆
        # 计算次数 两个matmul + 一个softmax
        calc_times = embeddim * max_seq + 4
        # import pdb;pdb.set_trace()
        if calc_times < 2048:
            error = 2**(-8)
        else :
            error = 2**(-7)
        error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
        return (diff <= error_threshold).all()

        
    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])
    
    

if __name__ == '__main__':
    unittest.main()
