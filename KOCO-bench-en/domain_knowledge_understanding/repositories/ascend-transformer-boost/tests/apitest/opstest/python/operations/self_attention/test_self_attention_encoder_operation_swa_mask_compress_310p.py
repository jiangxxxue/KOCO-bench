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
        seqlen_aligned_arange = np.arange(1, min(batch + 1, num + 1)) * 16
        if batch > num:
            seqlen_aligned_remain = np.random.randint(
                1, max_seq, size=(batch - num))
            seqlen_aligned_remain[:] = (
                (seqlen_aligned_remain[:] + 15) // 16) * 16
            seqlen_aligned = np.concatenate(
                (seqlen_aligned_arange, seqlen_aligned_remain), 0)
        else:
            seqlen_aligned = seqlen_aligned_arange
        sp_list = np.random.randint(0, 15, size=(batch))
        seqlen = seqlen_aligned - sp_list
        print(seqlen)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
        sp_list = sp_list.astype(np.int32)
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)
        print(seqlen)
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
    return score


def round_up(x, align):
    if align == 0:
        return -1
    return (x + align - 1) // align * align


def custom_pad(x, pad_dims):
    return torch.nn.functional.pad(x, pad_dims)


def custom_reshape(x, target_shape):
    return x.reshape(target_shape)


def custom_transpose(x, dim1, dim2):
    return x.transpose(dim1, dim2)

# ND2NZ 2D
def nd_to_nz_2d(in_tensor):
    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = 1
    aux_dims[1] = round_up(in_tensor.size(0), 16)

    pad_dims = [0, 0, 0, 0]
    pad_dims[3] = round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    aux_dims[2] = round_up(in_tensor.size(1), 16) // 16
    aux_dims[3] = 16
    pad_dims[1] = round_up(in_tensor.size(1), 16) - in_tensor.size(1)

    return custom_transpose(custom_reshape(custom_pad(in_tensor, pad_dims), aux_dims),
                            1, 2
                            ).contiguous()

# NZ2ND above 2D
def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(
        len(x.shape) - 2, [2, 0, 1, 3])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).permute(*array_trans)


def calc_expect_func(layer, batch, seqlen, max_seq, heads, kv_head, embed, window_size=0):
    is_mask = True
    variate_seq = False
    is_decoder = False
    src_type = 'float16'
    fp32 = True
    print(f"batch: {batch}, seqlen: {seqlen}, max_seq: {max_seq}, heads: {heads}, kv_head: {kv_head}, embed: {embed}, window_size: {window_size}")
    print("q_seq is:")
    if is_decoder:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(
            batch, 1, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(
            batch, seqlen, variate_seq)
    else:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(
            batch, seqlen, variate_seq)
        # crossattention时，q_seqlen != k_seqlen
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens

    max_s = np.max(q_seqlen)

    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens,
                          heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                          kv_head * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                          kv_head * embed)).astype(np.float16)
    
    layer_id = np.array([layer - 1]).astype(np.int32)
    # tokenOffset = np.random.randint(max_s, max_seq, (batch,)).astype(np.int32)
    tokenOffset = q_seqlen
    k_max = np.zeros((layer, batch * max_seq, kv_head * embed)).astype(np.float16)
    v_max = np.zeros((layer, batch * max_seq, kv_head * embed)).astype(np.float16)
    
    mask = np.ones(shape=(1, max_seq, max_seq)).astype(
        np.float16)  # 使用当前最大seqlen生成mask
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
    k_cache = None
    v_cache = None

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]

        q_slice = q[q_offset:q_offset + q_s][:]
        q_slice = q_slice.reshape(q_s, heads, embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
        k_slice = k[k_offset:k_offset + kv_s][:]
        k_slice = k_slice.reshape(kv_s, kv_head, embed)
        k_slice = np.transpose(k_slice, (1, 0, 2))
        # get K^T (kv_heads, embed, k_seq)
        k_slice_t = np.transpose(k_slice, (0, 2, 1))
        v_slice = v[v_offset:v_offset + kv_s][:]
        v_slice = v_slice.reshape(kv_s, kv_head, embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))
        score = group_matmul(heads, kv_head, q_slice, k_slice_t)
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
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
            p = score_exp.astype(
                np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
            out_sub = group_matmul(heads, kv_head, p, v_slice)
        else:
            score_sum = np.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.astype(np.float16).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
            p = score_exp.astype(np.float16)
            out_sub = group_matmul(heads, kv_head, p, v_slice)
            out_sub = out_sub / \
                score_sum.reshape((heads, q_s, 1)).astype(np.float16)

        out_sub = out_sub.reshape(heads, q_s, embed)
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

    # q = q.astype(src_type).reshape(-1, heads, 128)
    # k = k.astype(src_type).reshape(-1, kv_head, 128)
    # v = v.astype(src_type).reshape(-1, kv_head, 128)
    mask = mask.astype(src_type).reshape(max_seq, max_seq)

    q_len = q_seqlen.astype(np.int32)
    out = out.astype(src_type).reshape(1024,-1)
    # out = out.astype(src_type).reshape(-1, heads, 128)

    ret_data = q, k, v, k_max, v_max, mask, q_len, tokenOffset, layer_id, out
    return ret_data
if operation_test.get_soc_version() == 'Ascend310P':
    layer = 2
    batch = 1
    seqlen = 1024
    max_seq = 2048
    heads = 32
    kv_head = 32
    window_size = 400
    embed = 128 
    data = calc_expect_func(layer, batch, seqlen, max_seq, heads, kv_head, embed, window_size=window_size)
    # mask = np.ones(shape=( 512, 512)).astype(
    #         np.float16)  # 使用当前最大seqlen生成mask
    # mask_u = np.triu(mask, 1)
    # mask_l = np.tril(mask, -window_size)
    # mask = mask_u + mask_l
    # mask *= -10000.0
    # mask = torch.from_numpy(mask).npu()
    param_seqlen = data[6].tolist()
    param_tokenOffset = data[7].tolist()
    in_tensors = [torch.from_numpy(tensor) for tensor in data]
    in_tensors = [tensor.npu() for tensor in in_tensors]
    k_cache_nz = convert_nd_to_nz(in_tensors[3]).reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
    k_cache_nz = torch_npu.npu_format_cast(k_cache_nz.contiguous(), 29)
    in_tensors[3] = k_cache_nz
    v_cache_nz = convert_nd_to_nz(in_tensors[4]).reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
    v_cache_nz = torch_npu.npu_format_cast(v_cache_nz.contiguous(), 29)
    in_tensors[4] = v_cache_nz


    swa_block_size = (16384 // embed // 16 * 16) if embed > 128 else 128
    if 0 < window_size <= 3 * swa_block_size:
        mask = torch.ones((512, 512), dtype=torch.float16)
        mask_triu = torch.triu(mask, diagonal=1)
        mask_trid = torch.tril(mask, diagonal=(-window_size))
        mask = mask_triu + mask_trid
        mask *= -10000.0
    else:
        mask = torch.ones((512, 512), dtype=torch.float16)
        mask_triu = torch.triu(mask, diagonal=1)
        mask_trid = torch.tril(mask, diagonal=(-(window_size % swa_block_size + 2 * swa_block_size)))
        mask = mask_triu + mask_trid
        mask *= -10000.0
    # print(mask)
    mask_nz = nd_to_nz_2d(mask)
    mask_nz = mask_nz.npu()
    mask_nz = torch_npu.npu_format_cast(mask_nz, 29)

    in_tensors[5] = mask_nz
    a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

    OP_NAME = "SelfAttentionOperation"
    PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                        "calcType": 1, "maskType": 8, "windowSize": window_size})
    RUN_PARAM = json.dumps({"seqLen": param_seqlen, "tokenOffset": param_tokenOffset})
    print(PARAM, RUN_PARAM)


class TestFlashAttentionEncoderOperation910A(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[9]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        # print('out: ', out_tensor)
        # print('golden: ', golden_out_tensor)
        # return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)
    
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
        max_diff = diff.max().item()
        print("maxDiff: " , max_diff)
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
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], in_tensors[8]
        ])
    

if __name__ == '__main__':
    unittest.main()
