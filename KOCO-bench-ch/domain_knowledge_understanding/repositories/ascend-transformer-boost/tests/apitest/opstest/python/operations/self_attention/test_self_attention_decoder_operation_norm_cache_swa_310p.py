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


def calc_expect_func(layer, batch, seqlen, max_seq, heads, kv_head, embed):
    variate_seq = False
    is_decoder = True
    src_type = 'float16'
    fp32 = True
    print(f"batch: {batch}, seqlen: {seqlen}, max_seq: {max_seq}, heads: {heads}, kv_head: {kv_head}, embed: {embed}")
    print("q_seq is:")
    if is_decoder:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(
            batch, 1, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(
            batch, seqlen, variate_seq)

    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens,
                          heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                          kv_head * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                          kv_head * embed)).astype(np.float16)
    
    layer_id = np.array([layer - 1]).astype(np.int32)
    k_new = np.zeros((batch, 1, kv_head * embed)).astype(np.float16)
    v_new = np.zeros((batch, 1, kv_head * embed)).astype(np.float16)
    k_max = np.zeros((layer, batch, max_seq, kv_head * embed)).astype(np.float16)
    v_max = np.zeros((layer, batch, max_seq, kv_head * embed)).astype(np.float16)

    q_offset = 0
    k_offset = 0
    v_offset = 0

    out = None

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]
        k_max[layer_id[0]][idx][:kv_s-1] = k[k_offset:k_offset + kv_s - 1][:]
        k_new[idx][:] = k[k_offset + kv_s - 1:k_offset + kv_s][:]
        v_max[layer_id[0]][idx][:kv_s-1] = v[k_offset:k_offset + kv_s - 1][:]
        v_new[idx][:] = v[k_offset + kv_s - 1:k_offset + kv_s][:]

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
        tor = np.float16(1.0 / math.sqrt(1.0 * embed))
        score = score * tor
        if (window_size < kv_s):
            window_mask = np.zeros((q_s, max_seq)).astype(np.float16)
            window_mask[:q_s, :(kv_s - window_size)] = -10000.0
            score += window_mask[:q_s, :kv_s]
        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((heads, q_s, 1))
        score_exp = np.exp(score.astype(np.float32))
        if not fp32:
            score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
            p = score_exp.astype(
                np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
            out_sub = group_matmul(heads, kv_head, p, v_slice)
        else:
            score_sum = np.sum(score_exp, axis=-1)
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

    q = q.astype(src_type).reshape(-1, heads * embed)
    k_new = k_new.astype(src_type).reshape(-1, kv_head * embed)
    v_new = v_new.astype(src_type).reshape(-1, kv_head * embed)
    k_max = k_max.astype(src_type)
    v_max = v_max.astype(src_type)

    q_len = q_seqlen.astype(np.int32)
    kv_len = kv_seqlen.astype(np.int32)
    out = out.astype(src_type).reshape(-1, heads * embed)

    ret_data = q, k_new, v_new, k_max, v_max, q_len, kv_len, layer_id, out
    return ret_data
if operation_test.get_soc_version() == 'Ascend310P':
    layer = 2
    batch = 1
    seqlen = 8192
    max_seq = 10240
    heads = 40
    kv_head = 40
    embed = 128
    window_size = 2048
    data = calc_expect_func(layer, batch, seqlen, max_seq, heads, kv_head, embed)
    param_seqlen = data[5].tolist()
    param_tokenOffset = data[6].tolist()
    in_tensors = [torch.from_numpy(tensor) for tensor in data]
    in_tensors = [tensor.npu() for tensor in in_tensors]
    k_cache_nz = convert_nd_to_nz(in_tensors[3]).reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
    k_cache_nz = torch_npu.npu_format_cast(k_cache_nz.contiguous().npu(),29)
    in_tensors[3] = k_cache_nz
    v_cache_nz = convert_nd_to_nz(in_tensors[4]).reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
    v_cache_nz = torch_npu.npu_format_cast(v_cache_nz.contiguous().npu(),29)
    in_tensors[4] = v_cache_nz
    a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

    OP_NAME = "SelfAttentionOperation"
    PARAM = json.dumps({"headNum": heads, "qkScale": (1 / float(math.sqrt(embed))), "kvHeadNum": kv_head,
                        "calcType": 2, "maskType": 7, "cacheType": 0, "windowSize": window_size})
    RUN_PARAM = json.dumps({"tokenOffset": param_tokenOffset, "seqLen": param_seqlen})
    print(PARAM, RUN_PARAM)


class TestFlashAttentionEncoderOperation910A(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[8]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
                in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[6], in_tensors[5], in_tensors[7]
            ])


if __name__ == '__main__':
    unittest.main()
