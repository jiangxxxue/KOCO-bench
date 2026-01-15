# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import sys
import math
from collections import namedtuple
import numpy as np
import logging

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)


def gen_seq_len(batch, max_seq, variate_seq=False):
    if variate_seq:
        num = max_seq // 16
        seqlen_aligned_arange = np.arange(1, min(batch + 1, num + 1)) * 16
        if batch > num:
            seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
            seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
            seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
        else:
            seqlen_aligned = seqlen_aligned_arange
        sp_list = np.random.randint(0, 15, size=(batch))
        seqlen = seqlen_aligned - sp_list
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
        sp_list = sp_list.astype(np.int32)
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)

    ntokens = seqlen.sum()
    return seqlen, seqlen_aligned, ntokens


def group_matmul(heads, kvheads, mat_a, mat_b):
    group_heads = heads // kvheads
    score = None
    for i in range(kvheads):
        group_score = np.matmul(mat_a[i * group_heads: (i + 1) * group_heads].astype(np.float32), \
                                mat_b[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score


def calc_expect_func(input_args, batch_mask=False):
    layer_id = 1
    layer = 2
    src_type = 'float16'
    is_mask = True
    if input_args.mode == "decoder":
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(input_args.batch, 1, input_args.variate_seq)
        is_mask = False
    if input_args.mode == "prefill":
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(input_args.batch, input_args.qseqlen, input_args.variate_seq)
    kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(input_args.batch, input_args.kvseqlen, input_args.variate_seq)
    q_ntokens = q_seqlen_aligned.sum()
    kv_ntokens = kv_seqlen_aligned.sum()
    if (kv_ntokens > input_args.max_seq):
        return
    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, input_args.heads * input_args.embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(input_args.batch * input_args.kvseqlen, input_args.kvheads * input_args.embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(input_args.batch * input_args.kvseqlen, input_args.kvheads * input_args.embed)).astype(np.float16)

    k_max = np.zeros(shape=(layer, input_args.batch*input_args.max_seq, input_args.kvheads * input_args.embed)).astype(np.float16)
    v_max = np.zeros(shape=(layer, input_args.batch*input_args.max_seq, input_args.kvheads * input_args.embed)).astype(np.float16)
    k_max[layer_id][:input_args.batch * input_args.kvseqlen] = k
    v_max[layer_id][:input_args.batch * input_args.kvseqlen] = v

    if is_mask and batch_mask:
        mask = np.ones(shape=(input_args.batch, input_args.max_seq, input_args.max_seq)).astype(np.float16)
        mask = np.triu(mask, 1)
        mask *= -10000.0
        zero_indices = np.random.choice(range(input_args.batch*input_args.max_seq*input_args.max_seq), 30000, replace=False)
        mask.flat[zero_indices] = 0
    elif is_mask:
        mask = np.ones(shape=(1, input_args.max_seq, input_args.max_seq)).astype(np.float16)
        mask = np.triu(mask, 1)
        mask *= -10000.0
    else:
        mask = np.zeros(shape=(input_args.max_seq, input_args.max_seq))

    q_offset = 0
    k_offset = 0
    v_offset = 0

    input1_nd = None
    input2 = None
    input3 = None
    out = None

    for idx in range(input_args.batch):
        s_align = q_seqlen_aligned[idx]
        kv_align = kv_seqlen_aligned[idx]
        s = q_seqlen[idx]
        kv = kv_seqlen[idx]

        q_slice = q[q_offset:q_offset + s][:]
        q_slice = q_slice.reshape(s, input_args.heads, input_args.embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))  # (head, token, emd)

        # 计算
        k_slice = k_max[layer_id][k_offset:k_offset + kv][:]
        k_slice = k_slice.reshape(kv, input_args.kvheads, input_args.embed)
        k_slice = np.transpose(k_slice, (1, 0, 2))   # heads*max_seqlen*embed

        # 输入
        k_slice_max = k_max[layer_id][k_offset:k_offset + input_args.max_seq][:]
        k_slice_max = k_slice_max.reshape(input_args.max_seq, input_args.kvheads, input_args.embed)
        k_slice_max = np.transpose(k_slice_max, (1, 0, 2))   # heads*max_seqlen*embed
        k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T

        # 计算
        v_slice = v_max[layer_id][v_offset:v_offset + kv][:]
        v_slice = v_slice.reshape(kv, input_args.kvheads, input_args.embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))

        # 输入
        v_slice_max = v_max[layer_id][v_offset:v_offset + input_args.max_seq][:]
        v_slice_max = v_slice_max.reshape(input_args.max_seq, input_args.kvheads, input_args.embed)
        v_slice_max = np.transpose(v_slice_max, (1, 0, 2))

        score = group_matmul(input_args.heads, input_args.kvheads, q_slice, k_slice_t)
        tor = np.float16(math.sqrt(1.0 * input_args.embed))
        score = score / tor
        if is_mask and batch_mask:
            score = score + mask[idx, :s, :kv]
        elif is_mask:
            score = score + mask[:, :s, :kv]

        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((input_args.heads, s, 1))
        score_exp = np.exp(score.astype(np.float32))
        score_sum = np.sum(score_exp, axis=-1)
        p = score_exp / score_sum.reshape((input_args.heads, s, 1))
        p = p.astype(np.float16)
        o_mat = group_matmul(input_args.heads, input_args.kvheads, p.astype(np.float32),
                      v_slice.astype(np.float32)).astype(np.float16)
        o_mat = o_mat.reshape(input_args.heads, s, input_args.embed)
        q_pad = np.zeros((input_args.heads, s_align, input_args.embed))
        o_pad = np.zeros((input_args.heads, s_align, input_args.embed))
        logging.info(q_slice.shape)
        q_pad[:, :s, :] = q_slice
        o_pad[:, :s, :] = o_mat

        if input1_nd is None:
            input1_nd = q_pad
        else:
            input1_nd = np.concatenate((input1_nd, q_pad), 1)

        input2_slice = convert_nd_to_nz(k_slice_max)

        if input2 is None:
            input2 = input2_slice.reshape([-1, 16, 16])
        else:
            input2 = np.concatenate((input2, input2_slice.reshape([-1, 16, 16])), 0)

        input3_slice = convert_nd_to_nz(v_slice_max)
        if input3 is None:
            input3 = input3_slice.reshape([-1, 16, 16])
        else:
            input3 = np.concatenate((input3, input3_slice.reshape([-1, 16, 16])), 0)

        if out is None:
            out = o_pad
        else:
            out = np.concatenate((out, o_pad), 1)

        q_offset += s
        k_offset += input_args.max_seq
        v_offset += input_args.max_seq

    layerid = np.array([layer_id], dtype=np.int32)
    layerid.tofile("./layer_id.bin")

    input1_nz = convert_nd_to_nz(input1_nd)
    input1_nz.astype(src_type).tofile("./input1.bin")

    input2_shape0, input2_shape1, input2_shape2 = input2.shape
    input2_all = np.zeros(shape=(layer, input2_shape0, input2_shape1, input2_shape2)).astype(np.float16)
    input2_all[layer_id] = input2
    input2_all.astype(src_type).tofile("./input2.bin")

    input3_shape0, input3_shape1, input3_shape2 = input3.shape
    input3_all = np.zeros(shape=(layer, input3_shape0, input3_shape1, input3_shape2)).astype(np.float16)
    input3_all[layer_id] = input3
    input3_all.astype(src_type).tofile("./input3.bin")

    out_nz = convert_nd_to_nz(out)
    out_nz.astype(src_type).tofile("./expect.bin")

    q_seqlen.astype(np.int32).tofile("./qseqlen.bin")
    kv_seqlen.astype(np.int32).tofile("./kvseqlen.bin")
    q_ntokens.astype(np.int32).tofile("./q_ntokens.bin")

    mask = convert_nd_to_nz(mask)
    mask.astype(src_type).tofile("./mask.bin")


if __name__ == "__main__":
    batchNum = int(sys.argv[1])
    maxSeqLen = int(sys.argv[2])
    runMode = sys.argv[3]  # prefill, decoder   prefill: qSeq=kvSeq   decoder: qSeq=1,  kvSeq=kvSeq
    variateSeq = int(sys.argv[4])  # 0不变长  1变长
    qSeqLen = int(sys.argv[5])
    kvSeqLen = int(sys.argv[6])
    headsize = 32 # llama7b 32 ; llama33b  52
    kvHeads = 2
    embedSize = 128
    Inputs = namedtuple("Inputs", ["batch", "max_seq", "heads", "kvheads", "embed", "qseqlen", "kvseqlen", "mode", "variate_seq"])
    inputArgs = Inputs(batchNum, maxSeqLen, headsize, kvHeads, embedSize, qSeqLen, kvSeqLen, runMode, variateSeq)
    calc_expect_func(inputArgs, True)