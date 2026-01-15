# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import math
import sys

import numpy as np
import logging

np.random.seed(0)


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
        sp_list = np.random.randint(0, 15, size=batch)
        seqlen_aligned = np.array([max_seq] * batch)
        seqlen = seqlen_aligned  # 固定长度时，使用seqlen = seqlen_aligned / seqlen_aligned - sp_list
        seqlen = seqlen[-batch:]
        seqlen_aligned = seqlen_aligned[-batch:]
        logging.info(seqlen)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
        sp_list = sp_list.astype(np.int32)
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        logging.info(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)

    ntokens = seqlen.sum()
    logging.info(f"ntokens:{ntokens}")
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
    logging.info(score.shape)
    return score


def calc_expect_func(batch, seqlen, heads, embed,
                     is_mask=True, variate_seq=False,
                     is_decoder=False, max_seq=2048, src_type='float16', fp32=False, group_num=32):
    logging.info(f"group_num: {group_num}")
    logging.info("q_seq is:")
    if is_decoder:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, 1, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, seqlen, variate_seq)
    else:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, seqlen, variate_seq)
        # crossattention时，q_seqlen != k_seqlen
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens

    max_s = np.max(q_seqlen)
    ntokens2 = (q_seqlen * kv_seqlen).sum()

    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
    mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
    mask = np.triu(mask, 1)
    mask *= -10000.0
    logging.info(f"max_s: {max_s}, mask: {mask}")

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
        v_slice = v_slice.reshape(kv_s, group_num, embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))
        # score = np.matmul(q_slice.astype(np.float32),
        #                   k_slice_t.astype(np.float32)).astype(np.float16)  # (heads, q_seq, k_seq)
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
            # out_sub = np.matmul(p.astype(np.float32),
            #               v_slice.astype(np.float32)).astype(np.float16)
            out_sub = group_matmul(heads, group_num, p, v_slice)
        else:
            score_sum = np.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.astype(np.float16).reshape([-1, ])
            else:
                _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
            p = score_exp.astype(np.float16)
            # out_sub = np.matmul(p.astype(np.float32),
            #               v_slice.astype(np.float32)).astype(np.float16)
            out_sub = group_matmul(heads, group_num, p, v_slice)
            out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

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

    q.astype(src_type).tofile("input1.bin")
    k.astype(src_type).tofile("input2.bin")
    v.astype(src_type).tofile("input3.bin")
    mask.astype(src_type).tofile("input4.bin")
    s.astype(src_type).tofile("s.bin")
    _p.astype(src_type).tofile("p.bin")
    out.astype(src_type).tofile("expect.bin")
    q_seqlen.astype(np.int32).tofile("q_seqlen.bin")
    q_ntokens.astype(np.int32).tofile("q_ntokens.bin")
    kv_seqlen.astype(np.int32).tofile("kv_seqlen.bin")
    kv_ntokens.astype(np.int32).tofile("kv_ntokens.bin")
    ntokens2.astype(np.int32).tofile("ntokens2.bin")
    logging.info("==> data generate finished!")


if __name__ == "__main__":
    batch_size = int(sys.argv[1])  # (2, 256)
    seq_len = int(sys.argv[2]) # (32, 128)
    decoder = 0
    kv_heads = 32
    maxseqlen = 2048
    head_nums = 32  # llama7b  hidden_size 4096
    embed_size = 128
    if decoder:
        logging.info(f"==> infer decoder: {decoder}")
        calc_expect_func(batch_size, seq_len, head_nums, embed_size, is_mask=False,
                         variate_seq=False, is_decoder=True, fp32=True, group_num=kv_heads, max_seq=maxseqlen)
    else:
        calc_expect_func(batch_size, seq_len, head_nums, embed_size,
                         fp32=True, group_num=kv_heads, max_seq=maxseqlen, variate_seq=True)
