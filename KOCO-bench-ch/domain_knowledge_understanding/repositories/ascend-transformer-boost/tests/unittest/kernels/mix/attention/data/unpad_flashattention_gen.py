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
import os
import numpy as np
import math
import logging
np.random.seed(0)


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

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


def group_matmul(head, kv_head, A, B):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(A[i*group_num: (i+1)*group_num, :, :].astype(np.float32),
                                B[i: (i+1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    logging.info(score.shape)
    return score


def gen_mask(self, batch, heads, data_type, mask_type):
    import random
    q_max_seq = self.max_seq
    kv_max_seq = self.max_seq
    mask_type_dict = {
        # 四维的alibi mask
        MASK_TYPE_ALIBI_WITH_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
        # 三维的alibi mask
        MASK_TYPE_ALIBI_NO_BATCH : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
        MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
        MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
        MASK_TYPE_NO_BATCH : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
        # 不加mask
        MASK_TYPE_NO_MASK : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: 0))
    }
    # kernel中mask的系数
    if data_type == torch.float16:
        post_mask_coff = 1
        pre_mask_coff = -10000.0
    elif data_type == torch.bfloat16 and self.is_alibi:
        post_mask_coff = 1
        pre_mask_coff = -float("inf")
    elif data_type == torch.float32 and self.is_alibi:
        post_mask_coff = 1
        pre_mask_coff = 1
    else:
        post_mask_coff = -3e38
        pre_mask_coff = 1
    if data_type == torch.float16:
        if self.is_alibi or self.long_seq:
            select_zero = False
        else:
            select_zero = True
    elif data_type == torch.bfloat16:
        if self.is_alibi:
            select_zero = False
        elif self.dynamic_batch or self.is_decoder:
            select_zero = True
        else:
            select_zero = False
    else:
        if self.is_alibi or self.is_decoder:
            select_zero = True
        else:
            select_zero = False
    if self.is_triu_mask:
        select_zero = False

    self.mask_info = mask_type_dict[mask_type]
    mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
    mask = np.triu(mask, 1)
    zero_indice = random.choices(range(self.max_seq), k = 300)
    if self.is_alibi:
        self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
        mask += self.alibi_bias.numpy()
    if select_zero:
        mask.flat[zero_indice] = 0
    self.mask = torch.from_numpy(mask).to(torch.float32)
    self.post_mask_coff = post_mask_coff
    self.pre_mask_coff = pre_mask_coff

def calc_expect_func(batch, seqlen, heads, kv_head, embed, max_seq,
                     is_mask=True, is_decoder=False, variate_seq=False, src_type='float16', dynamic_batch=1):
    if is_decoder:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, 1, variate_seq)
    else:
        q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, seqlen, variate_seq)
    kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, seqlen, variate_seq)

    ntokens2 = (q_seqlen * kv_seqlen).sum()
    layer_id = np.array([1])
    layer_id[0] = 0
    q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, size=(1, batch, max_seq, kv_head * embed)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, size=(1, batch, max_seq, kv_head * embed)).astype(np.float16)
    if dynamic_batch:
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        mask = np.ones(shape=(batch, max_seq, max_seq)).astype(np.float16)
        mask = np.triu(mask, 1)
        mask *= -10000.0
        zero_indices = np.random.choice(range(batch*max_seq*max_seq), 30000, replace=False)
        mask.flat[zero_indices] = 0
    else:
        batch_state = [1, 1, 1, 1, 1, 1, 1, 1]
        mask = np.ones(shape=(batch, max_seq, max_seq)).astype(np.float16)
        mask = np.triu(mask, 1)
        mask *= -10000.0


    q_heads = np.array([1])
    q_heads[0] = heads
    kv_heads = np.array([1])
    kv_heads[0] = kv_head
    is_batch_dynamic = np.array([1])
    is_batch_dynamic[0] = dynamic_batch
    batch_num = np.array([1])
    batch_num[0] = batch
    embd_dim = np.array([1])
    embd_dim[0] = embed
    max_seqlen = np.array([1])
    max_seqlen[0] = max_seq
    q_heads.astype(np.uint32).tofile("./q_heads.bin")
    kv_heads.astype(np.uint32).tofile("./kv_heads.bin")
    is_batch_dynamic.astype(np.uint32).tofile("./is_batch_dynamic.bin")
    batch_num.astype(np.uint32).tofile("./batch_num.bin")
    embd_dim.astype(np.uint32).tofile("./embd_dim.bin")
    max_seqlen.astype(np.uint32).tofile("./max_seqlen.bin")
    np.array(batch_state).astype(np.uint32).tofile("./batchRunStatus.bin")

    q_offset = 0
    k_offset = 0
    v_offset = 0
    #
    s = None
    _p = None
    out = None
    tor = np.float16(1.0 / math.sqrt(1.0 * embed))
    q = q * tor

    for idx in range(batch):
        q_s = q_seqlen[idx]
        kv_s = kv_seqlen[idx]
        q_slice = q[q_offset:q_offset + q_s][:]
        q_slice = q_slice.reshape(q_s, heads, embed)
        q_slice = np.transpose(q_slice, (1, 0, 2))
        k_slice = k[0][idx][:kv_s][:]
        k_slice = k_slice.reshape(kv_s, kv_head, embed)
        k_slice = np.transpose(k_slice, (1, 0, 2))
        k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T
        v_slice = v[0][idx][:kv_s][:]
        v_slice = v_slice.reshape(kv_s, kv_head, embed)
        v_slice = np.transpose(v_slice, (1, 0, 2))

        kfilename = f"kvBatches/input2_{idx}.bin"
        k[0][idx].astype(src_type).tofile(kfilename)
        vfilename = f"kvBatches/input3_{idx}.bin"
        v[0][idx].astype(src_type).tofile(vfilename)

        score = group_matmul(heads, kv_head, q_slice, k_slice_t)
        if s is None:
            s = score.reshape([-1,])
        else:
            s = np.concatenate((s, score.reshape([-1,])), 0)

        scale = np.float16((layer_id[0] + 1))
        score = score * scale

        if is_mask and dynamic_batch:
            score = score + mask[idx, :q_s, :kv_s]
        elif is_mask:
            score = score + mask[idx, :q_s, :kv_s]
        score_max = np.max(score, axis=-1)
        score = score - score_max.reshape((heads, q_s, 1))
        score_exp = np.exp(score.astype(np.float32))
        score_sum = np.sum(score_exp.astype(np.float16), axis=-1)

        if _p is None:
            _p = score_exp.astype(np.float16).reshape([-1,])
        else:
            _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1,])), 0)
        p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

        output = group_matmul(heads, kv_head, p, v_slice)
        output = output.reshape(heads, q_s, embed)
        output = np.transpose(output, (1, 0, 2))

        output = np.ascontiguousarray(output)
        if batch_state[idx] == 0:
            output = np.zeros([heads, q_s, embed])
            output = np.transpose(output, (1, 0, 2))
        if out is None:
            out = output
        else:
            out = np.concatenate((out, output), 0)

        q_offset += q_s
        k_offset += max_seq
        v_offset += max_seq

    q.astype(src_type).tofile("input1.bin")
    k.astype(src_type).tofile("input2.bin")
    v.astype(src_type).tofile("input3.bin")
    mask.astype(src_type).tofile("input4.bin")
    # s.astype(src_type).tofile("s.bin")
    # _p.astype(src_type).tofile("p.bin")
    out.astype(src_type).tofile("expect.bin")
    q_seqlen.astype(np.uint32).tofile("q_seqlen.bin")
    # q_ntokens.astype(np.uint32).tofile("q_ntokens.bin")
    kv_seqlen.astype(np.uint32).tofile("kv_seqlen.bin")
    # kv_ntokens.astype(np.uint32).tofile("kv_ntokens.bin")
    # ntokens2.astype(np.uint32).tofile("ntokens2.bin")
    layer_id.astype(np.uint32).tofile("layerId.bin")


if __name__ == "__main__":
    # batch_case = int(sys.argv[1]) # 1
    # batch_case = 8
    # # qkv_seq = int(sys.argv[2]) # 2
    # qkv_seq = 114
    # # kvhead = int(sys.argv[3]) # kv_head num
    # kvhead = 32
    # isdecoder = 1 # prefill or decoder
    # heads = 32 # llama7b  hidden_size 4096
    # embedim = 128
    # max_seq = 2048

    batch_case = int(sys.argv[1])
    qkv_seq = int(sys.argv[2])
    kvhead = int(sys.argv[3])
    isdecoder = int(sys.argv[4])
    heads = int(sys.argv[5])
    embedim = int(sys.argv[6])
    max_seq = int(sys.argv[7])

    variate_seq = int(sys.argv[8])
    dynamic_batch = int(sys.argv[9])
    kv_batches_folder = "./kvBatches"
    delete_files_in_folder(kv_batches_folder)
    if isdecoder:
        calc_expect_func(batch_case, qkv_seq, heads, kvhead, embedim, max_seq, is_mask=True, is_decoder=isdecoder, variate_seq=variate_seq, src_type='float16', dynamic_batch=dynamic_batch)
    else:
        calc_expect_func(batch_case, qkv_seq, heads, kvhead, embedim, max_seq, dynamic_batch=0)
