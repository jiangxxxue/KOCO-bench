#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import time
import json
from enum import Enum
import torch
import logging
import unittest
import math
import numpy as np
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../", "paged_attention"))
import operation_test  # NOQA: E402
sys.path.append("./tests/pythontest")
from precision_calcu import compare_cv
save_path = "./"


class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2


np.random.seed(123)
MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5
MASK_TYPE_SWA = 6
MASK_TYPE_SWA_DECODER = 7
MASK_TYPE_ALIBI_WITH_PREFIX_BATCH = 8
MASK_TYPE_NO_BATCH_WITH_PREFIX = 9
MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX = 10

CAL_TYPE_PREFIX_ENCODER = 4
MASK_TYPE_ALIBI_COMPRESS = 4
MASK_TYPE_CAUSAL_MASK = 9
MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND = 2012
KERNELTYPE_HIGH_PRECISION = 1

logging.getLogger().setLevel(logging.DEBUG)


class TestFlashAttentionPrefixEncoder(operation_test.OperationTest):

    def close_pack(self, in_data, seq_len):
        kv = in_data.numpy()
        dim1len = np.size(kv, -2)
        if max(seq_len) > dim1len:
            return None
        kv = kv.reshape(np.prod(kv.shape[0:-1]), kv.shape[-1])
        c_offset = 0
        s_offset = 0
        for i, len in enumerate(seq_len):
            kv[c_offset:c_offset + seq_len[i]
               ][:] = kv[s_offset:s_offset + seq_len[i]][:]
            c_offset += seq_len[i]
            s_offset += dim1len
        return torch.from_numpy(kv[0:sum(seq_len)][:])

    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0, cache_type=0,
                        is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4,
                        batch=1, kv_head=1, heads=1, embeddim=128, embeddimv=0, max_seq=2048,
                        kv_seqLen=[], is_clamp=0, clamp_min=0,
                        clamp_max=0, data_type=torch.float16, op_type=0, mask_type=0,
                        no_cache=False, long_seq=False, is_triu_mask=False, is_multi_layer=False,
                        is_sqrt=False, left_align=False, scaleType=ScaleType.SCALE_TOR.value, fav3=False,
                        tor=1, bnsd=False, is_compress=False, q_seqlens=None, num_blocks=None,
                        block_size=None):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.alibi_dim = alibi_dim
        self.batch = batch
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim
        self.embeddimv = embeddimv
        self.max_seq = max_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.online = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.cache_type = cache_type
        self.q_seqlens = q_seqlens if q_seqlens is not None else kv_seqLen

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [
                                                             1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(
                batch, self.q_seqlens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        if is_multi_layer:
            self.layer_id = torch.from_numpy(
                np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(
                np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(
            np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)

        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        else:
            # kv cache shape: (num_blocks, block_size, num_heads, head_size)
            self.k_cache = torch.from_numpy(
                np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(
                np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)

            batch = len(kv_seqLen)
            max_context_len = max(kv_seqLen)
            print("max_context_len: ", max_context_len)
            print("block_size: ", block_size)
            max_num_blocks_per_seq = (
                max_context_len + block_size - 1) // block_size
            block_tables = []   # (batch, max_num_blocks_per_seq)
            offset = 0
            for i in range(batch):
                num_blocks_cur_seq = (kv_seqLen[i] + block_size - 1) // block_size
                # padding block table with 0
                block_table = [
                    random.randint(0, num_blocks-1) if j < num_blocks_cur_seq else 0 for j in range(max_num_blocks_per_seq)
                ]
                offset += num_blocks_cur_seq
                block_tables.append(block_table)
            self.block_tables = torch.from_numpy(np.array(block_tables)).to(torch.int32)
            print("block_tables : ", block_tables)
            self.block_tables = torch.from_numpy(
                np.array(block_tables)).to(torch.int32)
            self.k = torch.stack([self.k_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.v = torch.stack([self.v_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.k = self.k.reshape(
                1, batch, max_context_len, kv_head * self.embeddim)
            self.v = self.v.reshape(
                1, batch, max_context_len, kv_head * self.embeddim)

        if self.fav3:
            self.is_int8_flag = True
            self.q_scale, self.q_offset, self.q_int8 = self.quant_per_head(
                self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, self.k_int8 = self.quant_per_head(
                self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, self.v_int8 = self.quant_per_head(
                self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) *
                            torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset = (self.k_offset.view(kv_head, 1) *
                             torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) *
                            torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset = (self.v_offset.view(kv_head, 1) *
                             torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = torch.from_numpy(np.random.uniform(
                1 / 127, 3 / 127, size=(heads))).to(torch.float32)

            self.q_int8 = torch.from_numpy(
                np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim))).to(torch.int8)
            self.k_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(torch.int8)
            self.v_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(torch.int8)

        self.gen_mask(batch, heads, data_type, mask_type,
                      window_size, is_compress, cache_type)
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")


    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, kv_s-q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH: ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD: ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_SWA: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_MASK: ((1, q_max_seq, kv_max_seq),
                                (lambda mask, idx, q_s, kv_s: 0))
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
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi:
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
        zero_indice = random.choices(range(self.max_seq), k=300)
        if self.window_size > 0:
            pass
        if self.is_alibi:
            pass
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

    def qkMM(self, query,key):
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split : qk_k_split + sub_k]
            key_k = key[:, qk_k_split : qk_k_split + sub_k, :]
            result_split = torch.matmul(query_k.to(torch.float32), key_k.to(torch.float32))
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def softmax(self,
        qk_result,
        is_first,
        gm
    ):
        sim = qk_result.numpy()
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm #全局的减去最新的
        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(np.float32))
        if qk_result.dtype == torch.float16:
            sim_sub = sim_sub.astype(np.float16)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        return torch.from_numpy(sim_sub), row_sum, dm, gm #返回P~ 、P~的rowsum和rowmax的差值 都是局部的结果

    def fa_prefill_causal_mask(self, batch_idx, query, key, value):
        if not torch.is_tensor(query):
            query = torch.from_numpy(query, dtype=self.data_type)
        if not torch.is_tensor(key):
            key = torch.from_numpy(key, dtype=self.data_type)
        if not torch.is_tensor(value):
            value = torch.from_numpy(value, dtype=self.data_type)
        group_num = query.shape[0] // key.shape[0]
        if group_num != 1: # GQA 场景
            key = key.repeat_interleave(group_num, dim=0)
            value = value.repeat_interleave(group_num, dim=0)
        gl = None
        gl_high = None
        go = None
        go_high = None
        block_size = 128
        kv_seqlen = key.shape[2]
        mask = self.mask_info[1](self.mask, batch_idx, query.shape[1], kv_seqlen)
        mask = mask * self.post_mask_coff
        if query.shape[0] // mask.shape[0] != 1:
            mask = mask.repeat_interleave(query.shape[0] // mask.shape[0], dim=0)
        for kv_start in range(0, kv_seqlen - 1, block_size):
            sub_len = block_size
            if kv_start + block_size > kv_seqlen:
                sub_len = kv_seqlen - kv_start
            sub_key = key[:, :, kv_start : kv_start + sub_len]
            sub_mask = mask[:, :, kv_start : kv_start + sub_len]
            sub_value = value[:, kv_start : kv_start + sub_len, :]
            qk_result = self.qkMM(query, sub_key) #低精度结果
            qk_result_high = self.qkMM(query.to(torch.float32), sub_key.to(torch.float32)) #高精度结果
            qk_result = qk_result.to(torch.float16) * self.tor
            qk_result_high = qk_result_high * self.tor
            if self.is_mask:
                qk_result += sub_mask
                qk_result_high += sub_mask.to(torch.float32)
            if kv_start == 0:
                gm = None
            p_result, row_sum, dm, gm = self.softmax(qk_result, kv_start == 0, gm)
            if kv_start == 0:
                gm_high = None
            p_result_high, row_sum_high, dm_high, gm_high = self.softmax(qk_result_high, kv_start == 0, gm_high)
            lo = torch.matmul(p_result.to(torch.float32), sub_value.to(torch.float32))
            lo = lo.to(torch.float16)
            lo_high = torch.matmul(p_result_high, sub_value.to(torch.float32))
            lo = lo.numpy()
            lo_high = lo_high.numpy()
            if kv_start == 0:
                gl = row_sum
                gl_high = row_sum_high
                go = lo
                go_high = lo_high
            else:  #更新
                dm = np.exp(dm)
                dm_high = np.exp(dm_high)
                gl = gl * dm
                gl = gl + row_sum

                go = go * dm
                go = go + lo

                gl_high = gl_high * dm_high
                gl_high = gl_high + row_sum_high

                go_high = go_high * dm_high
                go_high = go_high + lo_high
        go = go / gl
        go_high = go_high / gl_high
        go = np.transpose(go, (1, 0, 2))
        go_high = np.transpose(go_high, (1, 0, 2))
        go = go.reshape((go.shape[0], -1))
        go_high = go_high.reshape((go_high.shape[0], -1))
        return torch.from_numpy(go), torch.from_numpy(go_high)

    def gen_out_tensor(self, online=False):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                    if not self.fav3:
                        out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    if not self.fav3:
                        out_true = torch.cat((out_true, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            o_t, o_t_high = self.fa_prefill_causal_mask(idx, q_slice, k_slice_t, v_slice)
            if out is None:
                out = o_t
                if not self.fav3:
                    out_true = o_t_high
            else:
                out = torch.cat((out, o_t), 0)
                if not self.fav3:
                    out_true = torch.cat((out_true, o_t_high), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        # golden data


        out = out.view(q_ntokens, heads * embedv)
        self.golden_out = out.to(self.data_type)
        out_true = out_true.view(q_ntokens, heads * embedv)
        self.golden_out_true = out_true.to(torch.float32)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type,
                          self.mask_type, 0, False, 0)
        
    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def golden_calc(self, in_tensors):
        golden_out = self.golden_out.clone().detach().requires_grad_(True).half().npu()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        result_double = compare_cv(self.golden_out_true.flatten(), golden_tensors.cpu().flatten(), out_tensors.cpu().flatten())
        return result_double

    def test_flash_attention_case_fa_encoder_withcache_bf16_causal_mask(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # [b,ms,ms]
        batch = 2
        kv_head = 2        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8
        embeddim = 128
        max_seq = 128 * 100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*10+1] * batch
        kv_seqLen = [128*12+1] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp": is_clamp, "clampMin": clamp_min, "clampMax": clamp_max, "maskType": MASK_TYPE_CAUSAL_MASK, "kvHead": kv_head,
                    "isTriuMask": 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch=dynamic_batch,
                             is_decoder=isdecoder, batch=batch, kv_head=kv_head, heads=heads,
                             embeddim=embeddim, max_seq=max_seq, kv_seqLen=kv_seqLen,
                             is_clamp=is_clamp, clamp_max=clamp_max, clamp_min=clamp_min,
                             data_type=data_type,
                             op_type=OP_PARAM["type"], mask_type=MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache=True, tor=tor, q_seqlens=q_seqlens,
                             num_blocks=num_blocks, block_size=block_size, is_triu_mask=True, is_mask=True)
        self.gen_out_tensor()
        PARAM = json.dumps(
            {"headNum": heads, "calcType": CAL_TYPE_PREFIX_ENCODER, "maskType": MASK_TYPE_CAUSAL_MASK,
             "kvHeadNum": kv_head, "isTriuMask": 1, "qkScale": tor, "kernelType": KERNELTYPE_HIGH_PRECISION})
        RUN_PARAM = json.dumps({"seqLen": q_seqlens, "kvSeqLen": kv_seqLen, "CalcType": CAL_TYPE_PREFIX_ENCODER, "maskType": MASK_TYPE_CAUSAL_MASK})
        q_seqlen = np.array(q_seqlens)
        q_seqlen = torch.from_numpy(q_seqlen).to(torch.int32).npu()
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k_cache.npu(), self.v_cache.npu(),
                                self.block_tables.npu(), q_seqlen, kv_seqLen])


    def test_flash_attention_case_fa_encoder_withcache_fp16_causal_mask(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # [b,ms,ms]
        batch = 2
        kv_head = 2        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 4
        embeddim = 128
        max_seq = 128 * 100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [1025] * batch
        kv_seqLen = [2049] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp": is_clamp, "clampMin": clamp_min, "clampMax": clamp_max, "maskType": MASK_TYPE_NO_MASK, "kvHead": kv_head,
                    "isTriuMask": 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        data_type = torch.float16

        self.set_data_params(dynamic_batch=dynamic_batch,
                             is_decoder=isdecoder, batch=batch, kv_head=kv_head, heads=heads,
                             embeddim=embeddim, max_seq=max_seq, kv_seqLen=kv_seqLen,
                             is_clamp=is_clamp, clamp_max=clamp_max, clamp_min=clamp_min,
                             data_type=data_type,
                             op_type=OP_PARAM["type"], mask_type=MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache=True, tor=tor, q_seqlens=q_seqlens,
                             num_blocks=num_blocks, block_size=block_size, is_triu_mask=True, is_mask=True)
        self.gen_out_tensor()
        PARAM = json.dumps(
            {"headNum": heads, "calcType": CAL_TYPE_PREFIX_ENCODER, "maskType": MASK_TYPE_CAUSAL_MASK,
             "kvHeadNum": kv_head, "isTriuMask": 1, "qkScale": tor, "kernelType": KERNELTYPE_HIGH_PRECISION})
        RUN_PARAM = json.dumps({"seqLen": q_seqlens, "kvSeqLen": kv_seqLen, "CalcType": CAL_TYPE_PREFIX_ENCODER, "maskType": MASK_TYPE_CAUSAL_MASK})
        q_seqlen = np.array(q_seqlens)
        q_seqlen = torch.from_numpy(q_seqlen).to(torch.int32).npu()
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k_cache.npu(), self.v_cache.npu(),
                                self.block_tables.npu(), q_seqlen, kv_seqLen])

if __name__ == '__main__':
    unittest.main()
