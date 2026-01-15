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
import unittest
import json
import math
import torch
import torch_npu
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
import pdb

OP_NAME = "SelfAttentionOperation"

MASK_TYPE_NO_HEAD_DECODER = 5
torch.manual_seed(42)
class TestUnpadSelfAttentionOperation(operation_test.OperationTest):
    def test(self):
        if  not operation_test.get_soc_version() == 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        batch = 2
        self.data_type = torch.float16
        qkv_seq = 1024
        kv_head = 32        # kv_head num
        heads = 32
        self.heads =  heads         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 3
        layer = 2
        seqlen = qkv_seq
        embed = embeddim
        self.embed, self.max_seq = embed, max_seq
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256 = True, False, True, False, False, False, False
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, 1, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, seqlen, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(
            batch, seqlen, variate_seq)

        q_ntokens = q_seqlen_aligned.sum()
        kv_ntokens = kv_seqlen_aligned.sum()
        layer_id = np.array([layer-1], dtype=np.int32)
        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens,
                              heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(batch *
                              seqlen, kv_head * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(batch *
                              seqlen, kv_head * embed)).astype(np.float16)
        k_max = np.zeros(shape=(layer, batch * max_seq,
                         kv_head * embed)).astype(np.float16)
        v_max = np.zeros(shape=(layer, batch * max_seq,
                         kv_head * embed)).astype(np.float16)
        k_max[layer_id[0]][:batch * seqlen] = k
        v_max[layer_id[0]][:batch * seqlen] = v
        if is_mask:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = np.ones(shape=(bsz_heads, max_seq,max_seq))
                mask *= -60000
                mask = np.triu(mask, 1)
                mask = mask.astype(np.float16)
                self.alibi_bias = self.get_alibi_bias(heads, max_seq)
                mask = mask.reshape(
                    (batch, heads, max_seq, max_seq)) if mask_dim == 4 else mask
                mask += self.alibi_bias.numpy()
            elif mask_dim == 2:
                mask = np.ones(shape=(1, max_seq, max_seq)).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000.0
            else:
                mask = np.ones(shape=(batch, max_seq, max_seq)
                               ).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000.0
        elif is_decoder:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = np.ones(shape=(bsz_heads, 16, max_seq)
                               ).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000
                for i in range(bsz_heads):
                    mask[i] += np.random.uniform(-2, 2,
                                                 size=(16, max_seq)).astype(np.float16)
                mask = mask.reshape((batch, heads, 16, max_seq)
                                    ) if mask_dim == 4 else mask
            elif mask_dim == 2:
                mask = np.zeros(shape=(1, 16, max_seq)).astype(np.float16)
                mask[0, :1, :2] = -10000
            else:
                mask = np.zeros(shape=(batch, 16, max_seq)).astype(np.float16)
                for i in range(batch):
                    mask[i, :1, :i] = -10000
        # logn功能开关
        self.scaleType = 0
        if self.scaleType:
            m = 8192 # seq_length in qwen config
            base = int(1 * m - batch / 2) # 3 situations: base < m ; base > m ; base < m < base + batch
            self.logn_arr = np.array([
                    math.log(i, m) if i > m else 1
                    for i in range(base, base + batch)
                ]).astype(np.float32)
        else:
            self.logn_arr=[]
        logn = torch.tensor(self.logn_arr, dtype=torch.float16).npu()

        q_offset = 0
        k_offset = 0
        v_offset = 0
        input1_nd = None
        input2 = None
        input3 = None
        out = None

        for idx in range(batch):
            s_align = q_seqlen_aligned[idx]
            kv_align = kv_seqlen_aligned[idx]
            s = q_seqlen[idx]
            kv = kv_seqlen[idx]

            q_slice = q[q_offset:q_offset + s][:]
            q_slice = q_slice.reshape(s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (head, token, emd)

            # 计算
            k_slice = k_max[layer_id[0]][k_offset:k_offset + kv][:]
            k_slice = k_slice.reshape(kv, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice = np.transpose(k_slice, (1, 0, 2))

            # 输入
            k_slice_max = k_max[layer_id[0]][k_offset:k_offset + max_seq][:]
            k_slice_max = k_slice_max.reshape(max_seq, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice_max = np.transpose(k_slice_max, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T

            # 计算
            v_slice = v_max[layer_id[0]][v_offset:v_offset + kv][:]
            v_slice = v_slice.reshape(kv, kv_head, embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))

            # 输入
            v_slice_max = v_max[layer_id[0]][v_offset:v_offset + max_seq][:]
            v_slice_max = v_slice_max.reshape(max_seq, kv_head, embed)
            v_slice_max = np.transpose(v_slice_max, (1, 0, 2))

            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            tor = np.float16(math.sqrt(1.0 * embed))
            score = score / tor
            # logn缩放计算
            if self.scaleType:
                score = score * self.logn_arr[idx]
            if mask_dim != 0:
                if is_alibi:
                    if mask_dim == 4:
                        score += mask[idx][:, :s, :kv]
                    else:
                        score += mask[:, :s, :kv]
                elif mask_dim == 2:
                    score += mask[0][:s, :kv]
                else:
                    score = score + mask[idx, :s, :kv]

            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, s, 1))
            score_exp = np.exp(score.astype(np.float32))
            score_sum = np.sum(score_exp, axis=-1)
            p = score_exp / score_sum.reshape((heads, s, 1))
            p = p.astype(np.float16)
            o_mat = self.group_matmul(heads, kv_head, p.astype(np.float32),
                                      v_slice.astype(np.float32)).astype(np.float16)
            o_mat = o_mat.reshape(heads, s, embed)
            q_pad = np.zeros((heads, s_align, embed))
            o_pad = np.zeros((heads, s_align, embed))
            q_pad[:, :s, :] = q_slice
            o_pad[:, :s, :] = o_mat

            if input1_nd is None:
                input1_nd = q_pad
            else:
                input1_nd = np.concatenate((input1_nd, q_pad), 1)
            input2_slice = self.convert_nd_to_nz(k_slice_max)
            if input2 is None:
                input2 = input2_slice.reshape([-1, 16, 16])
            else:
                input2 = np.concatenate(
                    (input2, input2_slice.reshape([-1, 16, 16])), 0)
            input3_slice = self.convert_nd_to_nz(v_slice_max)
            if input3 is None:
                input3 = input3_slice.reshape([-1, 16, 16])
            else:
                input3 = np.concatenate(
                    (input3, input3_slice.reshape([-1, 16, 16])), 0)
            if out is None:
                out = o_pad
            else:
                out = np.concatenate((out, o_pad), 1)

            q_offset += s
            k_offset += max_seq
            v_offset += max_seq
        # input data
        q = self.convert_nd_to_nz(input1_nd)
        self.q = q.astype(np.float16).reshape(
            1, heads * embed // 16, q_ntokens, 16)
        #print("before---------------",self.q.shape)
        input2_shape0, input2_shape1, input2_shape2 = input2.shape
        k = np.zeros(shape=(layer, input2_shape0, input2_shape1,
                     input2_shape2)).astype(np.float16)
        k[layer_id] = input2
        self.k = k.reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
        input3_shape0, input3_shape1, input3_shape2 = input3.shape
        v = np.zeros(shape=(layer, input3_shape0, input3_shape1,
                     input3_shape2)).astype(np.float16)
        v[layer_id] = input3
        self.v = v.reshape(layer, batch, kv_head * embed // 16, max_seq, 16)
        if mask_dim == 4 and is_alibi_128:
            mask = mask[0, :, :, :128]
        mask = self.convert_nd_to_nz(mask).astype(np.float16)
        if is_mask:
            if is_alibi:
                if mask_dim == 4:
                    if is_alibi_128:
                        self.mask = mask.reshape(heads, 128 // 16, max_seq, 16)
                    else:
                        self.mask = mask.reshape(batch * heads, max_seq // 16, max_seq, 16)
                else:
                    self.mask = mask.reshape(heads, max_seq // 16, max_seq, 16)
            elif mask_dim == 2:
                self.mask = mask.reshape(1, max_seq // 16, max_seq, 16)
            else:
                self.mask = mask.reshape(batch, max_seq // 16, max_seq, 16)
        elif is_decoder:
            if is_alibi:
                if mask_dim == 4:
                    self.mask = mask.reshape(
                        batch * heads, max_seq // 16, 16, 16)
                else:
                    self.mask = mask.reshape(heads, max_seq // 16, 16, 16)
            elif mask_dim == 2:
                self.mask = mask.reshape(1, max_seq // 16, 16, 16)
            else:
                self.mask = mask.reshape(batch, max_seq // 16, 16, 16)
        if is_alibi_256:
            self.alibi_slopes *= -1
            mask = np.ones(shape=(256, 256)) * 60000
            mask = np.triu(mask, 1)
            mask = mask.astype(np.float16)
            if max_seq < 256:
                self.bias = torch.tensor(np.pad(self.bias, ((0, 256 - max_seq), (0, 256 - max_seq)), "constant"))
            mask = self.bias[:256, :256] * -1 + mask
            mask = mask.numpy().astype(np.float16)
            mask = self.convert_nd_to_nz(mask).astype(np.float16)
            self.mask = mask.reshape(1, 16, 256, 16)
        else:
            self.alibi_slopes = []
        self.layer_id = layer_id
        # golden data
        out_nz = self.convert_nd_to_nz(out).astype(
            np.float16).reshape(1, heads * embed // 16, q_ntokens, 16)
        self.golden_out = out_nz
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = kv_seqlen
        self.kv_ntokens = kv_ntokens
        self.golden_out = torch.from_numpy(out_nz).to(self.data_type)
        self.golden_out = self.golden_out.reshape(self.golden_out.shape[1], batch,self.golden_out.shape[2]//batch,self.golden_out.shape[3])
        self.golden_out = self.golden_out.permute(1, 0, 2, 3)
        self.golden_out = self.convert_nz_to_nd(self.golden_out)
        #self.golden_out = torch.from_numpy(out)
        self.q_scale = 1
        self.qk_scale = math.sqrt(1 / embeddim)
        #print("before---------------",self.q.shape)
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(torch.from_numpy(self.mask).contiguous().npu(),29)
        print(self.k.shape)
        print(self.v.shape)
        # Func Sel
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), 
                            "maskType": 1, "kvcacheCfg":1, "calcType":2, "kvHeadNum":32})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        # "scaleType" is used for correct binder count.If anyelse is added, please put it at the end of run_param.
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "scaleType":self.scaleType}) 
        #pdb.set_trace()
        if self.scaleType:
            self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), 
                     torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu(),logn])
        else:
            for i in range(1):
                self.execute_with_param(OP_NAME, param, run_param,
                        [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), 
                        torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
        
    def gen_seq_len(self, batch, max_seq, variate_seq=False):
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
    
    def shape_nd_to_nz(self, shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2]
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    def gen_axes_for_transpose(self, offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def convert_nd_to_nz(self, x):
        array_trans = self.gen_axes_for_transpose(
            len(x.shape) - 2, [2, 0, 1, 3])
        x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)

    def convert_nz_to_nd(self, x):
        aux_dims = [0, 0, 0]
        aux_dims[0] = x.size(0)
        aux_dims[1] = x.size(2)
        aux_dims[2] = x.size(1) * x.size(3)
        return x.transpose(1, 2).reshape(aux_dims)
    
    def group_matmul(self, heads, kvheads, mat_a, mat_b):
        group_heads = heads // kvheads
        score = None
        for i in range(kvheads):
            group_score = np.matmul(mat_a[i * group_heads: (i + 1) * group_heads].astype(np.float32),
                                    mat_b[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    def golden_calc(self, in_tensors):
        print(f"golden_out: dtype={self.golden_out.dtype}, shape={self.golden_out.shape}")
        return [self.golden_out]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor.cpu(), golden_out_tensor, rtol=0.001, atol=0.001)

if __name__ == '__main__':
    unittest.main()
