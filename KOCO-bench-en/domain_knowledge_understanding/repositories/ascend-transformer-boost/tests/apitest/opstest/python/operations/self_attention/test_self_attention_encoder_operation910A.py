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
            seqlen_aligned_remain = np.random.randint(
                1, max_seq, size=(batch - num))
            seqlen_aligned_remain[:] = (
                (seqlen_aligned_remain[:] + 15) // 16) * 16
            seqlen_aligned = np.concatenate(
                (seqlen_aligned_arange, seqlen_aligned_remain), 0)
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


class TestFlashAttentionEncoderOperation910A(operation_test.OperationTest):
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
    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes
    def get_alibi_bias(self, n_heads, max_seqlen):
        self.bias = torch.arange(max_seqlen)
        self.bias = self.bias[None, :] - self.bias[:, None]
        # if (self.is_sqrt):
        #     self.bias = torch.sqrt(torch.abs(self.bias)) * torch.sign(self.bias)
        bias = torch.empty(
            n_heads,
            max_seqlen,
            max_seqlen
        )[:, :max_seqlen, :max_seqlen].copy_(self.bias)
        self.alibi_slopes = self.get_alibi_slopes(n_heads)
        bias = bias * self.alibi_slopes[:, None, None]
        return bias
    def calc_expect_func(self, batch, seqlen, heads, embed, group_num=32, is_mask = False, is_alibi=False,
                         is_mask_compress=False):
        variate_seq = False
        is_decoder = False
        max_seq = 2048
        src_type = 'float16'
        fp32 = True
        print(f"group_num: {group_num}")
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
        ntokens2 = (q_seqlen * kv_seqlen).sum()

        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens,
                            heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                            group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens,
                            group_num * embed)).astype(np.float16)
        if is_alibi:
            # bsz_heads = batch * heads # if mask_dim == 4 else heads
            bsz_heads = heads
            mask = np.ones(shape=(bsz_heads, max_seq,
                            max_seq)).astype(np.float16)
            mask = np.triu(mask, 1)
            mask *= -10000
            self.alibi_bias = self.get_alibi_bias(heads, max_seq)
 
 
            # mask = mask.reshape(
            #     (batch, heads, max_seq, max_seq)) # if mask_dim == 4 else mask
            mask += self.alibi_bias.numpy()
        else:
            mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
            mask = np.triu(mask, 1)
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
            # get K^T (kv_heads, embed, k_seq)
            k_slice_t = np.transpose(k_slice, (0, 2, 1))
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed)
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
                    _p = np.concatenate(
                        (_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(
                    np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate(
                        (_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = group_matmul(heads, group_num, p, v_slice)
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

        q = q.astype(src_type).reshape(-1, heads, embed)
        k = k.astype(src_type).reshape(-1, group_num, embed)
        v = v.astype(src_type).reshape(-1, group_num, embed)
        
        if is_alibi:
            # mask = mask[0, :, :, :]
            mask = self.convert_nd_to_nz(mask).astype(np.float16)
            mask = mask.reshape(heads, max_seq // 16, max_seq, 16)
        elif is_mask_compress:
            mask = np.ones(shape=(128, 128)).astype(np.float16)  # 使用当前最大seqlen生成mask
            mask = np.triu(mask, 1)
            mask *= -10000.0
        else:
            mask = mask.astype(src_type).reshape(max_s, max_s)

        q_len = q_seqlen.astype(np.int32)
        out = out.astype(src_type).reshape(-1, heads, embed)

        ret_data = q, k, v, mask, q_len, out
        return ret_data

    def golden_calc(self, input_tensors):
        return [self.golden_out]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test_no_mask(self):
        if operation_test.get_soc_version() == 'Ascend910B' or \
           operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase only supports Ascend310P\Ascend910A")
            return
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head)
        param_seqlen = data[4].tolist()
        in_tensors = [torch.from_numpy(tensor) for tensor in data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        self.golden_out = in_tensors[5]
        mask_nz = nd_to_nz_2d(in_tensors[3])
        mask_nz = torch_npu.npu_format_cast(mask_nz, 29)
        in_tensors[3] = mask_nz
        a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                            "calcType": 3,})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(PARAM, RUN_PARAM)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[4]
        ])
        
    def test_norm_mask(self):
        if operation_test.get_soc_version() == 'Ascend910B' or \
           operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase only supports Ascend310P\Ascend910A")
            return
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head, is_mask=True)
        param_seqlen = data[4].tolist()
        in_tensors = [torch.from_numpy(tensor) for tensor in data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        self.golden_out = in_tensors[5]
        mask_nz = nd_to_nz_2d(in_tensors[3])
        mask_nz = torch_npu.npu_format_cast(mask_nz, 29)
        in_tensors[3] = mask_nz
        a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                            "calcType": 3,"maskType":1})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(PARAM, RUN_PARAM)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])
        
    def test_nd_mask(self):
        if operation_test.get_soc_version() == 'Ascend910B' or \
           operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase only supports Ascend310P\Ascend910A")
            return
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head, is_mask=True)
        param_seqlen = data[4].tolist()
        in_tensors = [torch.from_numpy(tensor) for tensor in data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        self.golden_out = in_tensors[5]
        # mask_nz = nd_to_nz_2d(in_tensors[3])
        # mask_nz = torch_npu.npu_format_cast(mask_nz, 29)
        # in_tensors[3] = mask_nz
        in_tensors[3] = in_tensors[3].unsqueeze(0)
        a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                            "calcType": 3,"maskType":1})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(PARAM, RUN_PARAM)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])
        
    def test_alibi_mask(self):
        if operation_test.get_soc_version() == 'Ascend910B' or \
           operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase only supports Ascend310P\Ascend910A")
            return
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head, is_mask=True, is_alibi=True)
        param_seqlen = data[4].tolist()
        in_tensors = [torch.from_numpy(tensor) for tensor in data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        self.golden_out = in_tensors[5]
        # mask_nz = nd_to_nz_2d(in_tensors[3])
        mask_nz = torch_npu.npu_format_cast(in_tensors[3].contiguous(), 29)
        in_tensors[3] = mask_nz
        a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                            "calcType": 3, "maskType":2})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(PARAM, RUN_PARAM)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])
        
    def test_norm_compress_mask(self):
        if operation_test.get_soc_version() != 'Ascend910P':
            print("this testcase only supports Ascend310P")
            return
        kv_head = 32
        data = self.calc_expect_func(16, 256, 32, 128, group_num=kv_head, is_mask=True, is_mask_compress=True)
        param_seqlen = data[4].tolist()
        in_tensors = [torch.from_numpy(tensor) for tensor in data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        self.golden_out = in_tensors[5]
        mask_nz = nd_to_nz_2d(in_tensors[3])
        mask_nz = torch_npu.npu_format_cast(mask_nz, 29)
        in_tensors[3] = mask_nz
        a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head,
                            "calcType": 3,"maskType":1, "isTriuMask":1})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(PARAM, RUN_PARAM)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])

if __name__ == '__main__':
    unittest.main()