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
class TestUnpadSelfAttentionOperation(operation_test.OperationTest):
    def generate_data(self):
        if operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend310P")
            return
        batch = 32
        self.data_type = torch.float16
        qkv_seq = 128
        kv_head = 32        # kv_head num
        heads = 32
        self.heads =  heads         # llama7b  hidden_size 4096
        embeddim = 96
        max_seq = 2048
        mask_dim = 3
        layer = 2
        seqlen = qkv_seq
        embed = embeddim
        self.embed, self.max_seq = embed, max_seq
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256 = True, False, True, False, False, False, False
        if self.calcType == 2:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, 1, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, seqlen, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(
            batch, seqlen, variate_seq)
        self.window_size = 64
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
        mask = None

        mask = np.ones(shape=(1, max_seq, max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
        mask_u = np.triu(mask, 1)
        mask_l = np.tril(mask, -self.window_size)
        mask = mask_u + mask_l
        mask *= -10000.0
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
            if self.calcType != 2:
                score += mask[:, :s, :kv]
            else:
                window_mask = np.zeros((s, max_seq)).astype(np.float16)
                if self.cacheType == 1:
                    window_mask[:s, self.window_size:] = -10000.0
                else:
                    window_mask[:s, :(kv - self.window_size)] = -10000.0
                score += window_mask[:s, :kv]
            # if mask_dim != 0:
            #     score += mask[:, :s, :kv]
            #     if is_alibi:
            #         if mask_dim == 4:
            #             score += mask[idx][:, :s, :kv]
            #         else:
            #             score += mask[:, :s, :kv]
            #     elif mask_dim == 2:
            #         score += mask[0][:s, :kv]
            #     else:
            #         score = score + mask[idx, :s, :kv]
            #         score = score + mask[:, :q_s, :kv_s]

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
        if self.mask_type == 8:
            swa_block_size = (16384 // embed // 16 * 16) if embed > 128 else 128
            if 0 < self.window_size <= 3 * swa_block_size:
                mask = torch.ones((512, 512), dtype=torch.float16)
                mask_triu = torch.triu(mask, diagonal=1)
                mask_trid = torch.tril(mask, diagonal=(-self.window_size))
                mask = mask_triu + mask_trid
                mask *= -10000.0
            else:
                mask = torch.ones((512, 512), dtype=torch.float16)
                mask_triu = torch.triu(mask, diagonal=1)
                mask_trid = torch.tril(mask, diagonal=(-(self.window_size % swa_block_size + 2 * swa_block_size)))
                mask = mask_triu + mask_trid
                mask *= -10000.0
        else:
            mask = mask.astype(np.float16).reshape(max_seq, max_seq)
            mask = torch.from_numpy(mask)
        
        
        self.mask = nd_to_nz_2d(mask)
        self.layer_id = layer_id
        # golden data
        print("out shape:", out.shape)
        out_nz = self.convert_nd_to_nz(out).astype(
            np.float16).reshape(1, heads * embed // 16, q_ntokens, 16)
        print("out_nz shape:", out_nz.shape)
        self.golden_out = out_nz
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = kv_seqlen
        self.kv_ntokens = kv_ntokens
        self.golden_out = torch.from_numpy(out_nz).to(self.data_type)
        # self.golden_out = self.golden_out.reshape(self.golden_out.shape[1], batch,self.golden_out.shape[2]//batch,self.golden_out.shape[3])
        self.golden_out = self.golden_out.reshape(self.golden_out.shape[1], self.golden_out.shape[2]//16, 16, self.golden_out.shape[3])
        self.golden_out = self.golden_out.permute(1, 0, 2, 3)
        print("golden_out shape:", self.golden_out.shape)
        self.golden_out = self.convert_nz_to_nd(self.golden_out)
        #self.golden_out = torch.from_numpy(out)
        self.q_scale = 1
        self.qk_scale = math.sqrt(1 / embeddim)

    def test_swa(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.calcType = 0
        self.mask_type = 7
        self.cacheType = 0
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":0,
                            "windowSize": self.window_size})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
    
    def test_swa_cache(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.calcType = 0
        self.mask_type = 7
        self.cacheType = 1
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":0,
                            "windowSize": self.window_size, "cacheType": 1})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
    
    def test_swa_cache_compress_mask(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.mask_type = 8
        self.calcType = 0
        self.cacheType = 1
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        print(self.k.shape)
        print(self.v.shape)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": self.mask_type, "kvcacheCfg":1,"calcType":0,
                            "windowSize": self.window_size, "cacheType": 1})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
    
    def test_swa_compress_mask(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.calcType = 0
        self.mask_type = 8
        self.cacheType = 0
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": self.mask_type, "kvcacheCfg":1,"calcType":0,
                            "windowSize": self.window_size, "cacheType": 0})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,maskin,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
    
    def test_swa_decoder(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.calcType = 2
        self.mask_type = 7
        self.cacheType = 0
        
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": self.mask_type, "kvcacheCfg":1,"calcType":self.calcType,
                            "windowSize": self.window_size, "cacheType": 0})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": self.mask_type})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
    
    def test_swa_decoder_cache(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.calcType = 2
        self.mask_type = 7
        self.cacheType = 1
        
        self.generate_data()
        q = torch.from_numpy(self.q).reshape(self.q.shape[1],self.q.shape[2]//16,16,self.q.shape[3])
        q = q.permute(1,0,2,3)
        #print("permute---------------",q.shape)
        q= self.convert_nz_to_nd(q)
        q=q.reshape(q.shape[0],q.shape[1],q.shape[2]//16,16)
        #print("after---------------",q.shape)
        k = torch_npu.npu_format_cast(torch.from_numpy(self.k).contiguous().npu(),29)#转NZ
        v = torch_npu.npu_format_cast(torch.from_numpy(self.v).contiguous().npu(),29)
        maskin = torch_npu.npu_format_cast(self.mask.contiguous().npu(),29)
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": self.mask_type, "kvcacheCfg":1,"calcType":self.calcType,
                            "windowSize": self.window_size, "cacheType": 1})
        self.param_seqlen = self.q_seqlen.tolist()
        self.param_token_offset = self.kv_seqlen.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": self.mask_type})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [q.npu(),k, v,torch.tensor(self.kv_seqlen.tolist()).to(torch.int32).npu(), torch.tensor(self.q_seqlen.tolist()).to(torch.int32).npu(), torch.from_numpy(self.layer_id).npu()])
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

    # def convert_nd_to_n2z(self, x):
    #     array_trans = self.gen_axes_for_transpose(
    #         len(x.shape) - 2, [2, 0, 1, 3])
    #     x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
    #     *_, n1, m1, m0, n0 = x_shape
    #     return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)
    # def convert_nd_to_nz(self, x):
    #     x= x.contiguous().permute(0, 1, 3, 2)
    #     x= x.reshape(x.shape[0],x.shape[1],int(x.shape[2]/16),x.shape[3],16)
    #     x = torch_npu.npu_format_cast(x.contiguous().npu(),29)
    #     return x
    
    # def convert_nd_to_nz_mask(self, x):
    #     padding = 15
    #     p = torch.zeros(int(x.shape[0]), padding,int(x.shape[2]))
    #     x=torch.cat((x,p),dim=1)
    #     x= x.contiguous().permute(0, 2, 1)
    #     x = x.reshape(x.shape[0],int(x.shape[1]/16),x.shape[2],16)

    #     x = torch_npu.npu_format_cast(x.contiguous().npu(),29)
    #     return x
    
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
        
        return [self.golden_out]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor.cpu(), golden_out_tensor, rtol=0.001, atol=0.001)

if __name__ == '__main__':
    unittest.main()
