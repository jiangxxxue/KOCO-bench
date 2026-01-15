# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 
import logging
import sys
import os
import unittest
import math
import numpy as np
import torch
import random
import json
import torch.nn.functional as F
import torch_npu
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test
from precision_calcu import *

OP_NAME = "MlaPreprocessOperation"
QUANTMAX = 127
QUANTMIN = -128
block_size = 128

def process_deq_scale(deq_scale: torch.Tensor) -> np.ndarray:
    ret = torch.frombuffer(deq_scale.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
    return ret


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat

def transdata_3d(nd_mat, block_size: tuple = (16, 16)):
    if nd_mat.ndim != 3:
        raise ValueError("Expected a 3-dimensional input array.")
    B, K, N = nd_mat.shape
    processed_slices = []

    for batch_index in range(B):
        current_slice = nd_mat[batch_index]
        nz_mat = transdata(current_slice, block_size)
        processed_slices.append(nz_mat)
    
    result = torch.stack(processed_slices, axis=0)
    return result

class TestMLAPrepross(operation_test.OperationTest):
    def __set_envs(self, env: dict):
        if env:
            for key, value in env.items():
                os.environ[key] = value

    def __unset_envs(self, env: dict):
        if env:
            for key, _ in env.items():
                os.environ[key] = ""
                
    def __get_npu_device(self):
        npu_device = os.environ.get("MKI_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def padHeadNum(self, headdim_sin_cos):
        return torch.tile(headdim_sin_cos, (1, self.headNum))

    def rotateHalfX(self, q_temp):
        # 拆分成 head_num 个 [n,head_dim] 的二维向量
        # print("============== q_temp", q_temp)
        # import pdb;pdb.set_trace()
        q_splits = torch.chunk(q_temp, self.headNum, dim=1)
        # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
        # import pdb;pdb.set_trace()
        processed_q_splits = []
        for q_split in q_splits:
            # print("============== q_split", q_split)
            # 分割第二维
            first_half, second_half = torch.chunk(q_split, 2, dim=1)
            # 拼接回 [n,head_dim] 的二维向量
            # import pdb;pdb.set_trace()
            processed_q_split = torch.cat((-second_half, first_half), dim=1)
            processed_q_splits.append(processed_q_split)

        # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
        return torch.cat(processed_q_splits, dim=1)

    def rotateHalf2(self, q_temp):
        # 拆分成 head_num 个 [n,head_dim] 的二维向量
        # print("============== q_temp", q_temp)
        q_splits = np.split(q_temp, self.headNum, axis=1)
        # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
        processed_q_splits = []
        for q_split in q_splits:
            # print("============== q_split", q_split)
            # 分割第二维
            first_half, second_half = np.split(q_split, 2, axis=1)
            # 拼接回 [n,head_dim] 的二维向量
            processed_q_split = np.concatenate((second_half, first_half), axis=1)
            processed_q_splits.append(processed_q_split)

        # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
        return np.concatenate(processed_q_splits, axis=1)

    def RopeConcatGolden(self, q, sin, cos, concatInput):
        pad_sin = self.padHeadNum(sin)
        pad_cos = self.padHeadNum(cos)
        # import pdb;pdb.set_trace()
        # self.qcosOut = self.rotateHalf(q)
        rope_res = q * pad_cos + self.rotateHalfX(q) * pad_sin
        rope_res = rope_res.reshape(self.input_token_num, self.headNum, self.rope_hidden_size)
        rope_res = rope_res.to(self.dtype)

        return torch.cat((concatInput.to(self.dtype), rope_res), dim=2)

    def rmsNormGolden(self, x, gamma):
        x_float32 = x.to(torch.float32)
        square_sum = torch.sum(torch.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / torch.sqrt(square_sum /self.rms_hidden_size + self.epsilon)  
        gamma_float32 = gamma.to(torch.float32)
        rmsNorm = rms * x_float32 * gamma_float32
        result = rmsNorm.to(self.dtype)
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:.15f}'.format})
        return result

    def rmsNormPerTokenGolden(self, intensors):
        out_shape = intensors[0].shape
        input0 = intensors[0].float()
        input1 = intensors[1].float()
        input2 = intensors[2].float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input0 * factor * input1
        if torch.numel(input2) != 0:
            output = output + input2
        output = output.half()
        scale, _ = torch.max(torch.abs(output), dim=-1, keepdim=True)
        scale = torch.div(torch.tensor([127], dtype=torch.float32), scale)
        output = torch.mul(output, scale)
        out_scale = torch.div(torch.tensor([1], dtype=torch.float32), scale)
        output = torch.clamp(torch.round(output.float()), min=QUANTMIN, max=QUANTMAX)
        # print(torch.tensor(out_scale).squeeze(-1).to(torch.float32))
        return [output.to(torch.int8), out_scale.squeeze(-1).to(torch.float32)]

    def rmsNormPerTokenNpuGolden(self, intensors):
        # RmsNorm
        in_tensors = [intensors[0].npu(), intensors[1].npu()]
        out_tensors = [torch.zeros(intensors[0].shape, dtype=self.dtype)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        op_name = "RmsNormOperation"
        op_param =json.dumps({"layerType":1, "normParam":{"quantType": 0, "epsilon": 1e-6}})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, out_tensors_npu)

        # Add beta
        in_tensors = [out_tensors_npu[0], intensors[2].npu()]
        out_tensors = [torch.zeros(intensors[0].shape, dtype=self.dtype)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in out_tensors]
        op_name = "ElewiseOperation"
        op_param =json.dumps({"elewiseType":8})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, out_tensors_npu)

        # DynamicQuant
        res_npu = [
            torch.zeros(intensors[0].shape, dtype=torch.int8).npu(),
            torch.zeros((self.input_token_num), dtype=torch.float32).npu(),
        ]
        op_name = "ElewiseOperation"
        op_param =json.dumps({"elewiseType":19, "quantParam":{"asymmetric":False}})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out([out_tensors_npu[0]], res_npu)
        return res_npu

    def ppMatmulDequantNpuGolden(self, intensors, outtensors):
        # Matmul
        dim1 = intensors[1].shape[0]
        bias = torch.zeros((1, dim1), dtype=torch.int32)
        deScale = torch.ones((dim1), dtype=torch.float32)
        if self.dtype == torch.float16:
            deScale = process_deq_scale(deScale)
        in_tensors = [intensors[0], intensors[1], bias, deScale]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu() for i in outtensors]
        op_name = "LinearOperation"
        op_param =json.dumps({"transposeA": False, "transposeB": True, "hasBias": True,
                              "outDataType": 27 if self.dtype == torch.bfloat16 else 1})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, out_tensors_npu)
        mm1Out1 = out_tensors_npu[0].to(torch.int32).to(torch.float32)
        # Mul perChannelDescale
        self.mm1_mul_descale_out = torch.zeros((self.input_token_num, dim1), dtype=torch.float16).npu()
        op_name = "ElewiseOperation"
        op_param =json.dumps({"elewiseType":9})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out([mm1Out1.to(torch.float16), intensors[3].to(torch.float16).npu()], [self.mm1_mul_descale_out])
        # Mul perTokenDescale
        out_tensors_npu = [torch.zeros((self.input_token_num, dim1), dtype=torch.float16).npu()]
        self.operation.execute_out([self.mm1_mul_descale_out, intensors[4].to(torch.float16)], out_tensors_npu)

        # Cast
        outtensors[0] = out_tensors_npu[0].to(self.dtype)

    def rotateHalf(self, k_temp):
        first_half, second_half = torch.chunk(k_temp, 2, dim=1)
        processed_k_split = torch.cat((-second_half, first_half), dim=1)
        return processed_k_split

    def RopeGolden(self, keyRope, sin, cos):
        RopeGolden = keyRope * cos + self.rotateHalf(keyRope) * sin
        return RopeGolden

    def RACGolden(self, keyRAC, slotMapping, keycacheout_golden):
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size
            token_key = keyRAC[i]

            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden

    def RmsNormAndRopeAndReshapeAndCacheGolden(self, x, gamma, keyRope, cos, sin, slotMapping, keycachein):
        rmsNormOutput = self.rmsNormGolden(x, gamma)
        ropeOutput = self.RopeGolden(keyRope, sin, cos)
        ropeReshape = ropeOutput.reshape(self.input_token_num, 1, self.rope_hidden_size)

        keyRAC = torch.cat((rmsNormOutput, ropeReshape), axis=-1)
        return self.RACGolden(keyRAC, slotMapping, keycachein)

    def rms_norm_quant_calc(self, input, gamma, beta, quantScale, quantOffset):
        out_shape = input.shape
        scale = 1.0 / quantScale.float().item()
        offset = quantOffset.float().item()
        square_sum = torch.sum(torch.square(input.float()), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input.float() * factor * gamma.float()
        output = (output + beta.float()) * scale + offset
        output = torch.round(output).half()
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half)).to(torch.int8)
        return output

    def calc_vec_mm_atb_data(self, N, headNum, data_type, quant_mode=0):
        hiddenStrate = 7168
        blockNum = 192
        blockSize = 128
        headdim = 576
        self.input_token_num = N
        self.rms_hidden_size = 512
        self.rope_hidden_size = 64
        self.headNum = headNum
        self.epsilon = 1e-6
        self.dtype = data_type

        self.input1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(N, 7168))).to(data_type)#
        self.gamma1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(hiddenStrate))).to(data_type)
        self.quantScale1 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset1 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)
        self.wdqkv = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(2112, 7168))).to(torch.int8)#

        self.deScale1 = torch.rand((2112), dtype=torch.float32) / 1000
        self.gamma2 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(1536))).to(data_type)
        self.quantScale2 = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(1))).to(data_type)
        self.quantOffset2 = torch.from_numpy(np.random.uniform(-128.0, 127.0, size=(1))).to(torch.int8)

        self.wuq = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum * 192, 1536))).to(torch.int8)#

        self.deScale2 = torch.rand((headNum * 192), dtype=torch.float32) / 1000

        self.gamma3 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(512))).to(data_type)
        self.sin1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.cos1 = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(N, 64))).to(data_type)
        self.keyCache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(blockNum, blockSize, 1, headdim))).to(data_type)

        self.slotMapping = torch.from_numpy(np.random.choice(192 * 128, N, replace=False).astype(np.int32)).to(torch.int32)

        self.wuk = torch.from_numpy(np.random.uniform(-2.0, 2.0, size=(headNum, 128, 512))).to(data_type)#
        self.sin2 = self.sin1
        self.cos2 = self.cos1

        self.bias1 = torch.from_numpy(np.random.randint(-10, 10, (1, 2112)).astype(np.int32)).to(torch.int32)
        self.bias2 = torch.from_numpy(np.random.randint(-10, 10, (1, headNum * 192)).astype(np.int32)).to(torch.int32)

        self.beta1 = torch.from_numpy(np.random.randint(-2, 2, (hiddenStrate)).astype(np.float16)).to(data_type)
        self.beta2 = torch.from_numpy(np.random.randint(-2, 2, (1536)).astype(np.float16)).to(data_type)
        
        self.calc_vec_mm_data(N, headNum, data_type, quant_mode)

        ## RmsNorm
        print("====================RmsNorm0====================")
        self.rms1Out1_npu = torch.zeros((N, 7168), dtype=torch.int8)
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)
        in_tensors = [self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1]

        if quant_mode == 0:
            out_tensors = [self.rms1Out1_npu]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu()
                                for i in out_tensors]
            op_name = "RmsNormOperation"
            op_param =json.dumps({"layerType":1, "normParam":{"quantType": 2, "epsilon": 1e-6}})
            self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
            self.op_param = op_param
            self.operation.set_param(op_param)
            self.operation.execute_out(in_tensors_npu, out_tensors_npu)
            self.rms1Out1_npu = out_tensors_npu[0].cpu().clone()
        else:
            out_tensors_npu = self.rmsNormPerTokenNpuGolden(in_tensors)
            pertoken_descale = out_tensors_npu[1].unsqueeze(1).expand(-1, 2112).npu()

        ##Ppmatmul
        print("====================Ppmatmul0====================")
        if quant_mode == 0:
            self.mm1Out1_npu = torch.zeros((N, 2112), dtype=data_type)
            in_tensors = [out_tensors_npu[0], self.wdqkv, self.bias1, self.deScale1]
            out_tensors = [self.mm1Out1_npu]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu()
                                for i in out_tensors]
            op_name = "LinearOperation"
            op_param =json.dumps({"transposeA": False, "transposeB": True, "hasBias": True,
                                  "outDataType": 27 if self.dtype == torch.bfloat16 else 1})
            self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
            self.op_param = op_param
            self.operation.set_param(op_param)
            self.operation.execute_out(in_tensors_npu, out_tensors_npu)
            self.mm1Out1_npu = out_tensors_npu[0].cpu().clone()
            print(self.mm1Out1_npu.size())
        else:
            in_tensors = [out_tensors_npu[0], self.wdqkv, self.bias1, self.deScale1, pertoken_descale]
            out_tensors_npu = [torch.zeros((N, 2112), dtype=data_type)]
            self.ppMatmulDequantNpuGolden(in_tensors, out_tensors_npu)
            # self.deScale1 = process_deq_scale(deq_scale=self.deScale1)
            # self.deScale2 = process_deq_scale(deq_scale=self.deScale2)

        ##SplitV
        print("====================SplitV0====================")
        splitSize = [512, 64, 1536]
        splitVDim = 1
        in_tensors_npu = [out_tensors_npu[0]]
        Split1_out_tensors_npu = []
        shape = in_tensors_npu[0].shape
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split1_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())
        op_name = "SplitOperation"
        op_param =json.dumps({"splitNum": 3, "splitDim": splitVDim, "splitSizes": splitSize})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, Split1_out_tensors_npu)
        print(Split1_out_tensors_npu[0].size())

        ## RmsNorm
        print("====================RmsNorm1====================")
        if quant_mode == 0:
            self.rms2Out_npu = torch.zeros((N, 1536), dtype=torch.int8)
            in_tensors = [Split1_out_tensors_npu[2], self.gamma2, self.beta2, self.quantScale2, self.quantOffset2]
            out_tensors = [self.rms2Out_npu]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu()
                                for i in out_tensors]
            op_name = "RmsNormOperation"
            op_param =json.dumps({"layerType":1, "normParam":{"quantType": 2, "epsilon": 1e-6}})
            self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
            self.op_param = op_param
            self.operation.set_param(op_param)
            self.operation.execute_out(in_tensors_npu, out_tensors_npu)
            self.rms2Out_npu = out_tensors_npu[0].cpu().clone()
            print(self.rms2Out_npu.size())
        else:
            in_tensors = [Split1_out_tensors_npu[2].cpu(), self.gamma2, self.beta2, self.quantScale2, self.quantOffset2]
            out_tensors_npu = self.rmsNormPerTokenNpuGolden(in_tensors)
            pertoken_descale = out_tensors_npu[1].unsqueeze(1).expand(-1, headNum * 192).npu()

        # ##Ppmatmul
        print("====================Ppmatmul1====================")
        if quant_mode == 0:
            self.mm2Out_npu = torch.zeros((N, headNum * 192), dtype=data_type)
            in_tensors = [out_tensors_npu[0], self.wuq, self.bias2, self.deScale2]
            out_tensors = [self.mm2Out_npu]
            in_tensors_npu = [tensor.npu() for tensor in in_tensors]
            mm2out_tensors_npu = [out_tensors[i] if isinstance(i, int) else i.npu()
                                for i in out_tensors]
            op_name = "LinearOperation"
            op_param =json.dumps({"transposeA": False, "transposeB": True, "hasBias": True,
                                  "outDataType": 27 if self.dtype == torch.bfloat16 else 1})
            self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
            self.op_param = op_param
            self.operation.set_param(op_param)
            self.operation.execute_out(in_tensors_npu, mm2out_tensors_npu)
            self.mm2Out_npu = mm2out_tensors_npu[0].cpu().clone()
            print(self.mm2Out_npu.size())
        else:
            in_tensors = [out_tensors_npu[0], self.wuq, self.bias2, self.deScale2, pertoken_descale]
            mm2out_tensors_npu = [torch.zeros((N, headNum * 192), dtype=data_type)]
            self.ppMatmulDequantNpuGolden(in_tensors, mm2out_tensors_npu)

        # # ##SplitV
        print("====================SplitV1====================")
        splitSize = [128, 64]
        splitVDim = 2
        in_tensors = mm2out_tensors_npu[0].reshape(N, headNum, 192)
        in_tensors_npu = in_tensors.npu()
        Split2_out_tensors_npu = []
        shape = in_tensors_npu.shape
        in_tensors_npu = [in_tensors_npu]
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            Split2_out_tensors_npu.append(torch.zeros(slice_shape, dtype=data_type).npu())
        op_name = "SplitOperation"
        op_param =json.dumps({"splitNum": 2, "splitDim": splitVDim, "splitSizes": splitSize})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, Split2_out_tensors_npu)
        print(Split2_out_tensors_npu[0].size())

        #EinSum
        print("====================EinSum====================")
        # EinSumInput = torch.transpose(Split2_out_tensors_npu[0], 0, 1)
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = headNum, N, 128, 512
        in_tensors = [Split2_out_tensors_npu[0], self.wuk]
        Einsumout_tensors = [torch.zeros((N, headNum, 512), dtype=data_type)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Einsumout_tensors_npu = [tensor.npu() for tensor in Einsumout_tensors]
        op_name = "LinearOperation"
        op_param =json.dumps({"matmulType": 1, "transposeA": False, "transposeB": False, "hasBias": False, "outDataType": -1})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, Einsumout_tensors_npu)

        # Einsumout_tensors_npu = [torch.transpose(Einsumout_tensors_npu[0], 0, 1)]
        self.einsumOut = Einsumout_tensors_npu[0].cpu().clone()
        print(self.einsumOut.size())
        # RopeQConcat
        print("====================RopeQConcat====================")
        self.qOut_npu = torch.zeros((N, headNum, 576), dtype=data_type)
        Split2_out_tensors_npu[1] = Split2_out_tensors_npu[1].cpu().reshape(N, headNum * 64)
        in_tensors = [Split2_out_tensors_npu[1], self.cos2, self.sin2, Einsumout_tensors_npu[0]]
        qOut_tensors = [self.qOut_npu]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        qout_tensors_npu = [qOut_tensors[i] if isinstance(i, int) else i.npu()
                            for i in qOut_tensors]
        op_name = "RopeQConcatOperation"
        op_param =json.dumps({})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, qout_tensors_npu)
        qout_tensors_npu[0] = torch.cat([qout_tensors_npu[0].cpu()[:,:,64:],qout_tensors_npu[0].cpu()[:,:,:64]], dim = 2)
        self.qOut_npu = qout_tensors_npu[0].cpu().clone()
        print(self.qOut_npu.size())
        # #Rope
        print("====================Rope====================") 
        rotaryCoeff = 2
        self.RopeOut0 = torch.zeros((N, 64), dtype=data_type)
        self.RopeOut1 = torch.zeros((N, 64), dtype=data_type)
        seqlen = torch.randint(1, 2, (N,), dtype=torch.int32)
        in_tensors = [Split1_out_tensors_npu[1].reshape(N, 1 * 64), Split1_out_tensors_npu[1].reshape(N, 1 * 64), self.cos1, self.sin1, seqlen]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        Ropeout_tensors_npu = [self.RopeOut0.npu(), self.RopeOut1.npu()]

        op_name = "RopeOperation"
        op_param =json.dumps({"rotaryCoeff": rotaryCoeff})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, Ropeout_tensors_npu)
        print(Ropeout_tensors_npu[0].size())

        #RmsNorm
        print("====================RmsNorm2====================") 
        in_tensors = [Split1_out_tensors_npu[0].reshape(N, 1, 512), self.gamma3]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        self.RmsNormOut = torch.zeros((N, 1, 512), dtype=data_type).npu()
        RmsNormOut_tensors_npu = [self.RmsNormOut]

        op_name = "RmsNormOperation"
        op_param =json.dumps({"layerType":1, "normParam":{"epsilon": 1e-6}})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, RmsNormOut_tensors_npu)
        
        out_tensors_npu = RmsNormOut_tensors_npu
        #Concat
        print("====================Concat====================") 
        in_tensors = [RmsNormOut_tensors_npu[0], Ropeout_tensors_npu[0].cpu().reshape(N, 1, 64)]
        ConCat2out_tensors = [torch.zeros((N, 1, 576), dtype=data_type)]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        ConCat2out_tensors_npu = [tensor.npu() for tensor in ConCat2out_tensors]

        op_name = "ConcatOperation"
        op_param =json.dumps({"concatDim": 2})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, ConCat2out_tensors_npu)

        #Reshape&Cache
        print("====================Reshape&Cache====================") 
        in_tensors = [ConCat2out_tensors_npu[0], self.keyOutTensor, self.slotMapping]
        out_tensors = [1]
        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]

        op_name = "ReshapeAndCacheOperation"
        op_param =json.dumps({"kvCacheCfg": 1})
        self.operation = torch.classes.OperationTorch.OperationTorch(op_name)
        self.op_param = op_param
        self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors_npu, out_tensors_npu)
        self.keyout_npu = out_tensors_npu[0].cpu().clone()

    def calc_vec_mm_data(self, N, headNum, data_type, quant_mode):
        if quant_mode == 0:
            mm1In = self.rms_norm_quant_calc(self.input1, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)
        else:
            [mm1In, perTokenDescale1] = self.rmsNormPerTokenGolden([self.input1, self.gamma1, self.beta1])

        self.rmsquantOut1 = mm1In.clone()
        mm1Out = torch.matmul(mm1In.to(torch.float32), self.wdqkv.transpose(0, 1).to(torch.float32))
        if quant_mode == 0:
            mm1Out = mm1Out.to(torch.int32) + self.bias1
            mm1Out = (mm1Out.to(torch.float32) * self.deScale1).to(data_type)
        else:
            perTokenDescale1 = perTokenDescale1.unsqueeze(1).expand(-1, 2112)
            mm1Out = (mm1Out.to(torch.float32) * self.deScale1 * perTokenDescale1).to(data_type)
        self.mm1Out1 = mm1Out
        if data_type == torch.float16 and quant_mode == 0:
            self.deScale1 = process_deq_scale(deq_scale=self.deScale1)
        mm1OutSplit1, mm1OutSplit2 = torch.split(mm1Out, [576, 1536], dim=1)

        if quant_mode == 0:
            rms2Out = self.rms_norm_quant_calc(
                mm1OutSplit2, self.gamma2, self.beta2, self.quantScale2, self.quantOffset2
            )
        else:
            [rms2Out, perTokenDescale2] = self.rmsNormPerTokenGolden([mm1OutSplit2, self.gamma2, self.beta2])
        self.rmsquantOut2 = rms2Out.clone()
        mm2Out = torch.matmul(rms2Out.to(torch.float32), self.wuq.transpose(0, 1).to(torch.float32))
        if quant_mode == 0:
            mm2Out = mm2Out.to(torch.int32) + self.bias2
            mm2Out = (mm2Out.to(torch.float32) * self.deScale2).to(data_type)
        else:
            perTokenDescale2 = perTokenDescale2.unsqueeze(1).expand(-1, headNum * 192)
            mm2Out = (mm2Out.to(torch.float32) * self.deScale2 * perTokenDescale2).to(data_type)

        self.mm2Out2 = mm2Out
        if data_type == torch.float16 and quant_mode == 0:
            self.deScale2 = process_deq_scale(deq_scale=self.deScale2)

        mm11OutSplit1, mm12OutSplit1 = torch.split(mm1OutSplit1, [512, 64], dim=1)
        mm11OutSplit1 = mm11OutSplit1.reshape(N, 1, 512)
        self.keyOutTensor = self.keyCache.clone()
        self.keyOut1 = self.RmsNormAndRopeAndReshapeAndCacheGolden(
            mm11OutSplit1, self.gamma3, mm12OutSplit1, self.cos1, self.sin1, self.slotMapping, self.keyCache
        )
        mm2Out = mm2Out.reshape(N, headNum, 192)
        mm2OutSplit1, mm2OutSplit2 = torch.split(mm2Out, [128, 64], dim=2)
        self.bmmOut = torch.permute(
            torch.matmul(torch.permute(mm2Out[:, :, :128], (1, 0, 2)).float(), self.wuk.float()),
            (1, 0, 2),
        )
        self.gg = mm2OutSplit2.clone()
        qOut = self.RopeConcatGolden(mm2OutSplit2.reshape(N, headNum * 64), self.sin2, self.cos2, self.bmmOut)
        self.qOut = qOut.to(data_type)

    def golden_calc(self, in_tensors):
        # out_tensors = torch.load("out_tensors.pt")
        # return [out_tensors[-2], out_tensors[-1]]
        if self.cacheMode == 1:
            return [self.qOut_npu[..., 0:512], self.keyout_npu[..., 0:512], self.qOut_npu[..., 512:576], self.keyout_npu[..., 512:576]]
        else:
            return [self.qOut_npu, self.keyout_npu]

    def compare_data(self, tensor1, tensor2):
        out = tensor1.flatten()
        out_len = out.shape[0]
        golden = tensor2.flatten()
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        print(f"maxDiff {max_diff}")
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)
        limit_error = torch.maximum(
            torch.abs(golden * 0.001), torch.tensor(0.001))
        strict_limit_error = torch.maximum(
            torch.abs(golden * 0.003), torch.tensor(0.003))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(
            diff, strict_limit_error).sum().item()
        print("1/1000 Accuracy is %f",
                        1 - float(error_count) / out_len)
        print("3/1000 Accuracy is %f",  1 -
                        float(strict_error_count) / out_len)
        print("accuracy is correct: %r", (float(strict_error_count) / out_len) <= 0.001)

    def golden_compare(self, out_tensors, golden_tensors):
        # self.compare_data(out_tensors.npu(), golden_tensors.npu())
        if self.cacheMode == 1:
            if self.compare_count == 0:
                result_double = compare_cv(self.qOut_npu[..., 0:512].npu(), self.qOut[..., 0:512].npu(), out_tensors.npu())
            elif self.compare_count == 1:
                result_double = compare_cv(self.keyout_npu[..., 0:512].npu(), self.keyOut1[..., 0:512].npu(), out_tensors.npu())
            elif self.compare_count == 2:
                result_double = compare_cv(self.qOut_npu[..., 512:576].npu(), self.qOut[..., 512:576].npu(), out_tensors.npu())
            elif self.compare_count == 3:
                result_double = compare_cv(self.keyout_npu[..., 512:576].npu(), self.keyOut1[..., 512:576].npu(), out_tensors.npu())
        else:
            if self.compare_count == 0:
                result_double = compare_cv(self.qOut_npu.npu(), self.qOut.npu(), out_tensors.npu())
            else:
                result_double = compare_cv(self.keyout_npu.npu(), self.keyOut1.npu(), out_tensors.npu())
        self.compare_count += 1
        return result_double
 
    def test_mla_preprocess_pertoken(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.compare_count = 0
        self.cacheMode = 0
        self.quant_mode = 1
        N = 32
        headNum = 32
        data_type = torch.float16
        OP_NAME = "MlaPreprocessOperation"
        PARAM = json.dumps({"quantMode":self.quant_mode})
        self.calc_vec_mm_atb_data(N,headNum,data_type,self.quant_mode)
        self.keyCache = self.keyCache.npu()
        self.execute_out(OP_NAME, PARAM,
                    [self.input1.npu(),
                    self.gamma1.npu(),
                    self.beta1.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wdqkv, (16, 32)).contiguous().npu(), 29),
                    self.deScale1.npu(),
                    torch.tensor([]).npu(),
                    self.gamma2.npu(),
                    self.beta2.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wuq, (16, 32)).contiguous().npu(), 29),
                    self.deScale2.npu(),
                    torch.tensor([]).npu(),
                    self.gamma3.npu(),
                    self.cos1.npu(),
                    self.sin1.npu(),
                    self.wuk.npu(),
                    self.keyCache.npu(),
                    torch.tensor([]).npu(),
                    self.slotMapping.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu()],
                    [torch.zeros((N, headNum, 576), dtype=data_type).npu(),
                    self.keyCache.npu()])

    def test_mla_preprocess_bf16_pertoken(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.compare_count = 0
        self.cacheMode = 0
        self.quant_mode = 1
        N = 32
        headNum = 32
        data_type = torch.bfloat16
        OP_NAME = "MlaPreprocessOperation"
        PARAM = json.dumps({"quantMode":self.quant_mode})
        self.calc_vec_mm_atb_data(N,headNum,data_type,self.quant_mode)
        self.keyCache = self.keyCache.npu()
        self.execute_out(OP_NAME, PARAM,
                    [self.input1.npu(),
                    self.gamma1.npu(),
                    self.beta1.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wdqkv, (16, 32)).contiguous().npu(), 29),
                    self.deScale1.npu(),
                    torch.tensor([]).npu(),
                    self.gamma2.npu(),
                    self.beta2.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wuq, (16, 32)).contiguous().npu(), 29),
                    self.deScale2.npu(),
                    torch.tensor([]).npu(),
                    self.gamma3.npu(),
                    self.cos1.npu(),
                    self.sin1.npu(),
                    self.wuk.npu(),
                    self.keyCache.npu(),
                    torch.tensor([]).npu(),
                    self.slotMapping.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu()],
                    [torch.zeros((N, headNum, 576), dtype=data_type).npu(),
                    self.keyCache.npu()])

    def test_mla_preprocess_split_pertoken(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.compare_count = 0
        self.cacheMode = 1
        self.quant_mode = 1
        N = 32
        headNum = 32
        data_type = torch.float16
        OP_NAME = "MlaPreprocessOperation"
        PARAM = json.dumps({"cacheMode":self.cacheMode, "quantMode":self.quant_mode})
        self.calc_vec_mm_atb_data(N,headNum,data_type,self.quant_mode)
        self.keyCache = self.keyCache.npu()
        self.execute_out(OP_NAME, PARAM,
                    [self.input1.npu(),
                    self.gamma1.npu(),
                    self.beta1.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wdqkv, (16, 32)).contiguous().npu(), 29),
                    self.deScale1.npu(),
                    torch.tensor([]).npu(),
                    self.gamma2.npu(),
                    self.beta2.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wuq, (16, 32)).contiguous().npu(), 29),
                    self.deScale2.npu(),
                    torch.tensor([]).npu(),
                    self.gamma3.npu(),
                    self.cos1.npu(),
                    self.sin1.npu(),
                    self.wuk.npu(),
                    self.keyCache[..., 0:512].npu(),
                    self.keyCache[..., 512:576].npu(),
                    self.slotMapping.npu(),                    
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu()],
                    [torch.zeros((N, headNum, 512), dtype=data_type).npu(),
                    self.keyCache[..., 0:512].npu(),
                    torch.zeros((N, headNum, 64), dtype=data_type).npu(),
                    self.keyCache[..., 512:576].npu()])

    def test_mla_preprocess_bf16_split_pertoken(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.compare_count = 0
        self.cacheMode = 1
        self.quant_mode = 1
        N = 32
        headNum = 32
        data_type = torch.bfloat16
        OP_NAME = "MlaPreprocessOperation"
        PARAM = json.dumps({"cacheMode":self.cacheMode, "quantMode":self.quant_mode})
        self.calc_vec_mm_atb_data(N,headNum,data_type,self.quant_mode)
        self.keyCache = self.keyCache.npu()
        self.execute_out(OP_NAME, PARAM,
                    [self.input1.npu(),
                    self.gamma1.npu(),
                    self.beta1.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wdqkv, (16, 32)).contiguous().npu(), 29),
                    self.deScale1.npu(),
                    torch.tensor([]).npu(),
                    self.gamma2.npu(),
                    self.beta2.npu(),
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu(),
                    torch_npu.npu_format_cast(transdata(self.wuq, (16, 32)).contiguous().npu(), 29),
                    self.deScale2.npu(),
                    torch.tensor([]).npu(),
                    self.gamma3.npu(),
                    self.cos1.npu(),
                    self.sin1.npu(),
                    self.wuk.npu(),
                    self.keyCache[..., 0:512].npu(),
                    self.keyCache[..., 512:576].npu(),
                    self.slotMapping.npu(),                    
                    torch.tensor([]).npu(),
                    torch.tensor([]).npu()],
                    [torch.zeros((N, headNum, 512), dtype=data_type).npu(),
                    self.keyCache[..., 0:512].npu(),
                    torch.zeros((N, headNum, 64), dtype=data_type).npu(),
                    self.keyCache[..., 512:576].npu()])

if __name__ == "__main__":
    unittest.main()
