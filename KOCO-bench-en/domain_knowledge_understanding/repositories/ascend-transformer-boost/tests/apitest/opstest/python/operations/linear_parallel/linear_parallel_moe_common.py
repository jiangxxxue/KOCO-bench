#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

from enum import Enum
import math
import torch
from collections import namedtuple
import collections
import numpy


class CommType(Enum):
    PURE_MATMUL = 101
    ALL_REDUCE = 102
    REDUCE_SCATTER = 103
    ALL_GATHER = 104
    ALL_GATHER_V2 = 105
    MATMUL_2D = 111
    ALLTOALLV_ALLGATHER_MATMUL = 305
    MATMUL_REDUCESCATTER_ALLTOALLV = 306
    ALLTOALLVC_ALLGATHER_MATMUL = 307
    MATMUL_REDUCESCATTER_ALLTOALLVC = 308
    ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN = 309
    MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN = 310


class QuantGranularity(Enum):
    QUANT_GRANULARITY_UNDEFINED = -1
    PER_TENSOR = 0
    PER_CHANNEL = 1
    PER_GROUP = 2
    PER_TOKEN = 3
    FLOAT32_SCALE_PER_CHANNEL = 4


class CoCDataTypeDesc(Enum):
    COC_DATA_TYPE_UNDEFINED = -1
    FP16FP16_FP32_FP16 = 0
    BF16BF16_FP32_BF16 = 1
    INT8INT8_INT32_FP16 = 2
    INT8INT8_INT32_BF16 = 3
    FP16INT8_INT32_FP16 = 4
    BF16INT8_INT32_BF16 = 5
    FP16INT8_FP32_FP16 = 6
    BF16INT8_FP32_BF16 = 7
    FP16INT4_FP32_FP16 = 8
    BF16INT4_FP32_BF16 = 9
    MAX = 10


CoCDataType = namedtuple('CoCDataType',
                         ['activation_dtype', 'weight_dtype', 'l0c_dtype', 'output_dtype', 'l0c_dtype_low'])

supported_coc_data_type_dict = {
    CoCDataTypeDesc.FP16FP16_FP32_FP16: CoCDataType(torch.float16, torch.float16, torch.float32, torch.float16,
                                                    torch.float16),
    CoCDataTypeDesc.BF16BF16_FP32_BF16: CoCDataType(torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16,
                                                    torch.bfloat16),
    CoCDataTypeDesc.INT8INT8_INT32_FP16: CoCDataType(torch.int8, torch.int8, torch.int32, torch.float16, torch.float16),
    CoCDataTypeDesc.INT8INT8_INT32_BF16: CoCDataType(torch.int8, torch.int8, torch.int32, torch.bfloat16,
                                                     torch.bfloat16),
    CoCDataTypeDesc.FP16INT8_FP32_FP16: CoCDataType(torch.float16, torch.int8, torch.float32, torch.float16,
                                                    torch.float16),
    CoCDataTypeDesc.BF16INT8_FP32_BF16: CoCDataType(torch.bfloat16, torch.int8, torch.float32, torch.bfloat16,
                                                    torch.bfloat16),
}


def generate_random_tensor(size, dtype):
    if dtype in [torch.float16, torch.bfloat16, torch.float32]:
        return torch.randn(size=size, dtype=dtype)
    elif dtype is torch.int8:
        return torch.randint(-16, 16, size=size, dtype=dtype)
    elif dtype is torch.int32:
        return torch.randint(-1024, 1024, size=size, dtype=dtype)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


class QuantInfo:
    def __init__(self, quant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, quant_group_size=None,
                 has_quant_offset=False,
                 dequant_granularity=QuantGranularity.QUANT_GRANULARITY_UNDEFINED, dequant_group_size=None,
                 has_dequant_offset=False):
        self.quant_granularity = quant_granularity
        self.quant_group_size = quant_group_size
        self.has_quant_offset = has_quant_offset
        self.dequant_granularity = dequant_granularity
        self.dequant_group_size = dequant_group_size
        self.has_dequant_offset = has_dequant_offset

        self.dequant_scale_origin = None
        self.dequant_args_shape = None

        self.quant_scale = None
        self.quant_offset = None
        self.dequant_scale = None
        self.dequant_offset = None

    def get_quant_args_shape(self, shape_info):
        m = shape_info[0]
        n = shape_info[1]
        granularity = self.dequant_granularity
        group_size = self.dequant_group_size
        if granularity is QuantGranularity.PER_TENSOR:
            return 1, 1
        elif granularity is QuantGranularity.PER_CHANNEL:
            return 1, n
        elif granularity is QuantGranularity.PER_GROUP:
            return math.ceil(m / group_size), n
        elif granularity is QuantGranularity.PER_TOKEN:
            return m, 1
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def broadcast_quant_args(self, quant_arg, shape_info):
        granularity = self.dequant_granularity
        m = shape_info[0]
        n = shape_info[1]
        group_size = self.dequant_group_size
        if granularity is QuantGranularity.PER_GROUP:
            return quant_arg.repeat_interleave(group_size, dim=0)[:m]
        else:
            return quant_arg.expand(m, n)

    def get_pertoken_quant_tensor(self, input_info):
        shape_info = [input_info[0], input_info[2]]
        quant_args_shape = self.get_quant_args_shape(shape_info)
        self.quant_scale = generate_random_tensor(size=quant_args_shape, dtype=torch.float32) / 127
        broadcast_quant_scale = self.broadcast_quant_args(self.quant_scale, shape_info)
        return broadcast_quant_scale

    def get_output_dequant_tensor(self, input_info, l0c_dtype, coc_dtype_desc):
        # W8A8, output dequant
        shape_info = [input_info[0], input_info[2]]
        is_per_token = 0
        if self.dequant_granularity is QuantGranularity.PER_TOKEN:
            self.dequant_granularity = QuantGranularity.PER_CHANNEL
            is_per_token = 1

        dequant_args_shape = self.get_quant_args_shape(shape_info)
        self.dequant_args_shape = dequant_args_shape
        self.dequant_scale_origin = generate_random_tensor(size=dequant_args_shape, dtype=torch.float32) / 127

        if coc_dtype_desc is CoCDataTypeDesc.INT8INT8_INT32_FP16:
            self.dequant_scale_origin = ((self.dequant_scale_origin.view(torch.int32) >> 13) << 13).view(torch.float32)
            self.dequant_scale = torch.zeros(size=dequant_args_shape, dtype=torch.int64)
            self.dequant_scale.view(torch.float32)[:, ::2] = self.dequant_scale_origin
        else:
            self.dequant_scale = self.dequant_scale_origin
        broadcast_scale = self.broadcast_quant_args(self.dequant_scale_origin, shape_info)
        if self.has_dequant_offset == 1:
            self.dequant_offset = generate_random_tensor(size=dequant_args_shape, dtype=l0c_dtype)
            broadcast_offset = self.broadcast_quant_args(self.dequant_offset, shape_info)
        else:
            broadcast_offset = torch.zeros(dequant_args_shape, dtype=l0c_dtype)
        if is_per_token:
            self.dequant_granularity = QuantGranularity.PER_TOKEN
        return broadcast_offset, broadcast_scale

    def get_weight_dequant_tensor(self, input_info, activation_dtype):
        shape_info = [input_info[1], input_info[2]]
        dequant_args_shape = self.get_quant_args_shape(shape_info)

        self.dequant_scale = generate_random_tensor(size=dequant_args_shape, dtype=activation_dtype) / 127
        broadcast_scale = self.broadcast_quant_args(self.dequant_scale, shape_info)

        if self.has_dequant_offset == 1:
            self.dequant_offset = generate_random_tensor(size=dequant_args_shape, dtype=activation_dtype)
            broadcast_offset = self.broadcast_quant_args(self.dequant_offset, shape_info)
        else:
            broadcast_offset = torch.zeros_like(broadcast_scale)

        dequant_dtype = torch.float32 if activation_dtype is torch.bfloat16 else activation_dtype

        return broadcast_offset, broadcast_scale, dequant_dtype

    def get_moe_dequant_tensor(self, m_matrix_a, n, TP, l0c_dtype):
        shape_info = [m_matrix_a, n]
        broadcast_scale = self.broadcast_quant_args(self.dequant_scale_origin, shape_info)
        if self.has_dequant_offset == 1:
            broadcast_offset = self.broadcast_quant_args(self.dequant_offset, shape_info)
        else:
            broadcast_offset = torch.zeros(self.dequant_args_shape, dtype=l0c_dtype)
        return broadcast_offset, broadcast_scale


class MoeTestDate:
    def __init__(self, comm_type, rank_size, batch_size, m, k, n, trans_b, expert_per_rank,
                 coc_dtype_desc, quant_info, EP, TP, maxOutputSize=-1):
        activation_dtype, weight_dtype, l0c_dtype, output_dtype, l0c_dtype_low = supported_coc_data_type_dict[coc_dtype_desc]
        self.matrix_a_list = []
        self.matrix_b_list = []
        self.matrix_c_list = []
        self.matrix_c_low_list = []

        self.dequant_offset_list = []
        self.dequant_scale_list = []
        self.quant_scale_list = []
        self.matrix_a = generate_random_tensor(size=(1, m, k), dtype=activation_dtype)
        self.matrix_b = generate_random_tensor(size=(batch_size, k, n), dtype=weight_dtype)
        for i in range(rank_size):
            self.matrix_a_list.append(self.matrix_a)
            # self.matrix_b_list.append(self.matrix_b)
        self.expert_num = expert_per_rank * EP
        self.expert_per_rank = expert_per_rank
        self.sequence_length = m
        self.input_info = [m, k, n]
        self.batch_size = batch_size
        self.maxOutputSize = maxOutputSize
        self.trans_b = trans_b
        self.m = m
        self.k = k
        self.n = n

        self.input_splits, self.output_splits, self.num_local_tokens_per_expert = self.get_moe_input_output_splits(
            expert_per_rank, EP)
        if comm_type is CommType.ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN or comm_type is CommType.MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN:
            self.matrix_a_i_list = self.alltoall_nopermute(self.matrix_a, k, activation_dtype, EP)
            if maxOutputSize > 0:
                for i in range(EP):
                    self.matrix_a_i_list[i] = self.matrix_a_i_list[i][:, :maxOutputSize, :]
        else:
            self.matrix_a_i_list = self.alltoall(self.matrix_a, k, activation_dtype, EP)
            self.matrix_a_i_list = self.allgather(self.matrix_a_i_list, k, activation_dtype, EP, TP)

        if comm_type in [CommType.ALLTOALLVC_ALLGATHER_MATMUL_HIDDEN]:
            self.generate_matrix_c_for_moe_309(coc_dtype_desc, rank_size, TP, EP, l0c_dtype, output_dtype, quant_info)
        if comm_type in [ CommType.MATMUL_REDUCESCATTER_ALLTOALLVC_HIDDEN]:
            self.generate_matrix_c_for_moe_310(coc_dtype_desc, rank_size, TP, EP, l0c_dtype, output_dtype, quant_info)
        if trans_b:
            self.matrix_b = self.matrix_b.transpose(1, 2).contiguous()
        self.matrix_b = self.matrix_b.repeat(1, expert_per_rank, 1)
        for i in range(rank_size):
            self.matrix_b_list.append(self.matrix_b)

    def get_num_local_tokens_per_expert(self):
        numpy.random.seed(0)
        indices = numpy.random.randint(self.expert_num, size=self.sequence_length)
        item_dict = collections.Counter(indices)
        num_local_tokens_per_expert = [item_dict.get(i, 0) for i in range(self.expert_num)]
        return num_local_tokens_per_expert, indices

    def get_moe_input_output_splits(self, expert_per_rank, EP):
        num_local_tokens_per_expert = [None] * EP
        for i in range(EP):
            num_local_tokens_per_expert[i], indices = self.get_num_local_tokens_per_expert()
        all_gather_res = num_local_tokens_per_expert[0]
        for i in range(1, EP):
            all_gather_res = all_gather_res + num_local_tokens_per_expert[i]
        input_splits = [None] * EP
        for i in range(EP):
            input_splits[i] = numpy.sum(numpy.array(num_local_tokens_per_expert[i]).reshape(EP, expert_per_rank),
                                        axis=1)
        output_splits = [None] * EP
        num_global_tokens_per_expert = numpy.array(all_gather_res).reshape(EP, self.expert_num)
        num_global_tokens_per_local_expert = [None] * EP
        for i in range(EP):
            num_global_tokens_per_local_expert[i] = num_global_tokens_per_expert[:,
                                                    i * expert_per_rank:(i + 1) * expert_per_rank]
            output_splits[i] = numpy.sum(num_global_tokens_per_local_expert[i], axis=-1)

        self.global_tokens_per_expert_matrix = torch.from_numpy(numpy.array(num_local_tokens_per_expert)).reshape(EP,
                                                                                                                  EP * expert_per_rank).to(
            dtype=torch.int32)
        return input_splits, output_splits, num_local_tokens_per_expert

    def alltoall(self, matrix_a, k, type, EP):
        offset_input_splits = [0] * EP
        m_matrix_a = [sum(self.output_splits[i]) for i in range(EP)]
        matrix_a_i_list = [generate_random_tensor(size=(self.batch_size, m_matrix_a[i], k), dtype=type) for i in
                           range(EP)]
        for i in range(EP):
            sum_offset = 0
            for j in range(EP):
                matrix_a_i_list[i][:, sum_offset:sum_offset + self.output_splits[i][j], :] = matrix_a[:,
                                                                                             offset_input_splits[j]:
                                                                                             offset_input_splits[j] +
                                                                                             self.output_splits[i][j],
                                                                                             :]
                sum_offset += self.output_splits[i][j]
                offset_input_splits[j] += self.input_splits[j][i]
        return matrix_a_i_list

    def alltoall_nopermute(self, matrix_a, k, type, EP):
        m_matrix_a = [sum(self.output_splits[i]) for i in range(EP)]
        matrix_a_i_list = [generate_random_tensor(size = (self.batch_size, m_matrix_a[i], k), dtype=type) for i in range(EP)]
        for src_ep in range(EP):
            src_offset = 0
            for local_expert_idx in range(self.expert_per_rank):
                expert_idx = local_expert_idx + src_ep * self.expert_per_rank
                for dst_ep in range(EP):
                    dst_expert_offset = 0
                    dst_expert_len = self.num_local_tokens_per_expert[dst_ep][expert_idx]
                    for i in range(expert_idx):
                        dst_expert_offset += self.num_local_tokens_per_expert[dst_ep][i]
                    matrix_a_i_list[src_ep][:,src_offset:src_offset + dst_expert_len, :] = matrix_a[:,dst_expert_offset:dst_expert_offset + dst_expert_len, :]
                    src_offset += dst_expert_len
        return matrix_a_i_list

    def allgather(self, matrix_a_i_list, k, type, EP, TP):
        for i in range(EP):
            tmp_matrix_a = generate_random_tensor(size=(self.batch_size, matrix_a_i_list[i].shape[1] * TP, k),
                                                  dtype=type)
            tmp_sum = 0
            for j in range(EP):
                tmp_matrix_a[:, tmp_sum * TP: tmp_sum * TP + TP * self.output_splits[i][j], :] = (
                    matrix_a_i_list[i][:, tmp_sum: tmp_sum + self.output_splits[i][j], :].repeat(1, TP, 1))
                tmp_sum += self.output_splits[i][j]
            matrix_a_i_list[i] = tmp_matrix_a
        return matrix_a_i_list

    def generate_matrix_c_for_moe_305_307(self, rank, coc_dtype_desc, TP, EP, l0c_dtype, output_dtype,
                                          quant_info):
        if coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
            ep_idx = rank // TP
            self.matrix_c = torch.matmul(self.matrix_a_i_list[ep_idx].to(l0c_dtype), self.matrix_b.to(l0c_dtype)).to(
                output_dtype)
        elif coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            # w8a8
            assert quant_info.dequant_granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN]
            quant_info.get_output_dequant_tensor(self.input_info, l0c_dtype, coc_dtype_desc)
            self.matrix_dequant_scale = quant_info.dequant_scale
            self.matrix_dequant_offset = quant_info.dequant_offset
            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                quant_info.get_pertoken_quant_tensor(self.input_info)
                quant_scale = quant_info.quant_scale
                quant_scale = quant_scale.unsqueeze(0)
                quant_scale_alltoall = self.alltoall(quant_scale, 1, torch.float32, EP)
                quant_scale_allgather = self.allgather(quant_scale_alltoall, 1, torch.float32, EP, TP)
                quant_scale_allgather = quant_scale_allgather
                quant_scale = quant_scale.squeeze(0)

                ep_idx = rank // TP
                quant_scale_allgather[ep_idx] = quant_scale_allgather[ep_idx].squeeze(0)
                self.matrix_quant_scale = quant_scale

            ep_idx = rank // TP
            self.matrix_c = torch.matmul(self.matrix_a_i_list[ep_idx].to(torch.float32),
                                         self.matrix_b.to(torch.float32)).to(l0c_dtype)
            broadcast_offset, broadcast_scale = quant_info.get_moe_dequant_tensor(self.output_splits[ep_idx],
                                                                                  self.input_info[2], TP, l0c_dtype)
            self.matrix_c = ((self.matrix_c + broadcast_offset).to(torch.float32) * broadcast_scale).to(output_dtype)

            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                broadcast_quant_scale = quant_info.broadcast_quant_args(quant_scale_allgather[ep_idx],
                                                                        [sum(self.output_splits[ep_idx]) * TP,
                                                                         self.input_info[2]])
                self.matrix_c = (self.matrix_c.to(torch.float32) * broadcast_quant_scale).to(output_dtype)

    def generate_matrix_c_for_moe_306_308(self, rank, coc_dtype_desc, TP, EP, l0c_dtype, l0c_dtype_low,
                                          quant_info):

        if coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
            tmp_matrix_c = torch.matmul(self.matrix_a.to(l0c_dtype), self.matrix_b.to(l0c_dtype))
            tmp_matrix_c_low = tmp_matrix_c.to(l0c_dtype_low)
            matrix_c_out = tmp_matrix_c.clone()
            matrix_c_out_low = tmp_matrix_c_low.clone()
            for i in range(TP - 1):
                matrix_c_out += tmp_matrix_c
                matrix_c_out_low += tmp_matrix_c_low
            self.matrix_c = matrix_c_out
        elif coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            assert quant_info.dequant_granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN]
            broadcast_offset, broadcast_scale = quant_info.get_output_dequant_tensor(self.input_info, l0c_dtype,
                                                                                     coc_dtype_desc)
            self.matrix_dequant_scale = quant_info.dequant_scale
            self.matrix_dequant_offset = quant_info.dequant_offset
            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                quant_info.get_pertoken_quant_tensor(self.input_info)
                quant_scale = quant_info.quant_scale
                quant_scale = quant_scale.unsqueeze(0)
                quant_scale_alltoall = self.alltoall(quant_scale, 1, torch.float32, EP)
                quant_scale_allgather = self.allgather(quant_scale_alltoall, 1, torch.float32, EP, TP)
                quant_scale_allgather = quant_scale_allgather
                quant_scale = quant_scale.squeeze(0)

                ep_idx = rank // TP
                quant_scale_allgather[ep_idx] = quant_scale_allgather[ep_idx].squeeze(0)
                self.matrix_quant_scale = quant_scale_allgather[ep_idx]
                broadcast_quant_scale = quant_info.broadcast_quant_args(quant_scale,
                                                                        [self.input_info[0], self.input_info[2]])

            tmp_matrix_c = torch.matmul(self.matrix_a.to(torch.float32), self.matrix_b.to(torch.float32)).to(l0c_dtype)
            matrix_c_out = ((tmp_matrix_c + broadcast_offset).to(torch.float32) * broadcast_scale).to(l0c_dtype_low)

            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                matrix_c_out = (matrix_c_out.to(torch.float32) * broadcast_quant_scale).to(l0c_dtype_low)

            tmp_matrix_c = matrix_c_out.clone()

            for i in range(TP - 1):
                matrix_c_out += tmp_matrix_c
            self.matrix_c = matrix_c_out
    
    def get_pvalue(self, data_type_len):
        maxPvalue = (self.k + 255) / 256
        pValue = 7
        if pValue > maxPvalue:
            pValue = maxPvalue
        if self.m < 128:
            pValue = (self.k + 256 - 1)/ 256
        maxPeerMemPerRank = 200 * 1024 * 1024 / data_type_len / 2
        if pValue *256 *self.m > maxPeerMemPerRank:
            pValue = maxPeerMemPerRank / self.m / 256
        ubMoveNum = 28672
        maxUbPingPongSize = ubMoveNum / 2
        if pValue *256 > maxUbPingPongSize:
            pValue = maxUbPingPongSize / 256
        return pValue

    def generate_matrix_c_for_moe_309(self, coc_dtype_desc, rank_size, TP, EP, l0c_dtype, output_dtype, quant_info):
        if l0c_dtype == torch.int32:
            data_type_len = 2
        else:
            data_type_len = 2
        pValue = int(self.get_pvalue(data_type_len))
        print("pvalue!!!!!!!!!!!!!!", pValue)
        if coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
            for i in range(rank_size):
                ep_idx = i // TP
                matrix_c_low = torch.zeros((1,self.matrix_a_i_list[ep_idx].size(1),self.n)).to(output_dtype)
                loop = math.ceil(self.k / (pValue * 256))
                for j in range(loop):
                    st = j * pValue * 256
                    ed = min(self.k, (j + 1) * pValue * 256)
                    matrix_c_j = torch.matmul(self.matrix_a_i_list[ep_idx][:,:,st:ed].to(torch.float32), self.matrix_b[:,st:ed,:].to(torch.float32))
                    matrix_c_j_low = matrix_c_j.to(output_dtype)
                    matrix_c_low = matrix_c_low + matrix_c_j_low
                matrix_c = torch.matmul(self.matrix_a_i_list[ep_idx].to(l0c_dtype), self.matrix_b.to(l0c_dtype))
                self.matrix_c_list.append(matrix_c)
                self.matrix_c_low_list.append(matrix_c_low)

        elif coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            assert quant_info.dequant_granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN, QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]

            quant_info.get_output_dequant_tensor(self.input_info, l0c_dtype, coc_dtype_desc)
            self.matrix_dequant_scale = quant_info.dequant_scale
            self.matrix_dequant_offset = quant_info.dequant_offset
            self.matrix_dequant_scale = self.matrix_dequant_scale.repeat(1,self.expert_per_rank)
            for i in range(rank_size):
                self.dequant_scale_list.append(self.matrix_dequant_scale)

            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                quant_info.get_pertoken_quant_tensor(self.input_info)
                quant_scale = quant_info.quant_scale
                quant_scale = quant_scale.unsqueeze(0)
                quant_scale_alltoall = self.alltoall_nopermute(quant_scale, 1, torch.float32, EP)

                quant_scale = quant_scale.squeeze(0)
                for rank in range(rank_size):
                    ep_idx = rank // TP
                    quant_scale_alltoall[ep_idx] = quant_scale_alltoall[ep_idx].squeeze(0)
                    if self.maxOutputSize > 0:
                        quant_scale_alltoall[ep_idx] = quant_scale_alltoall[ep_idx][:self.maxOutputSize, :]
                    self.quant_scale_list.append(quant_scale)


            for i in range(rank_size):
                ep_idx = i // TP
                matrix_c_low = torch.zeros((1,self.matrix_a_i_list[ep_idx].size(1),self.n)).to(output_dtype)
                broadcast_offset, broadcast_scale = quant_info.get_moe_dequant_tensor(self.matrix_a_i_list[ep_idx].size(1), self.input_info[2], TP, l0c_dtype)
                loop = math.ceil(self.k / (pValue * 256))
                for j in range(loop):
                    st = j * pValue * 256
                    ed = min(self.k, (j + 1) * pValue * 256)
                    matrix_c_j = torch.matmul(self.matrix_a_i_list[ep_idx][:,:,st:ed].to(torch.float32), self.matrix_b[:,st:ed,:].to(torch.float32))
                    matrix_c_j = (matrix_c_j * broadcast_scale)
                    matrix_c_j_low = matrix_c_j.to(output_dtype)
                    matrix_c_low = matrix_c_low + matrix_c_j_low
                matrix_c_low = matrix_c_low.to(torch.float32)
                matrix_c = torch.matmul(self.matrix_a_i_list[ep_idx].to(torch.float32), self.matrix_b.to(torch.float32)).to(l0c_dtype)
                
                matrix_c = ((matrix_c + broadcast_offset).to(torch.float32) * broadcast_scale)
                # matrix_c_low = ((matrix_c_low + broadcast_offset).to(torch.float32) * broadcast_scale)
                if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                    # print("!"*30, quant_scale_alltoall[ep_idx].shape)
                    broadcast_quant_scale = quant_info.broadcast_quant_args(quant_scale_alltoall[ep_idx], [self.matrix_a_i_list[ep_idx].size(1), self.input_info[2]])
                    matrix_c = (matrix_c * broadcast_quant_scale)
                    matrix_c_low = (matrix_c_low * broadcast_quant_scale)
                # self.matrix_c = matrix_c
                self.matrix_c_list.append(matrix_c)
                self.matrix_c_low_list.append(matrix_c_low.to(output_dtype))
                # self.matrix_c_low = matrix_c_low.to(output_dtype)

    def cal_trunc(self, EP):
        self.global_tokens_per_expert_matrix_temp = self.global_tokens_per_expert_matrix.clone()
        self.global_tokens_per_expert_matrix_temp = self.global_tokens_per_expert_matrix_temp.reshape(-1)
        for i in range(EP):
            sum_tokens = 0
            for local_expert_id in range(self.expert_per_rank):
                expert_id = i * self.expert_per_rank + local_expert_id
                for j in range(EP):
                    if (sum_tokens + self.global_tokens_per_expert_matrix_temp[j * self.expert_num + expert_id]) >= self.maxOutputSize:
                        self.global_tokens_per_expert_matrix_temp[j * self.expert_num + expert_id] = self.maxOutputSize - sum_tokens
                        sum_tokens = self.maxOutputSize
                    else:
                        sum_tokens += self.global_tokens_per_expert_matrix_temp[j * self.expert_num + expert_id]
        # print("self.global_tokens_per_expert_matrix_temp", self.global_tokens_per_expert_matrix_temp)

    def generate_matrix_c_for_moe_310(self, coc_dtype_desc, rank_size, TP, EP, l0c_dtype, l0c_dtype_low,
                                      quant_info):        
        if self.maxOutputSize > 0:
            self.cal_trunc(EP)
        if coc_dtype_desc in [CoCDataTypeDesc.FP16FP16_FP32_FP16, CoCDataTypeDesc.BF16BF16_FP32_BF16]:
            matrix_c_out = torch.matmul(self.matrix_a.to(l0c_dtype), self.matrix_b.to(l0c_dtype))
            # if coc_dtype_desc == CoCDataTypeDesc.BF16BF16_FP32_BF16:
            #     matrix_c_out_low = torch.matmul(self.matrix_a, self.matrix_b)
            # else:
            #     matrix_c_out_low = matrix_c_out.to(l0c_dtype_low)
            tmp_offset = 0
            for rank in range(rank_size):
                for ep_idx in range(EP):
                    # tmp_offset = 0
                    for local_expert_id in range(self.expert_per_rank):
                        expert_id = local_expert_id + ep_idx * self.expert_per_rank
                        if self.global_tokens_per_expert_matrix_temp[rank * self.expert_num + expert_id] < self.num_local_tokens_per_expert[rank][expert_id]:
                            l = tmp_offset + self.global_tokens_per_expert_matrix_temp[rank * self.expert_num + expert_id]
                            r = tmp_offset + self.num_local_tokens_per_expert[rank][expert_id]
                            matrix_c_out[:,l:r,:] = 0
                            # matrix_c_out_low[:,l:r,:] = 0
                        tmp_offset += self.num_local_tokens_per_expert[rank][expert_id]
                self.matrix_c_list.append(matrix_c_out)
                # self.matrix_c_low_list.append(matrix_c_out_low)
            # print("self.matrix_c:", self.matrix_c)
            
        elif coc_dtype_desc in [CoCDataTypeDesc.INT8INT8_INT32_FP16, CoCDataTypeDesc.INT8INT8_INT32_BF16]:
            assert quant_info.dequant_granularity in [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TENSOR,
                                                      QuantGranularity.PER_TOKEN, QuantGranularity.FLOAT32_SCALE_PER_CHANNEL]
            broadcast_offset, broadcast_scale = quant_info.get_output_dequant_tensor(self.input_info, l0c_dtype, coc_dtype_desc)
            self.matrix_dequant_scale = quant_info.dequant_scale
            self.dequant_offset_list = quant_info.dequant_offset
            self.matrix_dequant_scale = self.matrix_dequant_scale.repeat(1,self.expert_per_rank)
            for i in range(rank_size):
                self.dequant_scale_list.append(self.matrix_dequant_scale)
            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                quant_info.get_pertoken_quant_tensor(self.input_info)
                quant_scale = quant_info.quant_scale
                quant_scale = quant_scale.unsqueeze(0)
                quant_scale_alltoall = self.alltoall_nopermute(quant_scale, 1, torch.float32, EP)
                quant_scale = quant_scale.squeeze(0)

                for rank in range(rank_size):
                    ep_idx = rank // TP
                    quant_scale_alltoall[ep_idx] = quant_scale_alltoall[ep_idx].squeeze(0)                
                    self.quant_scale_list.append(quant_scale_alltoall[ep_idx])
                broadcast_quant_scale = quant_info.broadcast_quant_args(quant_scale, [self.input_info[0], self.input_info[2]])

            tmp_matrix_c = torch.matmul(self.matrix_a.to(torch.float32), self.matrix_b.to(torch.float32)).to(l0c_dtype)
            matrix_c_out = ((tmp_matrix_c + broadcast_offset).to(torch.float32) * broadcast_scale).to(torch.float32)

            if quant_info.dequant_granularity is QuantGranularity.PER_TOKEN:
                matrix_c_out = (matrix_c_out * broadcast_quant_scale).to(torch.float32)
            # matrix_c_out_low = matrix_c_out.to(l0c_dtype_low)

            tmp_offset = 0
            for rank in range(rank_size):
                for ep_idx in range(EP):
                    # tmp_offset = 0
                    for local_expert_id in range(self.expert_per_rank):
                        expert_id = local_expert_id + ep_idx * self.expert_per_rank
                        if self.global_tokens_per_expert_matrix_temp[rank * self.expert_num + expert_id] < self.num_local_tokens_per_expert[rank][expert_id]:
                            l = tmp_offset + self.global_tokens_per_expert_matrix_temp[rank * self.expert_num + expert_id]
                            r = tmp_offset + self.num_local_tokens_per_expert[rank][expert_id]
                            matrix_c_out[:,l:r,:] = 0
                            # matrix_c_out_low[:,l:r,:] = 0
                        tmp_offset += self.num_local_tokens_per_expert[rank][expert_id]
                self.matrix_c_list.append(matrix_c_out)
                # self.matrix_c_low_list.append(matrix_c_out_low)

