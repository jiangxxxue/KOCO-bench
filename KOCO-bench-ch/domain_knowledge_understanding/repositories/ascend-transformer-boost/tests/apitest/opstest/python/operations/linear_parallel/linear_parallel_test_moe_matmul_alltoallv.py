#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import json
import os
import sys
import unittest

import torch
import torch.multiprocessing as mp
import torch_npu
from linear_parallel_moe_common import QuantGranularity, QuantInfo, CommType, CoCDataTypeDesc, MoeTestDate

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)

torch.manual_seed(0)

def get_err_threshold_for_one_golden(dtype:torch.dtype):
    if dtype == torch.float32:
        dtype = torch.float16
    if dtype in [torch.float16]:
        precision_threshold = 2 ** (-7)
        eb_threshold = 2 ** (-10)
    if dtype in [torch.bfloat16]:
        precision_threshold = 2 ** (-6)
        eb_threshold = 2 ** (-7)
    return precision_threshold

def get_err_threshold_for_two_golden(dtype:torch.dtype):
    if dtype in [torch.bfloat16]:
        err_threshold = 2 ** (-8)
    if dtype in [torch.float16]:
        err_threshold = 2 ** (-11)
    if dtype in [torch.float32]:
        err_threshold = 2 ** (-14)
    return err_threshold

def get_eb_threshold(dtype:torch.dtype):
    eb_threshold = 0
    if dtype in [torch.bfloat16]:
        eb_threshold = 2**(-7)
    if dtype in [torch.float16]:
        eb_threshold = 2**(-10)
    if dtype in [torch.float32]:
        eb_threshold = 2**(-14)
    return eb_threshold

def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual - golden
    EB = torch.mean(actual_error / golden_nmax)
    return EB

def one_golden_compare(tensor_a, tensor_b):
    err = get_err_threshold_for_one_golden(tensor_a.dtype)
    if torch.isnan(tensor_a).any():
        print("********Warning: npu result contains NaN!*************")
        return 1
    tensor_a = tensor_a.to(torch.float32).reshape(-1)
    tensor_b = tensor_b.to(torch.float32).reshape(-1)
    if tensor_a.size(0) == 0 and tensor_b.size(0) == 0:
        print("result is same with expect")
        return 1
    # 确定性计算要求2次npu计算结果完全一致
    if os.getenv('LCCL_DETERMINISTIC', '0') == "1":
        if torch.equal(tensor_a, tensor_b):
            return 0
        return 1
    golden_nmax = torch.clamp(torch.abs(tensor_b), min = 1)
    abs_error = torch.abs(tensor_a - tensor_b)
    print("!!!!!!!!!!!!!abs:", torch.max(abs_error))
    max_relative_error_idx = torch.argmax(abs_error)
    temp_tensor = abs_error / golden_nmax
    temp_id = torch.argmax(temp_tensor)
    max_relative_error_value_a = tensor_a[temp_id]
    max_relative_error_value_b = tensor_b[temp_id]
    result = (abs_error <= err * golden_nmax).all()
    print("re!!!!!!!!!!!!!!npu,cpu, id", result, max_relative_error_value_a, max_relative_error_value_b, temp_id)
    if result:
        print("result is same with expect")
        return 1
    else:
        print("result is error")
        return 0

def main_worker(rank, comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                data_type, quant_info, EP, TP, quant_type, out_data_ype, matrix_a_list, matrix_b_list, dequant_scale_list,
                quant_scale_list, global_tokens_per_expert_matrix, matrix_c_list, matrix_c_low_list, outpusize):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_matmul_alltoall_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    outputSize = outpusize
    acl_param = json.dumps({"type": 6, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "transWeight": bool(trans_b), "backend": "lcoc",
                            "quantType": quant_type, "outDataType": out_data_ype,
                            "moeInfo": {"epSize": world_size, "localExpertNums":
                                local_expert_nums, "tpSize": 1}})

    acl_matmul_alltoall_operation.set_param(acl_param)

    in_tensors = []
    ep_idx = rank // TP
    matrix_a_i_list = matrix_a_list[rank]
    new_M = matrix_a_i_list.shape[1]
    input_tensor = matrix_a_i_list
    input_tensor = input_tensor.reshape(new_M, K)
    
    if input_tensor.shape[0] == 0:
        input_tensor = torch.zeros(outputSize, input_tensor.shape[1], dtype=input_tensor.dtype)
    else:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, outputSize - new_M ), mode='constant', value=0)
    in_tensors.append(input_tensor.to(torch.device('npu')))

    weight_tensor = matrix_b_list[rank]
    if trans_b:
        weight_tensor = weight_tensor.reshape(local_expert_nums, N, K)
    else:
        weight_tensor = weight_tensor.reshape(local_expert_nums, K, N)
    in_tensors.append(weight_tensor.to(torch.device('npu')))
    if quant_type == 3:
        dequantScale = dequant_scale_list[rank]
        dequantScale = dequantScale.reshape(N * local_expert_nums)
        in_tensors.append(dequantScale.to(torch.device('npu')))

        quantScale = quant_scale_list[rank]
        quantScale = quantScale.reshape(quantScale.shape[0])
        empty_tensor = torch.zeros(outputSize-new_M)
        quantScale = torch.cat([quantScale, empty_tensor], dim=0)
        in_tensors.append(quantScale.to(torch.device('npu')))
    elif quant_type == 1:
        dequantScale = dequant_scale_list[rank]
        dequantScale = dequantScale.reshape(N * local_expert_nums)
        in_tensors.append(dequantScale.to(torch.device('npu')))
    

    global_tokens_per_expert_matrix = global_tokens_per_expert_matrix.reshape(EP, EP * local_expert_nums).to(dtype=torch.int32)
    
    in_tensors.append(global_tokens_per_expert_matrix.to(torch.device('npu')))

    maxOutputSize = torch.zeros(outputSize, dtype=torch.int32)
    in_tensors.append(maxOutputSize.to(torch.device('npu')))

    out_tensor = acl_matmul_alltoall_operation.execute(in_tensors)

    torch.npu.synchronize()

    golden_out_tensor = matrix_c_list[rank]
    # golden_out_tensor_low = matrix_c_low_list[rank]
    out_tensor_compare = out_tensor[0].to(torch.device('cpu'))[:golden_out_tensor.shape[1], :]
    # assert check_precision_new(out_tensor_compare, golden_out_tensor, golden_out_tensor_low)
    assert one_golden_compare(out_tensor_compare, golden_out_tensor)


def check_precision_new(tensor_a, tensor_b, tensor_c):
    if torch.isnan(tensor_a).any():
        print("********Warning: npu result contains NaN!*************")
        return 1
    epsilon = 1e-7
    d_type = tensor_a.dtype
    err_threshold = get_err_threshold_for_two_golden(d_type)
    eb_threshold = get_eb_threshold(d_type)

    tensor_a = tensor_a.to(torch.float32).reshape(-1)
    tensor_b = tensor_b.to(torch.float32).reshape(-1)
    tensor_c = tensor_c.to(torch.float32).reshape(-1)

    relative_error_npu = torch.abs(tensor_a - tensor_b) / (torch.abs(tensor_b) + epsilon)
    relative_error_cpu = torch.abs(tensor_c - tensor_b) / (torch.abs(tensor_b) + epsilon)
    if relative_error_npu.size(0) == 0 and relative_error_cpu.size(0) == 0:
        print("result is same with expect")
        return 1
    max_relative_error_npu = torch.max(relative_error_npu)
    max_relative_error_cpu = torch.max(relative_error_cpu)
    mean_relative_error_npu = torch.mean(relative_error_npu)
    mean_relative_error_cpu = torch.mean(relative_error_cpu)
    # 计算均方根误差
    mse_npu = torch.mean((tensor_a - tensor_b) ** 2)
    rmse_npu = torch.sqrt(mse_npu)
    mse_cpu = torch.mean((tensor_c - tensor_b) ** 2)
    rmse_cpu = torch.sqrt(mse_cpu)

    EB = torch.abs(get_eb(tensor_b, tensor_a))

    print("最大相对误差npu:", max_relative_error_npu)
    print("最大相对误差cpu:", max_relative_error_cpu)
    print("平均相对误差npu:", mean_relative_error_npu)
    print("平均相对误差cpu:", mean_relative_error_cpu)
    print("均方根误差npu:", rmse_npu)
    print("均方根误差cpu:", rmse_cpu)
    print("误差均衡性EB:", EB)

    max_relative_error_idx = torch.argmax(relative_error_npu)
    max_relative_error_value_a = tensor_a[max_relative_error_idx]
    max_relative_error_value_b = tensor_b[max_relative_error_idx]
    max_relative_error_value_c = tensor_c[max_relative_error_idx]

    if max_relative_error_npu / max(max_relative_error_cpu, err_threshold) >= 10:
        print(f"Max Relative Error Value: npu, golden, cpu, id: {max_relative_error_value_a.item()}, {max_relative_error_value_b.item()}, {max_relative_error_value_c.item()}, {max_relative_error_idx}")
        if one_golden_compare(tensor_a, tensor_b):
            print("resule is error")
            return 0

    if mean_relative_error_npu / max(mean_relative_error_cpu, err_threshold) >= 2 or rmse_npu / max(rmse_cpu, err_threshold) >= 2 or EB >= eb_threshold:
        print("result is error")
        return 0
    print("result is same with expect")
    return 1

def find_nearest_multiple(n: int, k: int = 512) -> int:
    if n % k == 0:
        return n
    return ((n + k - 1) // k) * k


class LinearParallelCoverOperationTest(operation_test.OperationTest):

    def test_linear_paraller_fp16_qunat(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp test start ————————")
        print("------------MATMUL REDUCESCATTER ALLTOALLVC Quantify scenarios-----------")
        world_size = 8
        comm_type = 310
        batch = 1
        M = 25
        K = 9957
        N = 868
        trans_b = 1
        quant_granularity = -1
        quant_group_size = -1
        has_quant_offset = -1
        dequant_group_size = -1
        local_expert_nums = 15
        EP = 8
        TP = 1
        out_data_type = 1
        dequant_granularity = 1
        has_dequant_offset = -1
        data_type = 2
        quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                               QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)

        if M <= 128:
            outpusize = M * 8
        else:
            outpusize = M * 2

        moedata = MoeTestDate(CommType(comm_type), world_size, batch, M, K, N, trans_b, local_expert_nums,
                              CoCDataTypeDesc(data_type), quant_info, EP, TP, outpusize)
        matrix_a_list = moedata.matrix_a_i_list
        matrix_b_list = moedata.matrix_b_list
        dequant_scale_list = moedata.dequant_scale_list
        quant_scale_list = [moedata.quant_scale_list[i].clone() for i in range(len(moedata.quant_scale_list))]
        global_tokens_per_expert_matrix = moedata.global_tokens_per_expert_matrix
        matrix_c_list = moedata.matrix_c_list
        matrix_c_low_list = moedata.matrix_c_low_list

        
        mp.spawn(main_worker, nprocs=world_size,
                 args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                       CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type, 
                       matrix_a_list, matrix_b_list, dequant_scale_list, quant_scale_list, 
                       global_tokens_per_expert_matrix, matrix_c_list, matrix_c_low_list, outpusize))

    def test_linear_paraller_fp16(self):
        i = 0
        for _ in range(10000):
            if not operation_test.get_soc_version() == 'Ascend910B':
                return
            print(f"———————— LinearParallelCoverOp test start ————————")
            print("------------MATMUL REDUCESCATTER ALLTOALLVC Non quantitative scenarios-----------")
            import random
            world_size = random.choice([2, 4, 8])
            comm_type = 310

            def generate_batch(num):
                low_range = list(range(1, 10000))
                middle_range = list(range(10000, 30000))
                high_range = list(range(30000, num * 1024))
                candidates = low_range + high_range + middle_range
                weights = [85 / 10000] * len(low_range) + [10 / 20000] * len(middle_range) + [5 / 72400] * len(
                    high_range)
                batch = random.choices(candidates, weights=weights, k=1)[0]
                return batch

            batch = 1
            M = random.randint(1, 128)

            K = generate_batch(32)

            N = generate_batch(32)
            # N = 3290
            while M > 200 * 1024:
                M = generate_batch(100)
            trans_b = random.randint(0, 1)
            # trans_b = 1
            quant_granularity = -1
            quant_group_size = -1
            has_quant_offset = -1
            dequant_group_size = -1
            # local_expert_nums = 4 # 1- 16
            local_expert_nums = random.randint(1, 16)  # 1- 16
            # EP = 8 # EP * TP = WORLDSIZE
            EP = world_size
            while EP > world_size:
                EP = random.choice([1, 2, 4, 8, 16])
            TP = world_size // EP

            dequant_granularity = random.choice([-1, 1, 3])  # 量化类型 -1 非量化
            # dequant_granularity = 3  # 量化类型 -1 非量化
            if dequant_granularity == -1:
                out_data_type = random.choice([1, 27])
                data_type = 0 if out_data_type == 1 else 1
            else:
                out_data_type = 1
                data_type = 2

            has_dequant_offset = -1
            if data_type == 2:
                kalign = find_nearest_multiple(K, 512)
            else:
                kalign = find_nearest_multiple(K, 256)
            max_int32 = 2147483647
            if M * kalign * 2 >= max_int32:
                continue

            i += 1
            print(
                f"--M:{M}--N:{N}--K:{K}--world_size:{world_size}--local_expert_nums:{local_expert_nums}--EP:{EP}--TP:{TP}--dequant_granularity:{dequant_granularity}--out_data_type:{out_data_type}--data_type:{data_type}--i:{i}")

            quant_info = QuantInfo(QuantGranularity(quant_granularity), quant_group_size, has_quant_offset,
                                   QuantGranularity(dequant_granularity), dequant_group_size, has_dequant_offset)

            moedata = MoeTestDate(CommType(comm_type), world_size, batch, M, K, N, trans_b, local_expert_nums,
                                CoCDataTypeDesc(data_type), quant_info, EP, TP, M*2)
            matrix_a_list = moedata.matrix_a_i_list
            matrix_b_list = moedata.matrix_b_list
            dequant_scale_list = moedata.dequant_scale_list
            quant_scale_list = [moedata.quant_scale_list[i].clone() for i in range(len(moedata.quant_scale_list))]
            global_tokens_per_expert_matrix = moedata.global_tokens_per_expert_matrix
            matrix_c_list = moedata.matrix_c_list
            matrix_c_low_list = moedata.matrix_c_low_list

            
            mp.spawn(main_worker, nprocs=world_size,
                    args=(comm_type, world_size, batch, M, K, N, trans_b, local_expert_nums,
                        CoCDataTypeDesc(data_type), quant_info, EP, TP, dequant_granularity, out_data_type, 
                        matrix_a_list, matrix_b_list, dequant_scale_list, quant_scale_list, 
                        global_tokens_per_expert_matrix, matrix_c_list, matrix_c_low_list))

            if i >= 700:
                break



if __name__ == '__main__':
    unittest.main()
