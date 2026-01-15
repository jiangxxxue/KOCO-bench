#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_atb
import re
import time

def check_float(tensor_list1, tensor_list2, err=2**-8):
    tensor_list1 = [tensor.cpu() for tensor in tensor_list1]
    if len(tensor_list1) != len(tensor_list2):
        print(f"Error: The lengths of tensor_list1 and tensor_list2 are not the same!")
        return False
    all_success = True  # 假设所有检查都成功，若发现任何失败的情况，则设置为 False
    for i, (out_tensor, golden_out_tensor) in enumerate(zip(tensor_list1, tensor_list2)):
        max_err = err * torch.max(torch.ones_like(golden_out_tensor), torch.abs(golden_out_tensor))
        
        error = torch.abs(out_tensor - golden_out_tensor)
        
        num_failures = torch.sum(error > max_err).item()
        
        if num_failures > 0:
            print(f"Error at index {i}: {num_failures} elements exceed the error threshold.")
            all_success = False
    
    return all_success

def check_move(tensor_list1, tensor_list2):
    tensor_list1 = [tensor.cpu() for tensor in tensor_list1]
    
    if len(tensor_list1) != len(tensor_list2):
        return False
    
    # 遍历每对 Tensor 进行比较
    for tensor1, tensor2 in zip(tensor_list1, tensor_list2):
        # 检查形状是否一致
        if tensor1.shape != tensor2.shape:
            return False
        # 检查数据是否一致
        if not torch.equal(tensor1, tensor2):
            return False
    
    return True

def extract_time(perf_time_str: str, label: str) -> int:
    match = re.search(rf"{label}:\s*(\d+)\s*us", perf_time_str)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"{label} not found in the input string.")

def run_perf_test(operation, input_tensors, num_runs=100):
    def get_diff_time(start_time, end_time, runtime):
        python_time = (end_time - start_time) * 1_000_000
        diff_time = python_time - run_time
        torch.npu.synchronize()
        return diff_time

    perf_diff_time = float('inf')
    for _ in range(num_runs):
        start_time = time.time()
        operation_outputs = operation.forward(input_tensors)
        run_time = torch_atb.Prof.get_prof_stats().get_run_time_stats(operation.name)[-1]
        end_time = time.time()
        python_time = (end_time - start_time) * 1_000_000
        diff_time = get_diff_time(start_time, end_time, run_time)
        perf_diff_time = min(perf_diff_time, diff_time)

    assert perf_diff_time < 300
    print("Time for converting C++ to Python:", perf_diff_time, "us")

def ret_check(ret):
    if ret != 0:
        raise RuntimeError(f"Set device failed, error code (ret): {ret}")