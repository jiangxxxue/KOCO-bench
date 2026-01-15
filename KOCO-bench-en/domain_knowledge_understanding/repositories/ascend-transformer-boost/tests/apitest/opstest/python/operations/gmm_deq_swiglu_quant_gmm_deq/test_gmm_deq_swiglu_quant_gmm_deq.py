#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import torch
import torch_npu

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test
from precision_calcu import compare_cv

OP_NAME = "GmmDeqSwigluQuantGmmDeqOperation"

def grouped_matmul_dequant(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor,
    group_list: torch.Tensor, calculate_dtype: torch.dtype, output_dtype: torch.dtype) -> torch.Tensor:
    # shape 分析
    m, k = x.shape
    group_num, _, n = weight.shape

    result = torch.empty(size=(m, n), dtype=calculate_dtype)
    for group_idx in range(group_num):
        start = 0 if group_idx == 0 else group_list[group_idx - 1]
        end = group_list[group_idx]
        res = torch.matmul(
            x[start:end, :].to(torch.float32),
            weight[group_idx, :, :].to(torch.float32)
        ).to(calculate_dtype)
        res *= scale[group_idx, :].to(calculate_dtype)
        res *= per_token_scale[start:end, None].to(calculate_dtype)
        result[start:end, :] = res
    return result.to(output_dtype)

def swiglu(x: torch.Tensor) -> torch.Tensor:
    x0, gate = x.chunk(2, dim=-1)
    swish = x0 * torch.sigmoid(x0)
    y = swish * gate
    return y

def quant(x: torch.Tensor):
    x_row_max = torch.max(torch.abs(x), dim=-1).values
    quant_result = x * 127. / x_row_max[:, None]
    y = torch.round(quant_result).to(torch.int8)
    scale = (x_row_max / 127.).to(torch.float32)
    return y, scale

def permute_weight(w: torch.Tensor, tile_n: int):
    *dims, n = w.shape
    order = list(range(len(dims))) + [-2, -3, -1]
    return w.reshape(*dims, 2, n // tile_n, tile_n // 2).permute(order).reshape(*dims, n)

def random_uniform(lower: float, upper: float, size: tuple, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand(size=size, dtype=dtype) * (upper - lower) + lower

def generate_data(group_list, m, n, k):
    group_num = len(group_list)
    m_actual = group_list[-1]
    n2, k2 = k, n // 2

    # 数据生成
    x1 = torch.randint(-16, 16, size=(m, k), dtype=torch.int8)
    weight1 = torch.randint(-16, 16, size=(group_num, k, n), dtype=torch.int8)
    scale1 = random_uniform(0.004, 0.005, size=(group_num, n,), dtype=torch.float32)
    per_token_scale1 = random_uniform(0.004, 0.005, size=(m,), dtype=torch.float32)
    group_list = torch.Tensor(group_list).to(torch.int64)
    weight2 = torch.randint(-16, 16, size=(group_num, k2, n2,), dtype=torch.int8)
    scale2 = random_uniform(0.004, 0.005, size=(group_num, n2,), dtype=torch.float32)
    # 中间参数
    x2 = torch.zeros(size=(m, k2), dtype=torch.int8)
    per_token_scale2 = torch.zeros(size=(m,), dtype=torch.float32)

    # 计算真值
    gmm1_out = grouped_matmul_dequant(x1, weight1, scale1, per_token_scale1, group_list, torch.float64, torch.float64)
    swiglu_out = swiglu(gmm1_out[:m_actual])
    x2[:m_actual], per_token_scale2[:m_actual] = quant(swiglu_out)
    true_value = grouped_matmul_dequant(x2, weight2, scale2, per_token_scale2, group_list, torch.float64, torch.float64)

    # 计算标杆
    gmm1_out = grouped_matmul_dequant(x1, weight1, scale1, per_token_scale1, group_list, torch.float32, torch.float16)
    if not torch.isfinite(gmm1_out[:m_actual].to(torch.float16)).all():
        return None

    swiglu_out = swiglu(gmm1_out[:m_actual].to(torch.float32))
    if not torch.isfinite(swiglu_out.to(torch.float16)).all():
        return None

    x2[:m_actual], per_token_scale2[:m_actual] = quant(swiglu_out)
    if not torch.isfinite(x2[:m_actual]).all() or not torch.isfinite(per_token_scale2[:m_actual]).all():
        return None

    golden = grouped_matmul_dequant(x2, weight2, scale2, per_token_scale2, group_list, torch.float32, torch.float16)
    if not torch.isfinite(golden[:m_actual].to(torch.float16)).all():
        return None

    return (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2, true_value, golden)

def generate_data_safe(group_list, m, n, k):
    max_attempt_times = 5
    for _ in range(max_attempt_times):
        data = generate_data(group_list, m, n, k)
        if data is not None:
            return data
    raise ValueError(f"Try {max_attempt_times} times to generate data but still get illegal value.")

class TestGmmDeqSwigluQuantGmmDeq(operation_test.OperationTest):
    def golden_calc(self, in_tensor):
        return [self.golden]

    def golden_compare(self, out_tensor, golden_out_tensor):
        output, golden = out_tensor[:self.m_actual].cpu(), golden_out_tensor[:self.m_actual].cpu()
        true_value = self.true_value[:self.m_actual]

        return compare_cv(true_value, golden, output)

    def test_gmm_deq_swiglu_quant_gmm_deq_1(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [432, 496]
        m, n, k = 512, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(group_list, m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_gmm_deq_swiglu_quant_gmm_deq_2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [360]
        m, n, k = 2464, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(group_list, m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_gmm_deq_swiglu_quant_gmm_deq_3(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "groupListType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        group_list = [128, 256, 384]
        m, n, k = 2464, 4096, 7168
        self.m_actual = group_list[-1]

        (x1, weight1, scale1, per_token_scale1, group_list, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(group_list, m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            group_list.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

if __name__ == '__main__':
    unittest.main()
