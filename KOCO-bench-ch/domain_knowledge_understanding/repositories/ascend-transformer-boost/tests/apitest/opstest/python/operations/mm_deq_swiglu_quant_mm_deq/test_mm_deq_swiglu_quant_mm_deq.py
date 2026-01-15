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

OP_NAME = "MmDeqSwigluQuantMmDeqOperation"

def matmul_dequant(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor,
    calculate_dtype: torch.dtype, output_dtype: torch.dtype) -> torch.Tensor:
    # shape 分析
    m, k = x.shape
    _, n = weight.shape

    result = torch.matmul(x.to(torch.float32), weight.to(torch.float32)).to(calculate_dtype)
    result *= scale[None, :].to(calculate_dtype)
    result *= per_token_scale[:, None].to(calculate_dtype)
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

def generate_data(m, n, k):
    n2, k2 = k, n // 2

    # 数据生成
    x1 = torch.randint(-16, 16, size=(m, k), dtype=torch.int8)
    weight1 = torch.randint(-16, 16, size=(k, n), dtype=torch.int8)
    scale1 = random_uniform(0.004, 0.005, size=(n,), dtype=torch.float32)
    per_token_scale1 = random_uniform(0.004, 0.005, size=(m,), dtype=torch.float32)
    weight2 = torch.randint(-16, 16, size=(k2, n2,), dtype=torch.int8)
    scale2 = random_uniform(0.004, 0.005, size=(n2,), dtype=torch.float32)

    # 计算真值
    mm1_out = matmul_dequant(x1, weight1, scale1, per_token_scale1, torch.float64, torch.float64)
    swiglu_out = swiglu(mm1_out)
    x2, per_token_scale2 = quant(swiglu_out)
    true_value = matmul_dequant(x2, weight2, scale2, per_token_scale2, torch.float64, torch.float64)

    # 计算标杆
    mm1_out = matmul_dequant(x1, weight1, scale1, per_token_scale1, torch.float32, torch.float16)
    if not torch.isfinite(mm1_out.to(torch.float16)).all():
        return None

    swiglu_out = swiglu(mm1_out.to(torch.float32))
    if not torch.isfinite(swiglu_out.to(torch.float16)).all():
        return None

    x2, per_token_scale2 = quant(swiglu_out)
    if not torch.isfinite(x2).all() or not torch.isfinite(per_token_scale2).all():
        return None

    golden = matmul_dequant(x2, weight2, scale2, per_token_scale2, torch.float32, torch.float16)
    if not torch.isfinite(golden.to(torch.float16)).all():
        return None

    return (x1, weight1, scale1, per_token_scale1, weight2, scale2, true_value, golden)

def generate_data_safe(m, n, k):
    max_attempt_times = 5
    for _ in range(max_attempt_times):
        data = generate_data(m, n, k)
        if data is not None:
            return data
    raise ValueError(f"Try {max_attempt_times} times to generate data but still get illegal value.")

def calculate_golden(x1, weight1, scale1, per_token_scale1, weight2, scale2):
    weight1 = weight1.unsqueeze(0)
    scale1 = scale1.unsqueeze(0)
    group_list = torch.Tensor([x1.shape[0]]).to(torch.int64).npu()
    weight2 = weight2.unsqueeze(0)
    scale2 = scale2.unsqueeze(0)

    [out1] = torch_npu.npu_grouped_matmul(
        x=[x1], weight=[weight1], scale=[scale1], per_token_scale=[per_token_scale1],
        group_list=group_list, split_item=2, group_type=0, group_list_type=0, act_type=0, output_dtype=torch.float16
    )

    swiglu_out = torch_npu.npu_swiglu(input=out1, dim=-1)

    x2, per_token_scale2 = torch_npu.npu_dynamic_quant(input=swiglu_out, dst_type=torch.int8)

    [out2] = torch_npu.npu_grouped_matmul(
        x=[x2], weight=[weight2], scale=[scale2], per_token_scale=[per_token_scale2],
        group_list=group_list, split_item=2, group_type=0, group_list_type=0, act_type=0, output_dtype=torch.float16
    )

    torch.npu.synchronize()
    return out2

class TestMmDeqSwigluQuantMmDeq(operation_test.OperationTest):
    def golden_calc(self, in_tensor):
        return [self.golden]

    def golden_compare(self, out_tensor, golden_out_tensor):
        output, golden = out_tensor.cpu(), golden_out_tensor.cpu()
        true_value = self.true_value

        return compare_cv(true_value, golden, output)

    def test_mm_deq_swiglu_quant_mm_deq_1(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        m, n, k = 32, 4096, 7168

        (x1, weight1, scale1, per_token_scale1, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_mm_deq_swiglu_quant_mm_deq_2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        m, n, k = 256, 4096, 7168

        (x1, weight1, scale1, per_token_scale1, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

    def test_mm_deq_swiglu_quant_mm_deq_3(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        op_param = {
            "outputType": 0,
            "weightUpPermuteType": 0,
            "transposeWeightUp": False,
            "transposeWeightDown": True
        }
        tile_n = 256

        m, n, k = 576, 4096, 7168

        (x1, weight1, scale1, per_token_scale1, weight2, scale2,
            self.true_value, self.golden) = generate_data_safe(m, n, k)

        in_tensors = [
            x1.npu(),
            torch_npu.npu_format_cast(permute_weight(weight1, tile_n).contiguous().npu(), 29),
            permute_weight(scale1, tile_n).contiguous().npu(),
            per_token_scale1.npu(),
            torch_npu.npu_format_cast(weight2.mT.contiguous().npu(), 29),
            scale2.npu()
        ]
        self.execute(OP_NAME, op_param, in_tensors)

if __name__ == '__main__':
    unittest.main()
