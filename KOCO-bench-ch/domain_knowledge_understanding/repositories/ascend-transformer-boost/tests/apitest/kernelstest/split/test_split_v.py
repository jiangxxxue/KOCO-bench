#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import sys
import op_test
import random

# 设置随机种子以确保测试的可重复性
random.seed(42)
np.random.seed(42)

# 随机数数组
generalize = [15, 16, 128]
OP_NAME = "SplitOperation"

class TestSplit(op_test.OpTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        splitSize = self.op_desc["specificParam"]["splitSize"]
        splitVDim = self.op_desc["specificParam"]["splitVDim"][0]
        y = torch.split(x, splitSize, dim=splitVDim)
        return list(y)

    def golden_compare(self, out_tensors, golden_out_tensors):
        splitNum = self.op_desc["specificParam"]["splitNum"]
        for i in range(splitNum):
            out_tensor = out_tensors[i]
            golden_tensor = golden_out_tensors[i]
            if not torch.equal(out_tensor, golden_tensor):
                return False
        return True

    def generate_random_split_sizes(self, splitNum, total_size):
        """生成随机split_size"""
        sizes = []
        remaining_size = total_size
        for i in range(splitNum - 1):
            min_size = 1
            max_size = remaining_size - (splitNum - len(sizes) - 1) * min_size
            size = random.randint(min_size, max_size)
            sizes.append(size)
            remaining_size -= size
        sizes.append(remaining_size)
        return sizes

    def generate_dimension_variations(self, base_value, multiplier, count):
        """生成增加和减少的维度值列表"""
        variations = []
        for i in range(1, count + 1):
            increase = base_value + multiplier * i
            decrease = base_value - multiplier * i
            variations.append(increase)
            if decrease > 0:
                variations.append(decrease)
        return variations

    def run_test_case(self, shape, splitVDim, dataType, splitNum):
        """设置输入张量的形状"""
        total_size = shape[splitVDim]
        splitSize = self.generate_random_split_sizes(splitNum, total_size)

        # 设置操作参数
        OP_PARAM = {
            "splitNum": splitNum,
            "splitVDim": [splitVDim],
            "splitSize": splitSize
        }
        self.set_param(OP_NAME, OP_PARAM)

        # 生成输入张量
        input_data = np.random.uniform(low=4, high=7, size=shape).astype(np.float32)
        input_tensor = torch.from_numpy(input_data).to(dataType)

        # 生成输出张量列表
        output_tensors = []
        for size in splitSize:
            slice_shape = list(shape)
            slice_shape[splitVDim] = size
            output_tensors.append(torch.zeros(slice_shape, dtype=dataType))

        self.execute([input_tensor], output_tensors)

    @op_test.only_910b
    def test_SplitV_tensor_values_range_910b(self):
        """测试用例一：2维3维张量值域遍历910b"""
        data_type_operation = [torch.bfloat16, torch.half, torch.int64]
        split_nums = [2, 3]
        dims_count = 5
        count = 80
        count = count // dims_count // len(data_type_operation) // len(split_nums)

        for _ in range(count):
            dim0 = random.choice(generalize)
            dim1 = random.choice(generalize)
            dim2 = random.choice(generalize)
            shape_operations = [
                (dim0, dim1),
                (dim0, dim1, dim2)
            ]
            for shape in shape_operations:
                for split_v_dim in range(len(shape)):
                    for data_type in data_type_operation:
                        split_num = 2 if data_type == torch.int64 else random.choice(split_nums)
                        config = {
                            "test_type": "generalize_random",
                            "shape" : shape,
                            "splitVDim": split_v_dim,
                            "dataType": data_type
                        }
                        with self.subTest(**config):
                            self.run_test_case(shape, split_v_dim, data_type, split_num)

    @op_test.only_910b
    def test_SplitV_generalized_16B_alignment_910b(self):
        """测试用例二：2维3维张量泛化16B对齐910b"""
        data_type_operation = [torch.bfloat16, torch.half, torch.int64]
        base_dim = 64
        split_nums = [2, 3]
        dims_count = 5
        count = 30
        count = count // dims_count // len(data_type_operation) // len(split_nums)

        variations_16 = self.generate_dimension_variations(base_dim, 16, count)
        for dim_variation in variations_16:
            dim0 = random.choice(generalize)
            dim1 = random.choice(generalize)
            dim2 = random.choice(generalize)
            shape_operations = [
                (dim_variation, dim1),
                (dim0, dim_variation),
                (dim_variation, dim1, dim2),
                (dim0, dim_variation, dim2),
                (dim0, dim1, dim_variation)
            ]
            for shape in shape_operations:
                split_v_dim = random.choice(range(len(shape)))
                for data_type in data_type_operation:
                    split_num = 2 if data_type == torch.int64 else random.choice(split_nums)
                    config = {
                        "test_type": "16B_aligment",
                        "shape" : shape,
                        "splitVDim": split_v_dim,
                        "dataType": data_type
                    }
                    with self.subTest(**config):
                        self.run_test_case(shape, split_v_dim, data_type, split_num)

    @op_test.only_910b
    def test_SplitV_generalized_16B_misalignment_910b(self):
        """测试用例三：2维3维张量泛化非16B对齐910b"""
        data_type_operation = [torch.bfloat16, torch.half, torch.int64]
        base_dim = 64
        split_nums = [2, 3]
        dims_count = 5
        count = 30
        count = count // dims_count // len(data_type_operation) // len(split_nums)

        variations_15 = self.generate_dimension_variations(base_dim, 15, count)
        for dim_variation in variations_15:
            dim0 = random.choice(generalize)
            dim1 = random.choice(generalize)
            dim2 = random.choice(generalize)
            shape_operations = [
                (dim_variation, dim1),
                (dim0, dim_variation),
                (dim_variation, dim1, dim2),
                (dim0, dim_variation, dim2),
                (dim0, dim1, dim_variation)
            ]
            for shape in shape_operations:
                split_v_dim = random.choice(range(len(shape)))
                for data_type in data_type_operation:
                    split_num = 2 if data_type == torch.int64 else random.choice(split_nums)
                    config = {
                        "test_type": "16B_misaligment",
                        "shape" : shape,
                        "splitVDim": split_v_dim,
                        "dataType": data_type
                    }
                    with self.subTest(**config):
                        self.run_test_case(shape, split_v_dim, data_type, split_num)

    @op_test.only_310p
    def test_SplitV_generalized_16B_alignment_310p(self):
        """测试用例二：2维3维张量泛化16B对齐310p"""
        data_type_operation = [torch.half]
        base_dim = 64
        split_nums = [2, 3]
        dims_count = 5
        count = 30
        count = count // dims_count // len(data_type_operation) // len(split_nums)

        variations_16 = self.generate_dimension_variations(base_dim, 16, count)
        for dim_variation in variations_16:
            dim0 = random.choice(generalize)
            dim1 = random.choice(generalize)
            dim2 = random.choice(generalize)
            shape_operations = [
                (dim_variation, dim1),
                (dim0, dim_variation),
                (dim_variation, dim1, dim2),
                (dim0, dim_variation, dim2),
                (dim0, dim1, dim_variation)
            ]
            for shape in shape_operations:
                split_v_dim = random.choice(range(len(shape)))
                for data_type in data_type_operation:
                    for split_num in split_nums:
                        config = {
                            "test_type": "16B_aligment",
                            "shape" : shape,
                            "splitVDim": split_v_dim,
                            "dataType": data_type
                        }
                        with self.subTest(**config):
                            self.run_test_case(shape, split_v_dim, data_type, split_num)

    @op_test.only_310p
    def test_SplitV_generalized_16B_misalignment_310p(self):
        """测试用例三：2维3维张量泛化非16B对齐310p"""
        data_type_operation = [torch.half]
        base_dim = 64
        split_nums = [2, 3]
        dims_count = 5
        count = 30
        count = count // dims_count // len(data_type_operation) // len(split_nums)

        variations_15 = self.generate_dimension_variations(base_dim, 15, count)
        for dim_variation in variations_15:
            dim0 = random.choice(generalize)
            dim1 = random.choice(generalize)
            dim2 = random.choice(generalize)
            shape_operations = [
                (dim_variation, dim1),
                (dim0, dim_variation),
                (dim_variation, dim1, dim2),
                (dim0, dim_variation, dim2),
                (dim0, dim1, dim_variation)
            ]
            for shape in shape_operations:
                split_v_dim = random.choice(range(len(shape)))
                for data_type in data_type_operation:
                    for split_num in split_nums:
                        config = {
                            "test_type": "16B_misaligment",
                            "shape" : shape,
                            "splitVDim": split_v_dim,
                            "dataType": data_type
                        }
                        with self.subTest(**config):
                            self.run_test_case(shape, split_v_dim, data_type, split_num)

if __name__ == '__main__':
    unittest.main()