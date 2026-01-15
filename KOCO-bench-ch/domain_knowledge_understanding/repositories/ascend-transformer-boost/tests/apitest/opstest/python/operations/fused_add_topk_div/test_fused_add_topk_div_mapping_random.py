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
import torch
import torch_npu
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "FusedAddTopkDivOperation"
PARAM = {"groupNum":8,"groupTopk":4,"n":2,"k":8,"activationType":8,"isNorm":True,"scale":1.0,"enableExpertMapping":True}

OFFSET = 7
PRIME = 100000007
MAX_INT32 = 2147483647

LOOP_TIMES = 500

SUPPORT_DTYPE = [torch.float16]
MAX_REDUNDANT_EXPERT_NUM = 10

count_mapping = []
for i in range(MAX_REDUNDANT_EXPERT_NUM + 1):
    count_mapping.append(torch.zeros(i).to(torch.int32))

class TestFusedAddTopkDivOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        group_num = PARAM['groupNum']
        group_topk = PARAM['groupTopk']
        is_norm = PARAM['isNorm']
        n = PARAM['n']
        k = PARAM['k']
        scale = PARAM['scale']
        x = in_tensors[0].cpu().to(torch.float32)
        add_num = in_tensors[1].cpu().to(torch.float32)
        m = torch.nn.Sigmoid()
        output_sig = m(x)
        # add
        input0 = torch.add(output_sig, add_num)
        # group_topk
        token_num, expert_num = input0.shape
        group_eles = expert_num // group_num
        input0 = torch.reshape(input0, (token_num, group_num, group_eles))
        output = input0.clone()
        group_tensor = torch.topk(input0, n).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable')) 
        cols_to_use = torch.arange(group_topk, group_num, dtype=torch.long) 
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = float(0)
        group_top_k_res = torch.reshape(output, (token_num, expert_num))
        # topk
        sort_output = torch.sort(group_top_k_res, descending=True, stable=True)
        # gather
        gather_res = torch.gather(output_sig, -1, sort_output.indices[:, 0:k])
        if is_norm:
            # reduce_sum
            sum_res = torch.sum(gather_res, -1, keepdim=True)
            # div
            y = torch.div(gather_res, sum_res)
            # mul
            y = y * torch.tensor(scale, dtype=torch.float32)
        else:
            y = gather_res

        out_indices = sort_output.indices.to(torch.int32)
        enableExpertMapping = PARAM['enableExpertMapping']
        if enableExpertMapping is True:
            offset = OFFSET
            prime = PRIME
            mapping_num = in_tensors[2].cpu()
            mapping_table = in_tensors[3].cpu()
            out_indices_clone = out_indices.clone().detach()
            for bs in range(token_num):
                indices_offset = torch.tensor(sort_output.indices[bs][group_topk * group_eles - 1] + offset, dtype=torch.int32)
                rand_value =  torch.remainder(prime, indices_offset) / indices_offset.to(torch.float32)
                mapping_indices = torch.floor(mapping_num.to(torch.float32) * rand_value).to(torch.int32)
                count_mapping[mapping_table.shape[1]][mapping_indices[0]] = count_mapping[mapping_table.shape[1]][mapping_indices[0]] + 1
                for ki in range(k):
                    expert_id = out_indices_clone[bs][ki]
                    out_indices[bs][ki] = mapping_table[expert_id][mapping_indices[expert_id]]

        return y.npu(), out_indices[:, 0:k].npu()

    def golden_compare(self, out_tensor, golden_out_tensor):
        return True

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        for max_buffer in range(2, MAX_REDUNDANT_EXPERT_NUM + 1):
            for loop in range(LOOP_TIMES):
                for dtype in SUPPORT_DTYPE:
                    a = 16
                    b = 256
                    input_x_golden = torch.from_numpy(np.random.uniform(1, 10, [a, b])).to(dtype)
                    input_add_num_golden = torch.from_numpy(np.random.uniform(1, 10, b)).to(dtype)
                    mapping_num = torch.ones(b, dtype=torch.int32) * max_buffer
                    mapping_table = torch.randint(0, b, [b, max_buffer], dtype=torch.int32)
                    for i in range(b):
                        for e in range(mapping_num[i], max_buffer):
                            mapping_table[i][e] = MAX_INT32
                                        
                self.execute(OP_NAME, PARAM, [input_x_golden.npu(), input_add_num_golden.npu(), mapping_num.npu(), mapping_table.npu()])
        
        logging.info('>>>>>>>>>>>>>>>>> count_mapping <<<<<<<<<<<<<<<<<')
        logging.info(count_mapping[max_buffer].tolist())
        
if __name__ == '__main__':
    unittest.main()
