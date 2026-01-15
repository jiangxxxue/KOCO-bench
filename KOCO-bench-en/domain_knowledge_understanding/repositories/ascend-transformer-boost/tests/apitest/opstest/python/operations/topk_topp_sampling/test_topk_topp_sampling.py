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
from ctypes import CDLL
import numpy as np
import torch
import torch_npu
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

rand_seed = 0
OP_NAME = "TopkToppSamplingOperation"
libc = CDLL("libc.so.6")
libc.srand(rand_seed)
rand_list = [libc.rand() / 0x7fffffff for i in range(512)]


class TestToppOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        topkToppSamplingType = self.op_param["topkToppSamplingType"]
        if topkToppSamplingType == 0:
            probs = in_tensors[0].cpu().numpy()
            topp = in_tensors[1].cpu().numpy()
            topk = self.op_param["topk"]
            probs_sorted = np.sort(probs, axis=-1)[..., ::-1][..., :topk]
            indices_sorted = np.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
            # 转npu计算以提高精度
            probs_sorted_sumed = torch.cumsum(torch.from_numpy(probs_sorted.copy()).npu().to(in_tensors[0].dtype), dim=-1).cpu().to(torch.float32).numpy()
            bool_judge = (probs_sorted_sumed < topp)
            sum_val = np.sum(bool_judge, axis=-1, keepdims=True) - 1
            sum_val[sum_val < 0] = 0
            topp_v = np.take_along_axis(probs_sorted_sumed, sum_val, axis=-1)
            topp_v *= np.array(rand_list).reshape(-1, 1)[0:probs.shape[0]]
            bool_judge_one = probs_sorted_sumed <= topp_v
            res = np.sum(bool_judge_one, axis=-1, keepdims=True)
            res[res < 0] = 0
            indices_sampled = np.take_along_axis(indices_sorted, res, axis=-1)
            probs_sampled = np.take_along_axis(probs_sorted, res, axis=-1)
            return [torch.from_numpy(indices_sampled.astype(np.int32)),
                    torch.from_numpy(probs_sampled.astype(np.float16))]
        else:
            probs = in_tensors[0].cpu().to(torch.float32)
            topk = in_tensors[1].cpu().to(torch.int64)
            topp = in_tensors[2].cpu().to(torch.float32)
            probs_sorted, idx_sorted = torch.sort(probs, descending=True, stable=True)
            gather_topk = torch.gather(probs_sorted, dim=1, index=topk)
            topk_mask = torch.lt(probs_sorted, gather_topk).to(torch.bool)
            probs_masked_topk = probs_sorted.masked_fill(topk_mask, 0)
            probs_masked_topk = probs_masked_topk.numpy()
            probs_cumsumed = np.cumsum(probs_masked_topk, axis=-1, dtype=np.float16).astype(np.float32)
            probs_cumsumed = torch.from_numpy(probs_cumsumed)
            probs_masked_topk = torch.from_numpy(probs_masked_topk)
            if topkToppSamplingType == 2 or topkToppSamplingType == 4:
                exp = in_tensors[3].cpu()
                topp_mask = torch.gt(probs_cumsumed, topp).to(torch.bool)
                probs_masked_topp = probs_masked_topk.masked_fill(topp_mask, 0)
                divided_probs = torch.div(probs_masked_topp.to(exp.dtype), exp)
                argmax_probs, argmax_idx = torch.sort(divided_probs, descending=True, stable=True)
                argmax_probs = argmax_probs[..., :1]
                argmax_idx = argmax_idx[..., :1]
                outtensor_probs = torch.gather(probs_sorted, dim=1, index=argmax_idx)
                outtensor_idx = torch.gather(idx_sorted, dim=1, index=argmax_idx)
                if topkToppSamplingType == 4:
                    logProbsSize = self.op_param["logProbsSize"]
                    mask_tensor = torch.logical_or(topp_mask, topk_mask)
                    mask_tensor[:, 0] = 0
                    sum_val = np.sum(~mask_tensor.numpy(), axis=-1, keepdims=True) - 1
                    sum_val[sum_val < 0] = 0
                    topp_v = np.take_along_axis(probs_cumsumed.numpy().astype(np.float32), sum_val, axis=-1)
                    logprobs_output = probs_sorted[:, :logProbsSize]
                    logprobs_output = logprobs_output.div(torch.from_numpy(topp_v)).log()
                    logprobs_output = logprobs_output.masked_fill(mask=mask_tensor[:, :logProbsSize], value=-9999.0)
                    return [outtensor_idx.to(torch.int32), outtensor_probs.to(torch.float16), logprobs_output.to(torch.float32)]

                return [outtensor_idx.to(torch.int32), outtensor_probs.to(torch.float16)]
            if topkToppSamplingType == 1 or topkToppSamplingType == 3:
                topp = topp.numpy().astype(np.float32)
                probs_cumsumed = probs_cumsumed.numpy().astype(np.float32)
                bool_judge = (probs_cumsumed < topp)
                sum_val = np.sum(bool_judge, axis=-1, keepdims=True) - 1
                sum_val[sum_val < 0] = 0
                topp_v = np.take_along_axis(probs_cumsumed, sum_val, axis=-1)
                randnp_new = np.zeros(probs_cumsumed.shape[0])
                topp_v_new = None
                for i in range(probs_cumsumed.shape[0]):
                    if topkToppSamplingType == 1:
                        randSeeds = self.op_param["randSeeds"]
                        libc.srand(randSeeds[i])
                        rand_num = libc.rand() / 0x7fffffff
                        randnp_new[i] = rand_num
                        randnp_new = randnp_new.reshape(-1, 1)
                        topp_v_new = randnp_new[0:probs_cumsumed.shape[0]] * topp_v
                    else:
                        topp_v_new = in_tensors[3].cpu().numpy().astype(np.float32) * topp_v
                bool_judge_one = (probs_cumsumed < topp_v_new)
                res = np.sum(bool_judge_one, axis=-1, keepdims=True)
                res[res < 0] = 0
                res_idx = torch.from_numpy(res).to(torch.int64)
                outtensor_probs = torch.gather(probs_sorted, dim=1, index=res_idx)
                outtensor_idx = torch.gather(idx_sorted, dim=1, index=res_idx)
                if topkToppSamplingType == 3:
                    bool_judge[:, 0] = 1
                    logProbsSize = self.op_param["logProbsSize"]
                    logprobs_output = probs_sorted[:, :logProbsSize]
                    logprobs_output = logprobs_output.div(torch.from_numpy(topp_v)).log()
                    logprobs_output = logprobs_output.masked_fill(mask=~torch.from_numpy(bool_judge[:, :logProbsSize]),
                                                                  value=-9999.0)
                    return [outtensor_idx.to(torch.int32), outtensor_probs.to(torch.float16),
                            logprobs_output.to(torch.float32)]
                else:
                    return [outtensor_idx.to(torch.int32), outtensor_probs.to(torch.float16)]

    def test_8_1024_top100(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.manual_seed(0)
        # probs = torch.randn(8, 1024).float()
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = torch.HalfTensor([[0.9]]).npu()
        # PARAM = dict()
        # PARAM["topk"] = 100
        # PARAM["randSeed"] = rand_seed
        # PARAM["topkToppSamplingType"] = 0
        # self.execute(OP_NAME, PARAM, [probs, topp])
        # PARAM = dict()
        # PARAM["topk"] = 200
        # PARAM["randSeed"] = rand_seed
        # PARAM["topkToppSamplingType"] = 0
        # self.execute_update_param(OP_NAME, PARAM, [probs, topp])

    def test_1_topk_batch(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.manual_seed(0)
        # probs = torch.randn(1, 100).float()
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = torch.HalfTensor([[0.2]]).npu()
        # PARAM = dict()
        # PARAM["topk"] = 1
        # PARAM["topkToppSamplingType"] = 0
        # self.execute(OP_NAME, PARAM, [probs, topp])

    def test_1_topk_batch_exponential(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # batch_size = 1
        # vocabulary_size = 100
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # exp_div = torch.empty_like(probs.cpu()).exponential_(1)
        # exp_div = exp_div.npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.tensor([[1]]).to(torch.int32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [0]
        # PARAM["topkToppSamplingType"] = 2
        # self.execute(OP_NAME, PARAM, [probs, topk, topp, exp_div])

    def test_512_1000_batch_multinomial(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.set_printoptions(threshold=50000)
        # batch_size = 512
        # vocabulary_size = 1000
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.randint(150, 150 + batch_size, size=(batch_size, 1), dtype=torch.int32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [i for i in range(batch_size)]
        # PARAM["topkToppSamplingType"] = 1
        # self.execute(OP_NAME, PARAM, [probs, topk, topp])

    def test_1_topk_batch_multinomial(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.set_printoptions(threshold=50000)
        # batch_size = 1
        # vocabulary_size = 1000
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.tensor([[1]]).to(torch.int32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [0]
        # PARAM["topkToppSamplingType"] = 1
        # self.execute(OP_NAME, PARAM, [probs, topk, topp])

    def test_64_1024_batch_log_probs_multinomial(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.set_printoptions(threshold=50000)
        # batch_size = 64
        # vocabulary_size = 1024
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.randint(150, 150 + batch_size, size=(batch_size, 1), dtype=torch.int32).npu()
        # rand = torch.rand(batch_size, 1, dtype=torch.float32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [i for i in range(batch_size)]
        # PARAM["topkToppSamplingType"] = 3
        # PARAM["logProbsSize"] = 500
        # self.execute(OP_NAME, PARAM, [probs, topk, topp, rand])

    def test_1_1024_log_probs_exponential(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # batch_size = 1
        # vocabulary_size = 512
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # exp_div = torch.empty_like(probs.cpu()).exponential_(1)
        # exp_div = exp_div.npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.tensor([[1]]).to(torch.int32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [0]
        # PARAM["topkToppSamplingType"] = 4
        # PARAM["logProbsSize"] = 512
        # self.execute(OP_NAME, PARAM, [probs, topk, topp, exp_div])
    
    def test_8_1024_top100_and_1_1024_log_probs_exponential(self):
        return True
        # if operation_test.get_soc_version() == 'Ascend910A' or \
        #         operation_test.get_soc_version() == 'Ascend310B':
        #     logging.info("this testcase don't supports Ascend910A\Ascend310B")
        #     return True
        # torch.manual_seed(0)
        # probs = torch.randn(8, 1024).float()
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # topp = torch.HalfTensor([[0.9]]).npu()
        # PARAM = dict()
        # PARAM["topk"] = 100
        # PARAM["randSeed"] = rand_seed
        # PARAM["topkToppSamplingType"] = 0
        # self.execute(OP_NAME, PARAM, [probs, topp])
        #
        # batch_size = 1
        # vocabulary_size = 512
        # torch.manual_seed(0)
        # probs = torch.randn(batch_size, vocabulary_size, dtype=torch.float32)
        # sm = nn.Softmax(dim=-1)
        # probs = sm(probs).half().npu()
        # exp_div = torch.empty_like(probs.cpu()).exponential_(1)
        # exp_div = exp_div.npu()
        # topp = ((0.9 - 0.2) * torch.rand(batch_size, 1, dtype=torch.float16) + 0.2).npu()
        # topk = torch.tensor([[1]]).to(torch.int32).npu()
        # PARAM = dict()
        # PARAM["randSeeds"] = [0]
        # PARAM["topkToppSamplingType"] = 4
        # PARAM["logProbsSize"] = 512
        # self.execute_update_param(OP_NAME, PARAM, [probs, topk, topp, exp_div])

if __name__ == '__main__':
    unittest.main()
