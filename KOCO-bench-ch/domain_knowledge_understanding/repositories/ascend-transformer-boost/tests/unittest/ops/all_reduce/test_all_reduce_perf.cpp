/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <thread>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <acl/acl.h>
#include <atb/utils/log.h>
#include <limits.h>
#include "test_utils/test_common.h"
#include "atb/utils/config.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "mki/utils/fp16/fp16_t.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

void RunAllReduceOp(int rank, int rankSize, const atb::TensorDesc &inTensorDesc, int *result)
{
    *(result) = -1;
    atb::infer::AllReduceParam param;
    param.rank = rank;
    param.rankSize = rankSize;
    param.backend = "lccl";
    param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
    atb::Operation *op = nullptr;
    atb::Status st = atb::CreateOperation(param, &op);
    if (st != atb::NO_ERROR || !op) {
        return;
    };
    OperationTest opTest;
    opTest.SetDeviceId(rank);
    opTest.IntRand(1, 1);
    opTest.FloatRand(1.0, 1.0);
    atb::Status status = opTest.RunPerf(op, { inTensorDesc });
    if (status != 0) {
        return;
    }
    *(result) = 0;
    DestroyOperation(op);
}

void TestAllReduceThread(const atb::TensorDesc &inTensorDesc, int rankSize)
{
    std::vector<std::thread> threads;
    int results[2] = {-1, -1};
    for (int i = 0; i < rankSize; ++i) {
        threads.push_back(std::thread(RunAllReduceOp, i, rankSize, inTensorDesc, results + i));
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();
    for (int i = 0; i < rankSize; ++i) {
        EXPECT_EQ(results[i], 0);
    }
}

TEST(TestAllReduceOperationPerf, HcclMulitThreadHalf)
{
    if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() ||
        GetSingleton<Config>().Is310B()) {
        return;
    }
    const int32_t n = 1;
    atb::TensorDesc inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { n , 1024 }, 2 } };
    int rankSize = 2;
    TestAllReduceThread(inTensorDesc, rankSize);

    inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { n , 1024 }, 2 }};
    TestAllReduceThread(inTensorDesc, rankSize);

    inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { n , 1024 }, 2}};
    TestAllReduceThread(inTensorDesc, rankSize);
}