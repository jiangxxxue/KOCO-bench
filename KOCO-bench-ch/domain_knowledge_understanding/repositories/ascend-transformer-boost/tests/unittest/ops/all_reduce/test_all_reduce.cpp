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

template <typename T> Mki::Status GoldenTwoRank(const OpsGoldenContext &context)
{
    // get constructed input/output tensors
    const Mki::Tensor inTensor = context.hostInTensors.at(0);
    const Mki::Tensor outTensor = context.hostOutTensors.at(0);
    at::ScalarType scalar = at::kFloat;
    if (std::is_same<T, Mki::fp16_t>::value) {
        scalar = at::kHalf;
    }
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), scalar);
    // construct ref input tensors
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), scalar).to(scalar);
    // get ref output tensor
    at::Tensor atOutRefTensor = (atInRefTensor * 2).to(scalar);

    T *atInArray = static_cast<T *>(atInRefTensor.storage().data_ptr().get());
    T *atOutArray = static_cast<T *>(atOutTensor.storage().data_ptr().get());
    T *atRefOutArray = static_cast<T *>(atOutRefTensor.storage().data_ptr().get());

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = atRefOutArray[i];
        float actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
        if (!judge) {
            return Mki::Status::FailStatus(1, "unequal");
        }
    }
    return Mki::Status::OkStatus();
}

Mki::Status GoldenTwoRankHalf(const OpsGoldenContext &context)
{
    return GoldenTwoRank<Mki::fp16_t>(context);
}

Mki::Status GoldenTwoRankFloat(const OpsGoldenContext &context)
{
    return GoldenTwoRank<float>(context);
}

Mki::Status GoldenTwoRankInt(const OpsGoldenContext &context)
{
    return GoldenTwoRank<int32_t>(context);
}

void RunAllReduceOp(int rank, int rankSize, const atb::TensorDesc &inTensorDesc, OpTestGolden golden, int *result)
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
    opTest.IntRand(2, 2);
    opTest.FloatRand(3.0, 3.0);
    opTest.Golden(golden);
    atb::Status status = opTest.Run(op, { inTensorDesc });
    if (status != 0) {
        return;
    }
    *(result) = 0;
    DestroyOperation(op);
}

void TestAllReduceThread(const atb::TensorDesc &inTensorDesc, int rankSize, OpTestGolden golden)
{
    std::vector<std::thread> threads;
    int results[2] = {-1, -1};
    for (int i = 0; i < rankSize; ++i) {
        threads.push_back(std::thread(RunAllReduceOp, i, rankSize, inTensorDesc, golden, results + i));
    }
    for (auto &t : threads) {
        t.join();
    }
    threads.clear();
    for (int i = 0; i < rankSize; ++i) {
        EXPECT_EQ(results[i], 0);
    }
}

TEST(TestAllReduceOperation, HcclMulitThreadHalf)
{
    if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() ||
        GetSingleton<Config>().Is310B()) {
        return;
    }
    atb::TensorDesc inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 2, 1024 }, 2 } };
    int rankSize = 2;
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankHalf);

    inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 32, 2752 }, 2 } };
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankHalf);

    inTensorDesc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 32, 8192 }, 2 } };
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankHalf);
}

TEST(TestAllReduceOperation, HcclMulitThreadFloat)
{
    if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() ||
        GetSingleton<Config>().Is310B()) {
        return;
    }
    atb::TensorDesc inTensorDesc = { ACL_FLOAT, ACL_FORMAT_ND, { { 64, 3072 }, 2 } };
    int rankSize = 2;
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankFloat);
}

TEST(TestAllReduceOperation, HcclMulitThreadInt)
{
    if (GetSingleton<Config>().Is310P() || GetSingleton<Config>().Is910A() ||
        GetSingleton<Config>().Is310B()) {
        return;
    }
    atb::TensorDesc inTensorDesc = { ACL_INT32, ACL_FORMAT_ND, { { 140, 8192 }, 2 } };
    int rankSize = 2;
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankInt);

    inTensorDesc = { ACL_INT32, ACL_FORMAT_ND, { { 1024, 140 }, 2 } };
    TestAllReduceThread(inTensorDesc, rankSize, &GoldenTwoRankInt);
}
