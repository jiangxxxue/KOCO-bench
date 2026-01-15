/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <securec.h>
#include <vector>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;

static Status GreaterGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor1;
    at::Tensor atInRefTensor2;
    if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_INT64) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kLong);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kLong);
    }

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kByte);
    at::Tensor refOutTensor = torch::greater(atInRefTensor1, atInRefTensor2);
    int8_t *atOutArray = (int8_t *)atOutTensor.storage().data_ptr().get();
    int8_t *atRefOutArray = (int8_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        int8_t expect = atRefOutArray[i];
        int8_t actual = atOutArray[i];
        if (expect != actual) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseGreater, GreaterCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseGreater, GreaterCase1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, 1, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseGreater, GreaterCase2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseGreater, GreaterF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseGreater, GreaterF16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseGreater, GreaterF16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseGreater, GreaterF32Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseGreater, GreaterF32Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, 1, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseGreater, GreaterF32Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MAX);
    opTest.Golden(&GreaterGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_GREATER};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 14, n = 5, k = 23, b = 12;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 1, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater0)
{
    LaunchParam launchParam;
    int64_t m = 14, n = 5, k = 23, b = 12;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater1)
{
    LaunchParam launchParam;
    int64_t m = 14, n = 5, k = 23, b = 12;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_ADD;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater2)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater3)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater4)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestOpElewiseGreater, TestCanSupportGreater5)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("GreaterInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseGreater, TestGetBestKernelGreater0)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {14, 5, 23, 12}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpElewiseGreater, TestGetBestKernelGreater1)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {14, 5, 23, 12}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpElewiseGreater, TestGetBestKernelGreater2)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {14, 5, 23, 12}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief wrong output dtype
 */
TEST(TestOpElewiseGreater, TestGetBestKernelGreater3)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {14, 5, 23, 12}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief wrong input dtype
 */
TEST(TestOpElewiseGreater, TestGetBestKernelGreater4)
{
    int64_t m = 14, n = 5, k = 23, b = 12;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1, 1, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {14, 5, 23, 12}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_GREATER;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

