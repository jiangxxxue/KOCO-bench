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
#include <mki/tensor.h>
#include <mki/types.h>
#include <mki/utils/status/status.h>
#include "asdops/params/elewise.h"
#include "test_utils/op_test.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;

Status LessGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor1;
    at::Tensor atInRefTensor2;
    if (inTensor1.desc.dtype == TENSOR_DTYPE_INT64) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kLong);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kLong);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf);
    } else if (inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT) {
        atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kFloat);
        atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kFloat);
    }

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kChar);
    at::Tensor refOutTensor = atInRefTensor1.less(atInRefTensor2);
    int8_t *atOutArray = (int8_t *)atOutTensor.storage().data_ptr().get();
    int8_t *atRefOutArray = (int8_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        int8_t expect = atRefOutArray[i];
        int8_t actual = atOutArray[i];
        if (expect != actual) {
            return Status::FailStatus(-1, "less judge not equal");
        }
    }
    return Status::OkStatus();
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseLess, LessCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseLess, LessCase1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, 1, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseLess, LessCase2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief dim wrong
 */
TEST(TestOpElewiseLess, LessCase3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseLess, LessF32Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(FLT_MIN, FLT_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseLess, LessF32Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, 1, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseLess, LessF32Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief dim wrong
 */
TEST(TestOpElewiseLess, LessF32Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN, LONG_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseLess, LessF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief broadcast
 */
TEST(TestOpElewiseLess, LessF16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseLess, LessF16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief dim wrong
 */
TEST(TestOpElewiseLess, LessF16Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&LessGolden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_LESS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseLess, TestGetBestKernel0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpElewiseLess, TestGetBestKernel1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpElewiseLess, TestGetBestKernel2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief wrong input dtype
 */
TEST(TestOpElewiseLess, TestGetBestKernel3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief wrong output dtype
 */
TEST(TestOpElewiseLess, TestGetBestKernel4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief different input dtype
 */
TEST(TestOpElewiseLess, TestInferShape)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief base testcase
 */
TEST(TestOpElewiseLess, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseLess, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief wrong output dtype
 */
TEST(TestOpElewiseLess, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief wrong input num
 */
TEST(TestOpElewiseLess, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief wrong output num
 */
TEST(TestOpElewiseLess, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_LESS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief wrong elewisetype
 */
TEST(TestOpElewiseLess, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_ADD;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("LessInt64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}