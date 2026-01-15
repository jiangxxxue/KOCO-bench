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
#include <future>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr float HALF_FLOAT_MIN = 1;
constexpr float HALF_FLOAT_MAX = 30000;

Status ConcatGolden(OpParam::Concat opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    opParam.concatDim = opParam.concatDim < 0 ? (opParam.concatDim + static_cast<int>(inTensor1.desc.dims.size())) :
        opParam.concatDim;

    at::Tensor refOutTensor = torch::cat({atInRefTensor1, atInRefTensor2}, opParam.concatDim);
    fp16_t *atOutArray = (fp16_t *)atOutTensor.storage().data_ptr().get();
    fp16_t *atRefOutArray = (fp16_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        EXPECT_EQ(judge, true);
    }
    return Status::OkStatus();
}

TEST(TestOpConcat, concatTest1)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Concat opParam = {1};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    int64_t m = 32;
    int64_t n = 32;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpConcat, concatTest2)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Concat opParam = {1};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    int64_t m = 320;
    int64_t n = 500;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpConcat, concatTest3)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Concat opParam = {-1};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    int64_t m = 320;
    int64_t n = 500;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpConcat, concatTest4)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Concat opParam = {-2};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    int64_t m = 320;
    int64_t n = 500;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpConcat, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ConcatF16Input2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpConcat, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ConcatF16Input2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}


/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpConcat, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ConcatF16Input2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor concatDim wrong
 */
TEST(TestOpConcat, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {-3};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ConcatF16Input2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpConcat, TestInferShapeConcat0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpConcat, TestInferShapeConcat1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpConcat, TestInferShapeConcat2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief inTensor concatDim wrong
 */
TEST(TestOpConcat, TestInferShapeConcat3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Concat opParam = {-3};
    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpConcat, TestOpConcatCaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Concat opParam = {1};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));

    Mki::Test::MkiOpTest opTest1;
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest1.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));

    Mki::Test::MkiOpTest opTest2;
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest2.Golden(std::bind(ConcatGolden, opParam, std::placeholders::_1));

    Mki::Test::UtOpDesc opDesc = {"ConcatOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"ConcatOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"ConcatOperation", opParam};

    int64_t m = 32;
    int64_t n = 32;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    SVector<TensorDesc> inTensorDesc1 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    SVector<TensorDesc> inTensorDesc2 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};

    auto test1 = [&]() {
        return opTest.Run(opDesc, inTensorDesc);
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, inTensorDesc1);
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, inTensorDesc2);
    };

    for (uint32_t i = 0; i < 100; i++) {
        std::future<Status> result1 = std::async(test1);
        std::future<Status> result2 = std::async(test2);
        std::future<Status> result3 = std::async(test3);
        Status status1 = result1.get();
        Status status2 = result2.get();
        Status status3 = result3.get();
        ASSERT_EQ(status1.Ok() && status2.Ok() && status3.Ok() , true);
    }
}