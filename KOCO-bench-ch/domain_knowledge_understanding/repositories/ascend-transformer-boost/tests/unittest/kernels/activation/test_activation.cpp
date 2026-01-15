/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <future>

#include <mki/utils/log/log.h>
#include "asdops/params/activation.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

TEST(TestOpActivationRelu, ReluF32Rand)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&Mki::Test::Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    OpParam::Activation opParam = {OpParam::Activation::ACTIVATION_RELU};
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}}, "ReluF32Kernel");
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}}, "ReluF32Kernel");
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpActivationRelu, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpActivationRelu, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SWISH;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpActivationRelu, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpActivationRelu, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationRelu, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestOpActivationRelu, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReluF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpActivationRelu, TestGetBestKernelRelu0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationRelu, TestGetBestKernelRelu1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @multi-thread auto tiling test
 */
TEST(TestOpActivationRelu, TestActivationReluF32RandCaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&Mki::Test::Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    OpParam::Activation opParam = {OpParam::Activation::ACTIVATION_RELU};
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(&Mki::Test::Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc1 = {"ActivationOperation", opParam};

    Mki::Test::MkiOpTest opTest2;
    opTest1.Golden(std::bind(&Mki::Test::Golden::InOutTensorEqual, ATOL, RTOL, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc2 = {"ActivationOperation", opParam};

    auto test1 = [&]() {
        return opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}});
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}}, "ReluF32Kernel");
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
