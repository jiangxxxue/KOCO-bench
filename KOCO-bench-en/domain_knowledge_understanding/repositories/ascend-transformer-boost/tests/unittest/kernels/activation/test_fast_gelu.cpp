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
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cmath>
#include "test_common.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = std::pow(65504, 1.0 / 3);
constexpr float HALF_FLOAT_MIN_NEG = -std::pow(65504, 1.0 / 3);
constexpr float HALF_FLOAT_MAX_NEG = -6.2E-5;

fp16_t FastGelu(fp16_t x)
{
    fp16_t a = static_cast<fp16_t>(0.851);
    fp16_t b = static_cast<fp16_t>(-1.702);
    fp16_t fp16zero = static_cast<fp16_t>(0);
    fp16_t absx = x >= fp16zero ? x : (fp16zero - x);
    float exp1 = a * (x - absx);
    float exp2 = b * absx;

    return x * static_cast<fp16_t>(exp(exp1)) / (static_cast<fp16_t>(1) + static_cast<fp16_t>(exp(exp2)));
}

Status FastGeluF16Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    fp16_t* inputs = static_cast<fp16_t*>(inTensor.data);
    fp16_t* outputs = static_cast<fp16_t*>(outTensor.data);

    std::vector<fp16_t> outRef(outTensor.Numel());
    for (int i = 0; i < outTensor.Numel(); ++i) {
        outRef[i] = FastGelu(inputs[i]);
    }

    for (int64_t i = 0; i < outTensor.Numel(); i++) {
        fp16_t expect = outRef[i];
        fp16_t result = outputs[i];
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(static_cast<float>(expect), static_cast<float>(result), atol, rtol)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}
}

TEST(TestOpActivationFastGelu, TestFastGeluF16)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&FastGeluF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Activation opParam = {OpParam::Activation::ACTIVATION_FAST_GELU};
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {100000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 16384}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpActivationFastGelu, TestFastGeluF16Neg)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&FastGeluF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN_NEG, HALF_FLOAT_MAX_NEG);
    OpParam::Activation opParam = {OpParam::Activation::ACTIVATION_FAST_GELU};
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {100000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 16384}});
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpActivationFastGelu, TestCanSupport0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpActivationFastGelu, TestCanSupport1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpActivationFastGelu, TestCanSupport2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpActivationFastGelu, TestCanSupport3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationFastGelu, TestCanSupport4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationFastGelu, TestCanSupport5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FastGeluF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpActivationFastGelu, TestGetBestKernelFastGelu0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationFastGelu, TestGetBestKernelFastGelu1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_FAST_GELU;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}