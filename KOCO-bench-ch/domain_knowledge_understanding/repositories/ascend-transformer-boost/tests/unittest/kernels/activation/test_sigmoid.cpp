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
#include <future>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "test_common.h"
#include "asdops/params/params.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = log(65504 / 5);
constexpr float EXTENT_OF_ERROR = 0.001;
constexpr size_t DATA_BATCH = 100000;

template <typename T> T Sigmoid(T x)
{
    float calcX = static_cast<float>(x);
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + static_cast<T>(exp(-calcX)));
}

void GetGroundtruthParallel(std::vector<fp16_t> &input, std::vector<fp16_t> &outputRef)
{
    size_t n = input.size();
    size_t cycCount = n % DATA_BATCH == 0 ? n / DATA_BATCH : n / DATA_BATCH + 1;
    std::vector<std::future<void>> results;
    for (size_t i = 0; i < cycCount; ++i) {
        results.push_back(std::async(std::launch::async, [i, &input, &outputRef]() {
            for (size_t j = 0; j < DATA_BATCH; ++j) {
                if (i * DATA_BATCH + j >= outputRef.size()) {
                    break;
                }
                outputRef[i * DATA_BATCH + j] = Sigmoid(input[i * DATA_BATCH + j]);
            }
        }));
    }

    for (auto &future : results) {
        future.get();
    }
}

Status SigmoidF16Compare(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    // golden
    std::vector<fp16_t> input(reinterpret_cast<fp16_t *>(inTensor.data),
                                        reinterpret_cast<fp16_t *>(inTensor.data) + inTensor.Numel());
    std::vector<fp16_t> outputRef(inTensor.Numel());
    GetGroundtruthParallel(input, outputRef);

    // compare at::Tensor
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensorRef = at::from_blob(outputRef.data(), ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    if (!at::allclose(atOutTensor, atOutTensorRef, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
} // namespace

/**
 * @brief accuracy compare ok
 */
TEST(TestOpActivationSigmoid, TestSigmoidF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SigmoidF16Compare, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Activation opParam = {OpParam::Activation::ACTIVATION_SIGMOID};
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {100000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 16384}});
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpActivationSigmoid, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpActivationSigmoid, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_RELU;  // wrong activation type
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief inPutNum wrong
 */
TEST(TestOpActivationSigmoid, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
/**
 * @brief outPutNum wrong
 */
TEST(TestOpActivationSigmoid, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationSigmoid, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(TestOpActivationSigmoid, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SigmoidF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpActivationSigmoid, TestGetBestKernelSigmoid0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpActivationSigmoid, TestGetBestKernelSigmoid1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam = {};
    opParam.activationType = OpParam::Activation::ACTIVATION_SIGMOID;
    Mki::Test::UtOpDesc opDesc = {"ActivationOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}