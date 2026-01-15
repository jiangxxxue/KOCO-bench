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
#include <cmath>
#include <gtest/gtest.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/multinomial.h"
#include "kernels/multinomial/multinomial/tiling/multinomial_tiling.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_utils/float_util.h"
#include "device_version_check.h"
using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float FLOAT_MIN = 0;
constexpr float FLOAT_MAX = 1;

/**
 * @brief ok
 */
TEST(Testmultinomial, TestCanSupport0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Multinomial opParam;
    opParam.numSamples = 3;
    Mki::Test::UtOpDesc opDesc = {"MultinomialOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MultinomialKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(Testmultinomial, MultinomialTiling)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Multinomial opParam;
    opParam.numSamples = 3;
    Mki::Test::UtOpDesc opDesc = {"MultinomialOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
 
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    if (launchBufferSize == 0) {
        MKI_LOG(ERROR) << "empty tiling size";
    }
    KernelInfo &kernelInfo = const_cast<KernelInfo &>(kernel->GetKernelInfo());
    kernelInfo.AllocTilingHost(launchBufferSize);
    status = MultinomialTiling(launchParam, kernelInfo);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(Testmultinomial, TestGetBestKernelMultinomial)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Multinomial opParam;
    opParam.numSamples = 3;
    Mki::Test::UtOpDesc opDesc = {"MultinomialOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}
