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
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/elewise.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = -3.14159265358979;
constexpr float HALF_FLOAT_MAX = 3.14159265358979;

float CosRef(float x)
{
    return 1.0 - std::pow(x, 2) / 2 + std::pow(x, 4) / 24 - std::pow(x, 6) / 720 + std::pow(x, 8) / 40320 -
           std::pow(x, 10) / 3628800;
}

Status CosF32Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    float *indata = static_cast<float *>(inTensor.data);
    float *outdata = static_cast<float *>(outTensor.data);

    for (size_t i = 0; i < inTensor.desc.Numel(); ++i) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(CosRef(indata[i]), outdata[i], atol, rtol)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}

Status CosF16Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    fp16_t *indata = static_cast<fp16_t *>(inTensor.data);
    fp16_t *outdata = static_cast<fp16_t *>(outTensor.data);

    for (size_t i = 0; i < inTensor.desc.Numel(); ++i) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(CosRef(static_cast<float>(indata[i])), static_cast<float>(outdata[i]), atol,
                                        rtol)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}
} // namespace

TEST(TestOpElewiseCos, TestCosF16)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&CosF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_COS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10000000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 64}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCos, TestCosF32)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&CosF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_COS};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10000000}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {12, 64}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseCos, TestGetBestKernel)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1000}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1000}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

TEST(TestOpElewiseCos, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseCos, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseCos, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_SIN;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport6)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport7)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpElewiseCos, TestCanSupport8)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {50}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {50}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_COS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("CosF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}