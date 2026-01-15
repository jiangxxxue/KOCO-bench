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
constexpr float HALF_FLOAT_MIN = 6.2E-5 ;
constexpr float HALF_FLOAT_MAX = 65504 / 2;

Status MulsF32Compare(float scale, float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kFloat);
    at::Tensor atOutRefTensor = atInRefTensor.mul(scale);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atOutRefArray = (float *)atOutRefTensor.storage().data_ptr().get();
    for (int64_t i = 0; i < outTensor.Numel(); i++) {
        float expect = atOutRefArray[i];
        float result = atOutArray[i];
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}

Status MulsF16Compare(float scale, float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    at::Tensor atInRefTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atInRefTensorFloat = atInRefTensor.to(at::kFloat);

    at::Tensor atOutRefTensor = atInRefTensorFloat.mul(scale);

    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensorFloat = atOutTensor.to(at::kFloat);

    float *atOutArray = (float *)atOutTensorFloat.storage().data_ptr().get();
    float *atOutRefArray = (float *)atOutRefTensor.storage().data_ptr().get();

    for (int64_t i = 0; i < outTensor.Numel(); i++) {
        float expect = atOutRefArray[i];
        float result = atOutArray[i];
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect, result, atol, rtol)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}
}

TEST(TestOpElewiseMuls, TestMulsF32)
{
    Mki::Test::MkiOpTest opTest;
    float scale = -0.5f;
    opTest.Golden(std::bind(&MulsF32Compare, scale, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_MULS};
    opParam.varAttr = scale;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 32, 12, 12}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {12, 1, 4096}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {12, 1}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {12, 1, 32, 64}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {12, 1, 1, 64}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 1000000}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseMuls, TestMulsF16)
{
    Mki::Test::MkiOpTest opTest;
    float scale = -1.5f;
    opTest.Golden(std::bind(&MulsF16Compare, scale, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_MULS};
    opParam.varAttr = scale;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};

    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32, 12, 12}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 4096}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 32, 64}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 1, 1, 64}});
    ASSERT_EQ(status.Ok(), true);

    status = opTest.Run(opDesc, {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 1000000}});
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpElewiseMuls, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(TestOpElewiseMuls, TestCanSupport1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_ADD;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpElewiseMuls, TestCanSupport2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpElewiseMuls, TestCanSupport3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpElewiseMuls, TestCanSupport4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpElewiseMuls, TestCanSupport5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MulsF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpElewiseMul, TestGetBestKernelMuls0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpElewiseMul, TestGetBestKernelMuls1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpElewiseMul, TestGetBestKernelMuls2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_MULS;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}