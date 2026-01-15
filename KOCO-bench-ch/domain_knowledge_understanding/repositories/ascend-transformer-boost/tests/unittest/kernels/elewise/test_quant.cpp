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
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/elewise.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "device_version_check.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
static constexpr float HALF_FLOAT_MIN = -1.0;
static constexpr float HALF_FLOAT_MAX = 1.0;
static constexpr float ATOL = 0.001;
static constexpr float RTOL = 0.001;
static constexpr float SCALE = 80.22;
static constexpr int OFFSET = -36;

namespace AsdOps {
Status PLNMatchAtTensorInt(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    int64_t countRight = 0;
    int64_t countFalse = 0;
    int8_t *result0 = static_cast<int8_t *>(out.storage().data_ptr().get());
    fp16_t *expect0 = static_cast<fp16_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        int res1 = result0[i];
        int res2 = std::round(static_cast<float>(expect0[i]));
        if (res1 != res2) {
            if (abs(abs(res1) - abs(res2)) == 1) {
                countFalse = countFalse + 1;
            }
        } else {
            countRight = countRight + 1;
        }
    }
    MKI_LOG(INFO) << out.numel() << "\t" << countRight << "\t" << countFalse;
    if (!Mki::Test::FloatUtil::FloatJudgeEqual(float(out.numel()), float(countRight + countFalse), atol, rtol)) {
        std::string msg =
            "expect: " + std::to_string(out.numel()) + ", result: " + std::to_string(countRight + countFalse);
        return Status::FailStatus(-1, msg);
    }
    return Status::OkStatus();
}
Status QuantGolden(float atol, float rtol, OpParam::Elewise opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inputX = context.hostInTensors.at(0);
    const Tensor &tensorOut = context.hostOutTensors.at(0);
    std::vector<int64_t> shape = {inputX.desc.dims[0], inputX.desc.dims[1], inputX.desc.dims[2]};
    auto attensorInputX = at::from_blob(inputX.data, shape, at::kHalf);

    auto res1 = at::mul(attensorInputX, SCALE);
    auto groundtruth = at::add(res1, OFFSET).to(at::kHalf);

    auto attensorOut = at::from_blob(tensorOut.data, shape, at::kChar);

    return PLNMatchAtTensorInt(atol, rtol, attensorOut, groundtruth);
}
} // namespace AsdOps

TEST(QuantF16Quant, QuantF16Quantop1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 6656;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.inputScale = SCALE;
    opParam.inputOffset = OFFSET;

    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_QUANT;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(QuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(QuantF16Quant, QuantF16Quantop2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 12288;
    uint32_t b = 4096;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.inputScale = SCALE;
    opParam.inputOffset = OFFSET;

    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_QUANT;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(QuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(QuantF16Quant, QuantF16Quantop3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 32;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.inputScale = SCALE;
    opParam.inputOffset = OFFSET;

    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_QUANT;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(QuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(QuantF16Quant, QuantF16Quantop4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 64;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.inputScale = SCALE;
    opParam.inputOffset = OFFSET;

    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_QUANT;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(QuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(QuantF16Quant, QuantF16Quantop5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 128;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.inputScale = SCALE;
    opParam.inputOffset = OFFSET;

    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_QUANT;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(QuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(QuantF16Quant, TestCanSupport0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief elewiseType wrong
 */
TEST(QuantF16Quant, TestCanSupport1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_NEG;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor format wrong
 */
TEST(QuantF16Quant, TestCanSupport2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor format wrong
 */
TEST(QuantF16Quant, TestCanSupport3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_HWCN, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inPutNum wrong
 */
TEST(QuantF16Quant, TestCanSupport4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(QuantF16Quant, TestCanSupport5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(QuantF16Quant, TestCanSupport6)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outTensor dtype wrong
 */
TEST(QuantF16Quant, TestCanSupport7)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("QuantF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(QuantF16Quant, TestInferShapeQuant0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

TEST(QuantF16Quant, TestInferShapeQuant1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(QuantF16Quant, TestInferShapeQuant2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(QuantF16Quant, TestInferShapeQuant3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

TEST(QuantF16Quant, TestInferShapeQuant4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_UNDEFINED;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(QuantF16Quant, TestGetBestKernelQuant0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_QUANT;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}