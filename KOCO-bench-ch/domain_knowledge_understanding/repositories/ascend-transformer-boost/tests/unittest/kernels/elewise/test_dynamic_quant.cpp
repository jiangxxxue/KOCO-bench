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
#include "device_version_check.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
static constexpr float HALF_FLOAT_MIN = -1;
static constexpr float HALF_FLOAT_MAX = 1;
static constexpr float ATOL = 0.001;
static constexpr float RTOL = 0.001;
static constexpr float QUANT_MAX_VALUE = 127;
static constexpr float QUANT_ASYMMETRIC_VALUE = 255;

namespace AsdOps {
Status DQMatchAtTensorFloat(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    float *outArr = (float *)out.storage().data_ptr().get();
    float *gtArr = (float *)gt.storage().data_ptr().get();
    for (int i = 0; i < out.numel(); i++) {
        float expect = gtArr[i];
        float actual = outArr[i];
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect, actual, ATOL, RTOL)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}

Status DQMatchAtTensorInt(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    int64_t countRight = 0;
    int64_t countFalse = 0;
    int8_t *result0 = static_cast<int8_t *>(out.storage().data_ptr().get());
    float *expect0 = static_cast<float *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        int actual = result0[i];
        int expect = std::round(expect0[i]);
        if (expect >= 128) {
            expect = 127;
        }
        if (expect != actual) {
            if (abs(abs(expect) - abs(actual)) == 1) {
                countFalse = countFalse + 1;
            }
        } else {
            countRight = countRight + 1;
        }
    }
    MKI_LOG(INFO) << "out.numel(): " << out.numel() << "\tcountRight: " << countRight \
        << "\tcountFalse: " << countFalse << "\tcorrectRate: " << (float)countRight / out.numel();
    if (!Mki::Test::FloatUtil::FloatJudgeEqual(float(out.numel()), float(countRight + countFalse), atol, rtol)) {
        std::string msg =
            "expect: " + std::to_string(out.numel()) + ", result: " + std::to_string(countRight + countFalse);
        return Status::FailStatus(-1, msg);
    }
    return Status::OkStatus();
}

Status DynamicQuantGolden(float atol, float rtol, OpParam::Elewise opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inputX = context.hostInTensors.at(0);
    const Tensor &tensorOut = context.hostOutTensors.at(0);
    const Tensor &tensorOutScale = context.hostOutTensors.at(1);
    const Tensor &tensorOutOffset = context.hostOutTensors.at(2);
    std::vector<int64_t> shape = {inputX.desc.dims[0], inputX.desc.dims[1], inputX.desc.dims[2]};
    std::vector<int64_t> scaleShape = {inputX.desc.dims[0], inputX.desc.dims[1]};
    auto attensorOut0 = at::from_blob(tensorOut.data, shape, at::kChar);
    auto attensorOut1 = at::from_blob(tensorOutScale.data, scaleShape, at::kFloat);
    auto attensorOut2 = at::from_blob(tensorOutOffset.data, scaleShape, at::kFloat);
    auto attensorInputX = at::from_blob(inputX.data, shape, at::kHalf);
    Status status;
    int64_t dim = inputX.desc.dims.size() - 1; // unsqueeze the last dim
    if (opParam.asymmetric) {
        auto resMax = std::get<0>(at::max(attensorInputX, dim, true));
        auto resMin = std::get<0>(at::min(attensorInputX, dim, true));
        auto scale = at::sub(resMax, resMin);
        auto offset = at::add(resMax, resMin);
        offset = at::div(offset, scale).to(at::kFloat);
        scale = at::div(scale, QUANT_ASYMMETRIC_VALUE).to(at::kFloat);
        offset = at::mul(offset, -127.5f); // -127.5 is -255 / 2
        auto outputTensor = at::div(attensorInputX, scale).to(at::kFloat);
        outputTensor = at::add(outputTensor, offset);
        auto groundtruth1 = at::unsqueeze(scale, dim);
        auto groundtruth2 = at::unsqueeze(offset, dim);
        MKI_LOG(INFO) << "check offset value";
        status = DQMatchAtTensorFloat(atol, rtol, attensorOut2, groundtruth2);
        if (!status.Ok()) {
            return status;
        }
        MKI_LOG(INFO) << "check scale value";
        status = DQMatchAtTensorFloat(atol, rtol, attensorOut1, groundtruth1);
        if (!status.Ok()) {
            return status;
        }
        return DQMatchAtTensorInt(atol, rtol, attensorOut0, outputTensor);
    } else {
        auto resAbs = at::abs(attensorInputX);
        auto resMax = std::get<0>(at::max(resAbs, dim, true));
        auto outputTensor = at::div(attensorInputX, resMax).to(at::kFloat);
        outputTensor = at::mul(outputTensor, QUANT_MAX_VALUE);
        resMax = at::div(resMax, QUANT_MAX_VALUE).to(at::kFloat);
        auto groundtruth1 = at::unsqueeze(resMax, dim);
        MKI_LOG(INFO) << "check scale value";
        status = DQMatchAtTensorFloat(atol, rtol, attensorOut1, groundtruth1);
        if (!status.Ok()) {
            return status;
        }
        return DQMatchAtTensorInt(atol, rtol, attensorOut0, outputTensor);
    }
}
} // namespace AsdOps

TEST(DynamicQuantFP16, DynamicQuantFP16Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 70;
    uint32_t c = 32;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 64;
    uint32_t b = 1;
    uint32_t c = 32;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 1;
    uint32_t c = 32;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 1;
    uint32_t c = 64;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 1;
    uint32_t c = 128;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case6)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 6144;
    uint32_t b = 1;
    uint32_t c = 4096;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case7)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 123;
    uint32_t b = 1;
    uint32_t c = 4096;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case8)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 50;
    uint32_t c = 50;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case9)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 1;
    uint32_t c = 100;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case10)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 1;
    uint32_t c = 300;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case11)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 123;
    uint32_t b = 1;
    uint32_t c = 1000;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case12)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 5000;
    uint32_t b = 1;
    uint32_t c = 4100;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = true;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(DynamicQuantFP16, DynamicQuantFP16Case13)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 5000;
    uint32_t b = 1;
    uint32_t c = 4100;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Elewise opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "ElewiseOperation";
    opParam.elewiseType = AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT;
    opParam.asymmetric = false;
    opDesc.specificParam = opParam;

    opTest.Golden(std::bind(DynamicQuantGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, b, c}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}