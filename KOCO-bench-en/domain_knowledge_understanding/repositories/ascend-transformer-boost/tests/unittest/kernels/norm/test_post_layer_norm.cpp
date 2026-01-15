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
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/norm.h"
#include "kernels/norm/postlayernorm/tiling/post_layer_norm_tiling.h"
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
static constexpr float EPS = 0.00001;

namespace AsdOps {
Status PLNMatchAtTensorHalf(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    fp16_t *result = static_cast<fp16_t *>(out.storage().data_ptr().get());
    fp16_t *expect = static_cast<fp16_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect[i], result[i], atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) +
                              ", expect: " + std::to_string(static_cast<float>(expect[i])) +
                              ", result: " + std::to_string(static_cast<float>(result[i]));
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

Status PostLayerNormGolden(float atol, float rtol, OpParam::Norm opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inputX = context.hostInTensors.at(0);
    const Tensor &resIn = context.hostInTensors.at(1);
    const Tensor &gamma = context.hostInTensors.at(2);
    const Tensor &beta = context.hostInTensors.at(3);
    const Tensor &tensorOut = context.hostOutTensors.at(0);
    std::vector<int64_t> shape = {inputX.desc.dims[0], inputX.desc.dims[1], inputX.desc.dims[2]};
    auto attensorInputX = at::from_blob(inputX.data, shape, at::kHalf).to(at::kFloat);
    auto attensorResIn = at::from_blob(resIn.data, shape, at::kHalf).to(at::kFloat);
    auto attensorGamma = at::from_blob(gamma.data, {gamma.desc.dims[0], gamma.desc.dims[1]}, at::kHalf).to(at::kFloat);
    auto attensorBeta = at::from_blob(beta.data, {beta.desc.dims[0], beta.desc.dims[1]}, at::kHalf).to(at::kFloat);
    attensorResIn = attensorResIn * opParam.zoomScaleValue;
    auto attensorTemp = at::add(attensorInputX, attensorResIn).to(at::kFloat);
    auto groundtruth =
        at::layer_norm(attensorTemp, {1, inputX.desc.dims[2]}, attensorGamma, attensorBeta, EPS, false).to(at::kHalf);
    auto attensorOut = at::from_blob(tensorOut.data, shape, at::kHalf);

    return PLNMatchAtTensorHalf(atol, rtol, attensorOut, groundtruth);
}
} // namespace AsdOps

TEST(PostLayerNormF16, PostLayerNormF16CanSupport)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 6656;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Mki::Kernel> kernel{nullptr};
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "PostLayerNormF16Kernel") {
            const std::string &opName = iter.opName;
            Mki::Operation *op = Mki::AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("PostLayerNormF16Kernel"));
        }
    }
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(PostLayerNormF16, PostLayerNormF16GetBestKernel)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    uint32_t a = 1;
    uint32_t b = 128;
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Mki::Operation *operation = Mki::AutoGen::GetOpByName("NormOperation");
    ASSERT_NE(operation, nullptr);

    auto kernel = operation->GetBestKernel(launchParam);
    ASSERT_NE(kernel, nullptr);
}

TEST(PostLayerNormF16, PostLayerNormF16InferShape)
{
    uint32_t a = 10;
    uint32_t b = 65504;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Mki::Operation *operation = Mki::AutoGen::GetOpByName("NormOperation");
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
        for (size_t j = 0; j < outTensor[i].desc.dims.size(); ++j) {
            flag &= outTensor[i].desc.dims[j] == launchParam.GetOutTensor(i).desc.dims[j];
        }
    }
    ASSERT_EQ(flag, true);
}

TEST(PostLayerNormF16, PostLayerNormF16InferShapeFailed)
{
    uint32_t a = 10;
    uint32_t b = 65504;
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 1, 1}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Mki::Operation *operation = Mki::AutoGen::GetOpByName("NormOperation");
    ASSERT_EQ(operation != nullptr, true);
    Status status = operation->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);

    bool flag = true;
    for (size_t i = 0; i < launchParam.GetOutTensorCount(); ++i) {
        flag &= outTensor[i].desc.dtype == launchParam.GetOutTensor(i).desc.dtype;
        flag &= outTensor[i].desc.format == launchParam.GetOutTensor(i).desc.format;
        flag &= outTensor[i].desc.dims.size() == launchParam.GetOutTensor(i).desc.dims.size();
        for (size_t j = 0; j < outTensor[i].desc.dims.size(); ++j) {
            flag &= outTensor[i].desc.dims[j] == launchParam.GetOutTensor(i).desc.dims[j];
        }
    }
    ASSERT_EQ(flag, false);
}

TEST(PostLayerNormF16, PostLayerNormF16op1)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 6656;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Norm opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "NormOperation";
    opParam.epsilon = 0.00001;
    opParam.opsMode = 0;
    opParam.zoomScaleValue = 1;
    opParam.normType = AsdOps::OpParam::Norm::LAYER_NORM;
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(PostLayerNormGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(PostLayerNormF16, PostLayerNormF16op2)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 4096;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Norm opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "NormOperation";
    opParam.epsilon = 0.00001;
    opParam.opsMode = 0;
    opParam.zoomScaleValue = 0.5;
    opParam.normType = AsdOps::OpParam::Norm::LAYER_NORM;
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(PostLayerNormGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(PostLayerNormF16, PostLayerNormF16op3)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 1;
    uint32_t b = 128;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Norm opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "NormOperation";
    opParam.epsilon = 0.00001;
    opParam.opsMode = 0;
    opParam.zoomScaleValue = 0.3;
    opParam.normType = AsdOps::OpParam::Norm::LAYER_NORM;
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(PostLayerNormGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(PostLayerNormF16, PostLayerNormF16op4)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 10;
    uint32_t b = 4096;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Norm opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "NormOperation";
    opParam.epsilon = 0.00001;
    opParam.opsMode = 0;
    opParam.zoomScaleValue = 2;
    opParam.normType = AsdOps::OpParam::Norm::LAYER_NORM;
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(PostLayerNormGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(PostLayerNormF16, PostLayerNormF16op5)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    uint32_t a = 4096;
    uint32_t b = 6656;
    Mki::Test::MkiOpTest opTest;
    AsdOps::OpParam::Norm opParam;
    Mki::Test::UtOpDesc opDesc;
    opDesc.opName = "NormOperation";
    opParam.epsilon = 0.00001;
    opParam.opsMode = 0;
    opParam.zoomScaleValue = 10;
    opParam.normType = AsdOps::OpParam::Norm::LAYER_NORM;
    opParam.inRes = true;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opDesc.specificParam = opParam;
    opTest.Golden(std::bind(PostLayerNormGolden, ATOL, RTOL, opParam, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {a, 1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
