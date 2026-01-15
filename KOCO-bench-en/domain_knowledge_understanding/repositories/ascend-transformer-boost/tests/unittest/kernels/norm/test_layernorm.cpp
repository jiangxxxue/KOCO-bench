/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <mki/base/operation_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/norm.h"
#include "kernels/norm/layernorm/tiling/layernorm_tiling.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "device_version_check.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = std::pow(65504, 1.0 / 4);
constexpr float EXTENT_OF_ERROR = 0.001;

Status LayernormF32Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &x = context.hostInTensors.at(0);
    const Tensor &gamma = context.hostInTensors.at(1);
    const Tensor &beta = context.hostInTensors.at(2);
    const Tensor &y = context.hostOutTensors.at(0);

    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Norm param = AnyCast<OpParam::Norm>(opDesc.specificParam);
    float epsilon = param.epsilon;
    at::Tensor atInTensor0, atOutTensor, atInTensor1, atInTensor2;
    at::Tensor atInTensorx, atOutTensory, atOutTensorytmp1, atOutTensorytmp2;

    if (x.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        atInTensor0 = at::from_blob(x.data, ToIntArrayRef(x.desc.dims), at::kHalf);
        atOutTensor = at::from_blob(y.data, ToIntArrayRef(y.desc.dims), at::kHalf);
        atInTensor1 = at::from_blob(gamma.data, ToIntArrayRef(gamma.desc.dims), at::kHalf);
        atInTensor2 = at::from_blob(beta.data, ToIntArrayRef(beta.desc.dims), at::kHalf);
        atInTensorx = atInTensor0.to(at::kFloat);
        atOutTensory = atOutTensor.to(at::kFloat);
        atOutTensorytmp1 = atInTensor1.to(at::kFloat);
        atOutTensorytmp2 = atInTensor2.to(at::kFloat);
    } else {
        atInTensor0 = at::from_blob(x.data, ToIntArrayRef(x.desc.dims), at::kFloat);
        atOutTensor = at::from_blob(y.data, ToIntArrayRef(y.desc.dims), at::kFloat);
        atInTensor1 = at::from_blob(gamma.data, ToIntArrayRef(gamma.desc.dims), at::kFloat);
        atInTensor2 = at::from_blob(beta.data, ToIntArrayRef(beta.desc.dims), at::kFloat);
        atInTensorx = atInTensor0;
        atOutTensory = atOutTensor;
        atOutTensorytmp1 = atInTensor1;
        atOutTensorytmp2 = atInTensor2;
    }

    std::vector<int64_t> shape;
    for (auto dim : gamma.desc.dims) {
        shape.push_back(dim);
    }

    at::Tensor groundtruth = at::layer_norm(atInTensorx, shape, atOutTensorytmp1, atOutTensorytmp2, epsilon, false);
    if (!at::allclose(groundtruth, atOutTensory, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }
    return Status::OkStatus();
}
} // namespace

std::unique_ptr<Mki::Kernel> GetKernelByName(const std::string &kernelName)
{
    Kernel *kernel = nullptr;
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == kernelName) {
            const std::string &opName = iter.opName;
            Mki::Operation *op = Mki::AutoGen::GetOpByName(opName);
            kernel = op->GetKernelByName(kernelName);
        }
    }
    return std::unique_ptr<Mki::Kernel>(kernel);
}

TEST(TestOpNormLayernorm, LayernormF16CanSupport)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Mki::Kernel> kernel = GetKernelByName("LayernormF16Kernel");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpNormLayernorm, LayernormBF16CanSupport)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Mki::Kernel> kernel = GetKernelByName("LayernormBF16Kernel");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpNormLayernorm, LayernormF32CanSupport)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Mki::Kernel> kernel = GetKernelByName("LayernormF32Kernel");
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpNormLayernorm, LayernormGetBestKernel)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Mki::Operation *operation = Mki::AutoGen::GetOpByName("NormOperation");
    ASSERT_NE(operation, nullptr);

    auto kernel = std::unique_ptr<Mki::Kernel>(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpNormLayernorm, LayernormInferShape)
{
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
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

TEST(TestOpNormLayernorm, LayernormInferShapeFailed)
{
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 3}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    outTensor.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 3}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 1}});
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
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

TEST(TestOpNormLayernorm, NormLayernormCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5, 6}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 4, 5, 6}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 4, 5, 6}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = -1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5, 6, 7}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = -1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = -1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = -1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase6)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase7)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 0;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase8)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase9)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 1;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 100}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {100}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {100}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase10)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 0;
    opParam.beginParamsAxis = 1;
    opParam.epsilon = 0.1f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1000000}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormLayernorm, NormLayernormCase11)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&LayernormF32Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Norm opParam = {OpParam::Norm::LAYER_NORM};
    opParam.beginNormAxis = 2;
    opParam.beginParamsAxis = 0;
    opParam.epsilon = 0.01f;
    opParam.inGamma = true;
    opParam.inBeta = true;
    opParam.outMean = true;
    opParam.outVarience = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {78, 79, 80, 81}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {80, 81}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {80, 81}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}