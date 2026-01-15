/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <gtest/gtest.h>
#include <mki/base/operation_base.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include <mki_loader/op_register.h>
#include "asdops/params/params.h"
#include "kernels/norm/rmsnorm/tiling/rms_norm_tiling.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "device_version_check.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = std::pow(65504, 1.0 / 4);

void RmsNormRef(const std::vector<double> &x, const std::vector<double> &g, std::vector<float> &y, float epsilon)
{
    // norm only for the last dimension
    MKI_LOG(INFO) << "groundtruth: x.size: " << x.size() << ", g.size: " << g.size();
    size_t n = g.size();
    if (n == 0) {
        MKI_LOG(ERROR) << "error g, compute groundtruth failed";
        return;
    }
    if (x.size() % n != 0) {
        MKI_LOG(ERROR) << "error input!";
        return;
    }

    for (size_t i = 0; i < x.size() / n; ++i) {
        double norm = 0.0;
        for (size_t j = 0; j < n; ++j) {
            norm += x[i * n + j] / static_cast<double>(n) * x[i * n + j];
        }
        norm += epsilon;
        for (size_t j = 0; j < n; ++j) {
            y.push_back(static_cast<float>(x[i * n + j] * g[j] / std::sqrt(norm)));
        }
    }
}

Status RmsNormCompare(float atol, float rtol, float epsilon, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor0 = context.hostInTensors.at(0);
    const Tensor &inTensor1 = context.hostInTensors.at(1);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    fp16_t *indata0 = static_cast<fp16_t *>(inTensor0.data);
    std::vector<double> x;
    for (int i = 0; i < inTensor0.Numel(); ++i) {
        x.push_back(static_cast<double>(indata0[i]));
    }

    fp16_t *indata1 = static_cast<fp16_t *>(inTensor1.data);
    std::vector<double> g;
    for (int i = 0; i < inTensor1.Numel(); ++i) {
        g.push_back(static_cast<double>(indata1[i]));
    }

    fp16_t *outdata = static_cast<fp16_t *>(outTensor.data);
    std::vector<float> y;
    for (int i = 0; i < outTensor.Numel(); ++i) {
        y.push_back(static_cast<float>(outdata[i]));
    }

    std::vector<float> yRef;
    RmsNormRef(x, g, yRef, epsilon);

    if (y.size() != yRef.size()) {
        return Status::FailStatus(-1, "size not equal");
    }

    for (size_t i = 0; i < y.size(); ++i) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(yRef[i], y[i], atol, rtol)) {
            return Status::FailStatus(-1, "judge not equal");
        }
    }

    return Status::OkStatus();
}
} // namespace

TEST(TestOpNormRmsNorm, TestOpNormRmsNormCanSupport)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    std::unique_ptr<Mki::Kernel> kernel{nullptr};
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &iter : kernelCreators) {
        if (iter.kernelName == "RmsNormKernel") {
            const std::string &opName = iter.opName;
            Mki::Operation *op = Mki::AutoGen::GetOpByName(opName);
            kernel.reset(op->GetKernelByName("RmsNormKernel"));
        }
    }
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpNormRmsNorm, TestOpNormRmsNormGetBestKernel)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);

    Mki::Operation *operation = Mki::AutoGen::GetOpByName("NormOperation");
    ASSERT_NE(operation, nullptr);

    auto kernel = std::unique_ptr<Mki::Kernel>(operation->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

TEST(TestOpNormRmsNorm, TestOpNormRmsNormInferShape)
{
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.inGamma = true;
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

TEST(TestOpNormRmsNorm, TestOpNormRmsNormInferShapeFailed)
{
    SVector<Tensor> outTensor;
    outTensor.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 32}});
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32}});
    launchParam.AddOutTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.inGamma = true;
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

TEST(TestOpNormRmsNorm, RmsNormF32Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-12;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 32}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-5;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 16}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-8;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 3200}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3200}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-9;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10000, 96}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {96}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-5;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4, 5, 6, 7, 32}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-8;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {20, 5120}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5120}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case6)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-5;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {40, 16016}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16016}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case7)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-16;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {96, 16368}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16368}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case8)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-6;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {80, 9600}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {9600}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case9)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-12;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {120, 8208}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {8208}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case10)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-14;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4, 5, 6, 10224}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10224}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case11)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-5;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2,3,4,16368}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16368}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case12)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 0;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16,16400}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16400}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpNormRmsNorm, RmsNormF32Case13)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    OpParam::Norm opParam{OpParam::Norm::RMS_NORM};
    opParam.epsilon = 1e-7;
    opParam.inGamma = true;
    Mki::Test::UtOpDesc opDesc = {"NormOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&RmsNormCompare, ATOL, RTOL, opParam.epsilon, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16,7200}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {16,7200}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7200}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), false);
}