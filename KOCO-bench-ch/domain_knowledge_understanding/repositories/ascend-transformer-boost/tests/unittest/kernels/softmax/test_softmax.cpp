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
#include <cmath>
#include <future>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "kernels/softmax/tiling/softmax_tiling.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "device_version_check.h"

static constexpr float EXTENT_OF_ERROR = 0.001;
static constexpr float HALF_FLOAT_MIN = -10.0;
static constexpr float HALF_FLOAT_MAX = 10.0;

using namespace AsdOps;
using namespace Mki;

template <typename T = float>
bool JudgeSoftmax(T *output, size_t size, float target)
{
    float count = 0;
    for (size_t i = 0; i < size; i++) {
        count += static_cast<float>(output[i]);
    }
    if (target == 0) {
        MKI_LOG(ERROR) << "wrong calculate target: " << target;
        return false;
    }
    if (abs((float)1 - count / target) < EXTENT_OF_ERROR) {
        return true;
    } else {
        MKI_LOG(ERROR) << "softmax sum: " << count << ", but expect: " << target;
        return false;
    }
}

template <typename T = float>
Status SoftMaxGolden(SVector<int64_t> &axes, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    auto &inDims = inTensor.desc.dims;

    float target = 1;
    for (size_t i = 0; i < inDims.size(); i++) {
        target *= inDims[i];
    }
    for (size_t i = 0; i < axes.size(); i++) {
        target /= inDims[axes[i]];
    }

    T *output = static_cast<T *>(outTensor.data);
    return JudgeSoftmax(output, outTensor.Numel(), target) ?
        Status::OkStatus() : Status::FailStatus(-1, "judge failed");
}

TEST(TestOpSoftmax, SoftmaxCase0)
{
    OpParam::Softmax opParam;
    opParam.axes = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase1)
{
    OpParam::Softmax opParam;
    opParam.axes = {2, 3};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase2)
{
    OpParam::Softmax opParam;
    opParam.axes = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 4, 5, 6}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase3)
{
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase4)
{
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden<fp16_t>, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {9, 8, 7, 6, 5}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase5)
{
    OpParam::Softmax opParam;
    opParam.axes = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden<fp16_t>, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4, 5, 6}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSoftmax, SoftmaxCase6)
{
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden<fp16_t>, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}});

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpSoftmax, TestGetBestKernelSoftmax0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSoftmax, TestGetBestKernelSoftmax1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpSoftmax, TestGetBestKernelSoftmax2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSoftmax, TestInferShapeSoftmax0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief opParam.axes size wrong
 */
TEST(TestOpSoftmax, TestInferShapeSoftmax1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief opParam.axes wrong
 */
TEST(TestOpSoftmax, TestInferShapeSoftmax2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 3};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpSoftmax, TestCanSupportSoftmax0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SoftmaxF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestOpSoftmax, TestCanSupportSoftmax1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SoftmaxF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpSoftmax, TestCanSupportSoftmax2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SoftmaxF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpSoftmax, TestCanSupportSoftmax3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {10, 10, 10, 10, 10, 10}}});
    OpParam::Softmax opParam;
    opParam.axes = {2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SoftmaxF32Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpSoftmax, TestSoftmaxCaseAutoTiling)
{
    OpParam::Softmax opParam;
    opParam.axes = {0, 1};

    Mki::Test::UtOpDesc opDesc = {"SoftmaxOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"SoftmaxOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"SoftmaxOperation", opParam};

    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.SetOutputNum(1);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest1.SetOutputNum(1);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(std::bind(SoftMaxGolden, opParam.axes, std::placeholders::_1));
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest2.SetOutputNum(1);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5}});
    SVector<TensorDesc> inTensorDescs1;
    inTensorDescs1.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5}});
    SVector<TensorDesc> inTensorDescs2;
    inTensorDescs2.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {2, 3, 4, 5}});

    auto test1 = [&]() {
        return opTest.Run(opDesc, inTensorDescs);
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, inTensorDescs1);
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, inTensorDescs2);
    };

    for (uint32_t i = 0; i < 100; i++) {
        std::future<Status> result1 = std::async(test1);
        std::future<Status> result2 = std::async(test2);
        std::future<Status> result3 = std::async(test3);
        Status status1 = result1.get();
        Status status2 = result2.get();
        Status status3 = result3.get();
        ASSERT_EQ(status1.Ok() && status2.Ok() && status3.Ok() , true);
    }
}