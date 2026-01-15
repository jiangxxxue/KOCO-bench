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
#include <gtest/gtest.h>
#include <future>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "kernels/split/split/tiling/split_tiling.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr float HALF_FLOAT_MIN = 1;
constexpr float HALF_FLOAT_MAX = 30000;
constexpr int64_t LONG_MIN_VALUE = 1;
constexpr int64_t LONG_MAX_VALUE = 30000;

Status SplitGolden(OpParam::Split opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);

    SVector<int64_t> dims = inTensor1.desc.dims;

    std::vector<Tensor> outTensors(0);
    std::vector<at::Tensor> atOutTensors(0);
    for (int i = 0; i < context.hostOutTensors.size(); i++) {
        outTensors.push_back(context.hostOutTensors.at(i));
        atOutTensors.push_back(at::from_blob(outTensors[i].data, ToIntArrayRef(outTensors[i].desc.dims), at::kHalf));
    }

    opParam.splitDim = (opParam.splitDim < 0) ? (static_cast<int>(dims.size()) + opParam.splitDim) : opParam.splitDim;
    std::vector<at::Tensor> refOutTensors =
        torch::split(atInRefTensor1, dims.at(opParam.splitDim) / opParam.splitNum, opParam.splitDim);
    for (int i = 0; i < refOutTensors.size(); i++) {
        refOutTensors[i] = refOutTensors[i].contiguous();
    }
    std::vector<fp16_t *> atOutArrays(0);
    std::vector<fp16_t *> atRefOutArrays(0);
    for (int i = 0; i < context.hostOutTensors.size(); i++) {
        atOutArrays.push_back((fp16_t *)atOutTensors[i].storage().data_ptr().get());
        atRefOutArrays.push_back((fp16_t *)refOutTensors[i].storage().data_ptr().get() + refOutTensors[i].storage_offset());
    }
    for (size_t i = 0; i < context.hostOutTensors.size(); i++) {
        for (int j = 0; j < outTensors[i].Numel(); j++) {
            float expect = static_cast<float>(atRefOutArrays[i][j]);
            float actual = static_cast<float>(atOutArrays[i][j]);
            bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
            if (!judge) {
                MKI_LOG(ERROR) << "expect:" << expect << ", actual:" << actual;
                return Status::FailStatus(-1, "error answer!");
            }
        }
    }
    return Status::OkStatus();
}

Status SplitInt64Golden(OpParam::Split opParam, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kLong);

    SVector<int64_t> dims = inTensor1.desc.dims;

    std::vector<Tensor> outTensors(0);
    std::vector<at::Tensor> atOutTensors(0);
    for (int i = 0; i < context.hostOutTensors.size(); i++) {
        outTensors.push_back(context.hostOutTensors.at(i));
        atOutTensors.push_back(at::from_blob(outTensors[i].data, ToIntArrayRef(outTensors[i].desc.dims), at::kLong));
    }

    opParam.splitDim = (opParam.splitDim < 0) ? (static_cast<int>(dims.size()) + opParam.splitDim) : opParam.splitDim;
    std::vector<at::Tensor> refOutTensors =
        torch::split(atInRefTensor1, dims.at(opParam.splitDim) / opParam.splitNum, opParam.splitDim);
    for (int i = 0; i < refOutTensors.size(); i++) {
        refOutTensors[i] = refOutTensors[i].contiguous();
    }
    std::vector<int64_t *> atOutArrays(0);
    std::vector<int64_t *> atRefOutArrays(0);
    for (int i = 0; i < context.hostOutTensors.size(); i++) {
        atOutArrays.push_back((int64_t *)atOutTensors[i].storage().data_ptr().get());
        atRefOutArrays.push_back((int64_t *)refOutTensors[i].storage().data_ptr().get() + refOutTensors[i].storage_offset());
    }
    for (size_t i = 0; i < context.hostOutTensors.size(); i++) {
        for (int j = 0; j < outTensors[i].Numel(); j++) {
            int64_t expect = atRefOutArrays[i][j];
            int64_t actual = atOutArrays[i][j];
            bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
            if (!judge) {
                MKI_LOG(ERROR) << "expect:" << expect << ", actual:" << actual;
                return Status::FailStatus(-1, "error answer!");
            }
        }
    }
    return Status::OkStatus();
}

TEST(TestOpSplit, splitTest1)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {1, 3};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    int64_t m = 3;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, m, m}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest2)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {2, 3};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 3}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest3)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {1, 2};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest4)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {2, 2};
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(std::bind(SplitInt64Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest5)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {0, 2};
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(std::bind(SplitInt64Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest6)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {-1, 2};
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(std::bind(SplitInt64Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSplit, splitTest7)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {-3, 2};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestGetBestKernelSplit0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    OpParam::Split opParam = {0, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestGetBestKernelSplit1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestGetBestKernelSplit2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    OpParam::Split opParam = {1, 3};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief tensor size wrong
 */
TEST(TestOpSplit, TestGetBestKernelSplit3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestInferShapeSplit0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief splitNum wrong
 */
TEST(TestOpSplit, TestInferShapeSplit1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, -1};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief splitDim wrong
 */
TEST(TestOpSplit, TestInferShapeSplit2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {4, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief inTensor format wrong
 */
TEST(TestOpSplit, TestInferShapeSplit3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief dims[splitDim] mod is not zero
 */
TEST(TestOpSplit, TestInferShapeSplit4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief splitDim wrong
 */
TEST(TestOpSplit, TestInferShapeSplit5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {-4, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestCanSupportSplit0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitF16Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestCanSupportSplit1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitInt64Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestOpSplit, TestCanSupportSplit2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 3, 4096 * 2}}});
    OpParam::Split opParam = {1, 3};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitF16Output3Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpSplit, TestCanSupportSplit3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitInt64Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpSplit, TestCanSupportSplit4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitInt64Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpSplit, TestCanSupportSplit5)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {8, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitInt64Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpSplit, TestCanSupportSplit6)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitF16Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor splitDim wrong
 */
TEST(TestOpSplit, TestCanSupportSplit7)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {7, 4, 4096 * 2}}});
    OpParam::Split opParam = {-3, 2};
    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SplitF16Output2Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}


/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpSplit, TestSplitCaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    OpParam::Split opParam = {1, 3};
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));

    Mki::Test::MkiOpTest opTest1;
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest1.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));

    Mki::Test::MkiOpTest opTest2;
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest2.Golden(std::bind(SplitGolden, opParam, std::placeholders::_1));

    Mki::Test::UtOpDesc opDesc = {"SplitOperation", opParam};
    int64_t m = 3;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, m, m}}};

    Mki::Test::UtOpDesc opDesc1 = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc1 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, m, m}}};

    Mki::Test::UtOpDesc opDesc2 = {"SplitOperation", opParam};
    SVector<TensorDesc> inTensorDesc2 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, m, m}}};

    auto test1 = [&]() {
        return opTest.Run(opDesc, inTensorDesc);
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, inTensorDesc1);
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, inTensorDesc2);
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
