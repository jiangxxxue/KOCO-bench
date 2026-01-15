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
#include <cmath>
#include <sstream>
#include <mki/utils/log/log.h>
#include "asdops/params/expand.h"
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
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;

void ExpandGolden(at::Tensor &atInTensor, at::Tensor &atOutTensor, at::Tensor &atOutRefTensor)
{
    atOutRefTensor = atInTensor.expand_as(atOutTensor);
}

Status ExpandF16Int64Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);

    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Expand param = AnyCast<OpParam::Expand>(opDesc.specificParam);
    SVector<int64_t> shape = param.shape;
    if (shape.size() < inTensor.desc.dims.size()) {
        return Status::FailStatus(-1, "wrong input shape");
    }

    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    at::Tensor atOutRefTensor;
    ExpandGolden(atInTensor, atOutTensor, atOutRefTensor);

    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
} // namespace

TEST(TestOpExpand, ExpandF16Int64Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {4, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpExpand, ExpandF16Int64Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 4, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {3, 4, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpExpand, ExpandF16Int64Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {3, 2, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpExpand, ExpandF16Int64Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {40, 1, 20, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {3, 40, 10, 20, 20};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpExpand, ExpandF16Int64Case4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 2, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {2, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    // shape[2] 需要比shape[1]大
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpExpand, ExpandF16Int64Case5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 2, 1}});

    OpParam::Expand opParam = {};
    opParam.shape = {-1, 2, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    // shape不支持负数
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpExpand, ExpandF16Int64Case6)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ExpandF16Int64Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 2, 2}});

    OpParam::Expand opParam = {};
    opParam.shape = {2, 2, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    // 扩展维度只能为1
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpExpand, TestCanSupportExpand0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 1}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {4, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ExpandF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpExpand, TestCanSupportExpand1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {2, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ExpandF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpExpand, TestCanSupportExpand2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ExpandF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpExpand, TestCanSupportExpand3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ExpandF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpExpand, TestCanSupportExpand4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ExpandF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpExpand, TestInferShapeExpand0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 1}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {6, 6}}});
    OpParam::Expand opParam = {};
    opParam.shape = {4, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief shape dim wrong
 */
TEST(TestOpExpand, TestInferShapeExpand1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 1, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {4, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpExpand, TestInferShapeExpand2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {4, 1}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {4, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief dims size wrong
 */
TEST(TestOpExpand, TestInferShapeExpand3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 2, 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {2, 2, 4};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief dims size wrong
 */
TEST(TestOpExpand, TestInferShapeExpand4)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 2, 1}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Expand opParam = {};
    opParam.shape = {-1, 2, 2};
    Mki::Test::UtOpDesc opDesc = {"ExpandOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}