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
#include <cmath>
#include <sstream>
#include <torch/torch.h>
#include <mki/utils/log/log.h>
#include "asdops/params/reverse.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;
namespace {
constexpr float ATOL = 0.01;
constexpr float RTOL = 0.01;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;

void ReverseGolden(at::Tensor &atInTensor, SVector<int32_t> &axis, at::Tensor &atOutRefTensor)
{
    std::vector <int64_t> vec(axis.begin(), axis.end());
    atOutRefTensor = torch::flip(atInTensor, vec);
}

static Status ReverseF16Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;

    OpParam::Reverse param = AnyCast<OpParam::Reverse>(opDesc.specificParam);
    SVector<int32_t> axis = param.axis;

    auto dtype = (inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) ? at::kHalf : at::kFloat;

    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), dtype);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), dtype);
    at::Tensor atOutRefTensor = atInTensor;
    ReverseGolden(atInTensor, axis, atOutRefTensor);
    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }
    return Status::OkStatus();
}
}

TEST(TestOpReverse, TestReverseF16Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 2, n = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReverse, TestReverseF16Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReverse, TestReverseF16Case2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {1, 3, 2};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 30, n = 58, p = 58, q = 17;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReverse, TestReverseF16Case3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {0, 2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 42, n = 12, k = 40, p = 18, q = 49;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReverse, TestReverseF16Case4)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {1, 1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpReverse, TestReverseF16Case5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {0, 1, 2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpReverse, TestReverseCanSupport0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0, 1, 2, 3, 4}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {0, 1, 2, 3, 4}}});
    OpParam::Reverse opParam;
    opParam.axis = {0, 1, 2, 3, 4};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReverseF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpReverse, TestReverseF16Case6)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {4};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}
 
TEST(TestOpReverse, TestReverseF16Case7)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {1, 5};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}
 
TEST(TestOpReverse, TestReverseF32Case8)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, p, q}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
 
TEST(TestOpReverse, TestReverseF16Case9)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReverseF16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reverse opParam;
    opParam.axis = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    int64_t m = 1, n = 2, p = 3, q = 4;
    int64_t i = 1, j = 2, k = 3, l = 4;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, p, q}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {i, j, k, l}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpReverse, TestReverseF16Case10)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Reverse opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReverseF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
 
TEST(TestOpReverse, TestReverseF16Case11)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 7, 23}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 2}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 7, 23}}});
    OpParam::Reverse opParam = {};
    opParam.axis = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}
 
TEST(TestOpReverse, TestReverseF16Case12)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11, 3, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {20, 23}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {11, 3, 5}}});
    OpParam::Reverse opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReverseF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
 
TEST(TestOpReverse, TestReverseF16Case13)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {11, 3, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {11, 3, 5}}});
    OpParam::Reverse opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReverseF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
 
TEST(TestOpReverse, TestReverseF16Case14)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {3, 5}}});
    OpParam::Reverse opParam = {};
    Mki::Test::UtOpDesc opDesc = {"ReverseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("ReverseF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}
