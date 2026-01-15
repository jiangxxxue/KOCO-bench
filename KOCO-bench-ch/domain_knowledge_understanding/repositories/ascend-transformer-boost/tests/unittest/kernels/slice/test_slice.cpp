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
#include <future>
#include <mki/utils/log/log.h>
#include "asdops/params/slice.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.01;
constexpr float RTOL = 0.01;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr int INT_RAND_MIN = -INT_MAX;
constexpr int INT_RAND_MAX = INT_MAX;
constexpr float EXTENT_OF_ERROR = 0.001;

template <at::ScalarType T = at::kHalf>
Status SliceCompare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Slice param = AnyCast<OpParam::Slice>(opDesc.specificParam);
    SVector<int64_t> offsets = param.offsets;
    SVector<int64_t> size = param.size;

    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), T);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), T);
    at::Tensor atOutRefTensor = atInTensor;

    for (size_t i = 0; i < inTensor.desc.dims.size(); i++) {
        int64_t offsetValue = offsets[i];
        int64_t sizeValue = size[i];
        if (offsets[i] < 0) {
            offsetValue = offsets[i] + inTensor.desc.dims[i];
        }
        if (size[i] == -1) {
            sizeValue = inTensor.desc.dims[i] - offsets[i];
        }
        atOutRefTensor = at::slice(atOutRefTensor, i, offsetValue, offsetValue + sizeValue);
    }

    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
} // namespace

TEST(TestOpSlice, GetBestKernelNullptr)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {32, 128}});
    OpParam::Slice opParam;
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

TEST(TestOpSlice, OperationInferShapeWrongOffset)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    OpParam::Slice opParam;
    opParam.offsets = {8};
    opParam.size = {10, 100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(op->InferShape(launchParam).Ok(), false);
}

TEST(TestOpSlice, OperationInferShapeWrongInputSize)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(op->InferShape(launchParam).Ok(), false);
}

TEST(TestOpSlice, OperationInferShapeWrongInputSizeValue)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {10, -100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(op->InferShape(launchParam).Ok(), false);
}

TEST(TestOpSlice, OperationInferShapeInputSizeOutOfBounds)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {10, 121};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    ASSERT_EQ(op->InferShape(launchParam).Ok(), false);
}

TEST(TestOpSlice, TestSliceF16Int64Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});

    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {10, 100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSlice, TestSliceF16Int64Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});

    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {-1, 100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSlice, TestSliceF16Int64Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});

    OpParam::Slice opParam;
    opParam.offsets = {-20, 8};
    opParam.size = {10, 100};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSlice, TestSliceF16Int64Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128, 256}});

    OpParam::Slice opParam;
    opParam.offsets = {-20, -30, 0};
    opParam.size = {-1, 10, -1};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSlice, TestSliceInt32Int64Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare<at::kInt>, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(INT_RAND_MIN, INT_RAND_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_UINT32, TENSOR_FORMAT_ND, {32, 128, 256}});

    OpParam::Slice opParam;
    opParam.offsets = {-20, -30, 0};
    opParam.size = {-1, 10, -1};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpSlice, TestSliceInt8Int64Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare<at::kChar>, ATOL, RTOL, std::placeholders::_1));
    opTest.Int8Rand(-100, 100);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {32, 128, 256}});

    OpParam::Slice opParam;
    opParam.offsets = {-20, -30, 0};
    opParam.size = {-1, 10, -1};
    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpSlice, TestSliceF16Int64CaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(std::bind(&SliceCompare, ATOL, RTOL, std::placeholders::_1));
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    SVector<TensorDesc> inTensorDescs1;
    inTensorDescs1.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});
    SVector<TensorDesc> inTensorDescs2;
    inTensorDescs2.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {32, 128}});

    OpParam::Slice opParam;
    opParam.offsets = {2, 8};
    opParam.size = {10, 100};

    Mki::Test::UtOpDesc opDesc = {"SliceOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"SliceOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"SliceOperation", opParam};

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