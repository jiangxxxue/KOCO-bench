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
#include "asdops/params/transpose.h"
#include "kernels/transpose/tiling/transpose_tiling.h"
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

void TransposeGolden(at::Tensor &atInTensor, int32_t dim0, int32_t dim1, at::Tensor &atOutTensor)
{
    atOutTensor = at::transpose(atInTensor, dim0, dim1);
}

Status TransposeCompare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    at::Tensor atInTensor;
    at::Tensor atOutTensor;
    OpParam::Transpose param = AnyCast<OpParam::Transpose>(opDesc.specificParam);
    SVector<int32_t> perm = param.perm;
    if (perm.size() != inTensor.desc.dims.size()) {
        return Status::FailStatus(-1, "wrong input perm");
    }

    int32_t count = 0;
    SVector<int32_t> dims;
    for (size_t i = 0; i < perm.size(); i++) {
        if (static_cast<size_t>(perm[i]) != i) {
            count++;
            dims.push_back(perm[i]);
        }
    }
    if (count < 2) {
        return Status::FailStatus(-1, "wrong input perm");
    }
    if (count > 2) {
        return Status::FailStatus(-1,
                                  "Currently, the test case does not support transpose of more than two dimensions. "
                                  "However, the operator supports simultaneous transpose of multiple dimensions.");
    }

    if (inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
        atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    } else if (inTensor.desc.dtype == TENSOR_DTYPE_INT64) {
        atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kLong);
        atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kLong);
    }

    at::Tensor atOutRefTensor;
    TransposeGolden(atInTensor, dims[0], dims[1], atOutRefTensor);

    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
} // namespace

TEST(TestOpTranspose, TestTransposeF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 32, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {0, 2, 1};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 32, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0, 2};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 32, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {2, 1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32, 12, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0, 2, 3};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32, 12, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case6)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32, 12, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {0, 2, 1, 3};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case7)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 32, 12, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {0, 3, 2, 1};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case8)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 1000000}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case9)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1000000, 10}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeF16Case10)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}});

    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeInt64Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.LongRand(LONG_MIN, LONG_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {150, 1}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeInt64Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(LONG_MIN, LONG_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 32, 12, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {0, 3, 2, 1};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeInt64Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(LONG_MIN, LONG_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {12, 32, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {2, 1, 0};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTranspose, TestTransposeInt64Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(LONG_MIN, LONG_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {12, 32, 128}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0, 2};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}


/**
 * @brief ok
 */
TEST(TestOpTranspose, TestGetBestKernelTranspose0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief input perm wrong
 */
TEST(TestOpTranspose, TestGetBestKernelTranspose1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpTranspose, TestInferShapeTranspose0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief input perm wrong
 */
TEST(TestOpTranspose, TestInferShapeTranspose1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief input perm wrong
 */
TEST(TestOpTranspose, TestInferShapeTranspose2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 1, 2, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief  Wrong perm or input
 */
TEST(TestOpTranspose, TestInferShapeTranspose3)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 7};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpTranspose, TestCanSupportTranspose0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("Transpose16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpTranspose, TestCanSupportTranspose1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, 5, 8, 100000, 8}}});
    OpParam::Transpose opParam = {};
    opParam.perm = {3, 1, 2, 0, 4};
    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("Transpose16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpTranspose, TestTransposeF16CaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(std::bind(&TransposeCompare, ATOL, RTOL, std::placeholders::_1));
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});
    SVector<TensorDesc> inTensorDescs1;
    inTensorDescs1.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});
    SVector<TensorDesc> inTensorDescs2;
    inTensorDescs2.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3}});

    OpParam::Transpose opParam = {};
    opParam.perm = {1, 0};

    Mki::Test::UtOpDesc opDesc = {"TransposeOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"TransposeOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"TransposeOperation", opParam};

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