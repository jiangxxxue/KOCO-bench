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
#include <mki/launch_param.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/gather.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;

template <typename T>
bool GatherGolden(std::vector<T> input, std::vector<int64_t> indices, int64_t axis,
                  std::vector<SVector<int64_t>> shapes,
                  std::vector<T> &output)
{
    if (axis < 0) {
        MKI_LOG(ERROR) << "wrong input of axis";
        return false;
    }
    SVector<int64_t> inputDims = shapes[0];
    SVector<int64_t> indicesDims = shapes[1];
    SVector<int64_t> outputDims = shapes[2];

    SVector<int64_t> outputDimsInfer;
    int64_t dim0 = 1;
    for (int i = 0; i < axis; i++) {
        outputDimsInfer.push_back(inputDims[i]);
        dim0 *= inputDims[i];
    }
    int64_t dim1 = inputDims[axis];
    for (int i = 0; i < static_cast<int>(indicesDims.size()); i++) {
        outputDimsInfer.push_back(indicesDims[i]);
    }
    int64_t dim2 = 1;
    for (int i = axis + 1; i < static_cast<int>(inputDims.size()); i++) {
        outputDimsInfer.push_back(inputDims[i]);
        dim2 *= inputDims[i];
    }

    bool infer = std::equal(outputDims.begin(), outputDims.end(), outputDimsInfer.begin());
    if (!infer) {
        MKI_LOG(ERROR) << "wrong input of output shape";
        return false;
    }

    if (dim0 * dim1 * dim2 != static_cast<int>(input.size())) {
        MKI_LOG(ERROR) << "Wrong compute shapes of fuse axis";
        return false;
    }

    output.resize(dim0 * static_cast<int64_t>(indices.size()) * dim2);
    int64_t idx = 0;
    for (int i = 0; i < dim0; i++) {
        int64_t inputIdx = i * dim1 * dim2;
        for (auto indice : indices) {
            if (indice >= dim1) {
                MKI_LOG(ERROR) << "Wrong indices: " << indice;
                return false;
            }
            for (int64_t k = 0; k < dim2; k++) {
                output[idx++] = input[inputIdx + indice * dim2 + k];
            }
        }
    }

    return true;
}

Status GatherV2F16Compare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &indices = context.hostInTensors.at(1);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Gather param = AnyCast<OpParam::Gather>(opDesc.specificParam);
    SVector<int64_t> axises = param.axis;
    if (axises.size() != 1) {
        return Status::FailStatus(-1, "multi axis input, not support now");
    }
    int64_t axis = axises[0];

    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atInTensorFloat = atInTensor.to(at::kFloat);
    at::Tensor atIndices = at::from_blob(indices.data, ToIntArrayRef(indices.desc.dims), at::kLong);

    std::vector<float> inputFloat(atInTensorFloat.data_ptr<float>(),
                                  atInTensorFloat.data_ptr<float>() + atInTensorFloat.numel());
    std::vector<fp16_t> input(inputFloat.size());
    std::transform(inputFloat.begin(), inputFloat.end(), input.begin(), [] (float f) {
        return static_cast<fp16_t>(f);
    });
    std::vector<int64_t> indicesVector(atIndices.data_ptr<int64_t>(),
                                       atIndices.data_ptr<int64_t>() + atIndices.numel());

    std::vector<SVector<int64_t>> shapes;
    shapes.push_back(inTensor.desc.dims);
    shapes.push_back(indices.desc.dims);
    shapes.push_back(outTensor.desc.dims);
    std::vector<fp16_t> output;
    if (!GatherGolden(input, indicesVector, axis, shapes, output)) {
        return Status::FailStatus(-1, "compute golden failed");
    }

    at::Tensor outputRef = at::from_blob(output.data(), ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    if (!at::allclose(outputRef, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
}

TEST(TestOpGather, TestGatherV2F16Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);
    OpParam::Gather opParam;
    opParam.axis = {0};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 8}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 3}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 150527);
    OpParam::Gather opParam;
    opParam.axis = {0};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {150528, 28}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 12}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 11);
    OpParam::Gather opParam;
    opParam.axis = {0};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {12, 64}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {12, 1}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 4);
    OpParam::Gather opParam;
    opParam.axis = {1};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 5}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 1}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 5);
    OpParam::Gather opParam;
    opParam.axis = {1};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 6, 7, 8, 9}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 4, 5}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 49);
    OpParam::Gather opParam;
    opParam.axis = {2};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 5, 50, 100}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 6, 2, 9}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case6)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 199);
    OpParam::Gather opParam;
    opParam.axis = {3};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {2, 3, 4, 200}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 4, 3, 6}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case7)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 6);
    OpParam::Gather opParam;
    opParam.axis = {0};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {7, 10, 5, 20}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {9, 8, 7, 6}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpGather, TestGatherV2F16Case8)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 7);
    OpParam::Gather opParam;
    opParam.axis = {4};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 5, 6, 7, 8, 9}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 4, 5}});
    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpGather, TestCanSupportGather1)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {4, 5, 6, 7, 8, 9, 10}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {3, 4, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Gather opParam;
    opParam.axis = {4};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("Gather16I64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpGather, TestCanSupportGather2)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 5, 6, 7, 8, 9, 10}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {3, 4, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Gather opParam;
    opParam.axis = {4};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("Gather16I64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpGather, TestGatherV2F16CaseAutoTiling)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest1.LongRand(0, 3);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(std::bind(&GatherV2F16Compare, ATOL, RTOL, std::placeholders::_1));
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest2.LongRand(0, 3);

    OpParam::Gather opParam;
    opParam.axis = {0};
    Mki::Test::UtOpDesc opDesc = {"GatherOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"GatherOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"GatherOperation", opParam};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 8}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 3}});
    SVector<TensorDesc> inTensorDescs1;
    inTensorDescs1.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 8}});
    inTensorDescs1.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 3}});
    SVector<TensorDesc> inTensorDescs2;
    inTensorDescs2.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 8}});
    inTensorDescs2.push_back({TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, 3}});
    
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