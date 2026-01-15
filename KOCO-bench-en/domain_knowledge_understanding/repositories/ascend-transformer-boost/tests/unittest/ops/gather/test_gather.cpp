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
#include <torch/torch.h>
#include <atb/utils/log.h>
#include "test_utils/test_common.h"
#include "atb/operation.h"
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "mki/utils/fp16/fp16_t.h"
#include "atb/utils/singleton.h"
#include "atb/utils/config.h"

using namespace atb;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
static constexpr float HALF_FLOAT_MIN = -65504;
static constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;


Mki::Status GatherGolden(const OpsGoldenContext &context, atb::infer::GatherParam param)
{
    const Mki::Tensor &inTensor = context.hostInTensors.at(0);
    const Mki::Tensor &indices = context.hostInTensors.at(1);
    const Mki::Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atInTensorFloat = atInTensor.to(at::kFloat);
    at::Tensor atIndices = at::from_blob(indices.data, ToIntArrayRef(indices.desc.dims), at::kLong);
    std::vector<float> inputFloat(atInTensorFloat.data_ptr<float>(),
        atInTensorFloat.data_ptr<float>() + atInTensorFloat.numel());
    std::vector<Mki::fp16_t> input(inputFloat.size());
    std::transform(inputFloat.begin(), inputFloat.end(), input.begin(),
        [](float f) { return static_cast<Mki::fp16_t>(f); });
    std::vector<int64_t> indicesVector(atIndices.data_ptr<int64_t>(),
        atIndices.data_ptr<int64_t>() + atIndices.numel());

    std::vector<Mki::SVector<int64_t>> shapes;
    shapes.push_back(inTensor.desc.dims);
    shapes.push_back(indices.desc.dims);
    shapes.push_back(outTensor.desc.dims);
    Mki::SVector<int64_t> inputDims = shapes[0];
    Mki::SVector<int64_t> indicesDims = shapes[1];
    Mki::SVector<int64_t> outputDims = shapes[2];
    Mki::SVector<int64_t> outputDimsInfer;
    std::vector<Mki::fp16_t> output;
    int64_t dim0 = 1;
    int64_t axis = param.axis;
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
        return Mki::Status::FailStatus(1, "outputDims judge not equal");
    }
    output.resize(dim0 * static_cast<int64_t>(indicesVector.size()) * dim2);
    int64_t idx = 0;
    for (int i = 0; i < dim0; i++) {
        int64_t inputIdx = i * dim1 * dim2;
        for (auto indice : indicesVector) {
            if (indice >= dim1) {
                return Mki::Status::FailStatus(1, "indice is valid");
            }
            for (int64_t k = 0; k < dim2; k++) {
                output[idx++] = input[inputIdx + indice * dim2 + k];
            }
        }
    }
    at::Tensor outputRef = at::from_blob(output.data(), ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    if (!at::allclose(outputRef, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Mki::Status::FailStatus(1, "judge not equal");
    }

    return Mki::Status::OkStatus();
}

TEST(TestGatherOperation, TestGather)
{
    atb::infer::GatherParam param;
    param.axis = { 1 };
    atb::Operation *op;
    atb::Status status0 = atb::CreateOperation<atb::infer::GatherParam>(param, &op);
    ASSERT_EQ(status0, atb::NO_ERROR);
    Mki::SVector<Mki::TensorDesc> InTensorDescs = {
        { Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 5, 16, 96 } },
        { Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 31, 16 } }
    };
    OperationTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);
    opTest.Golden(std::bind(&GatherGolden, std::placeholders::_1, param));
    Mki::Status status1 = opTest.Run(op, InTensorDescs);
    ASSERT_EQ(status1.Ok(), true);
    atb::Status status2 = DestroyOperation(op);
    ASSERT_EQ(status2, atb::NO_ERROR);
}

TEST(TestGatherOperation, TestGather2)
{
    atb::infer::GatherParam param;
    param.axis = { 0 };
    atb::Operation *op;
    atb::Status status0 = atb::CreateOperation<atb::infer::GatherParam>(param, &op);
    ASSERT_EQ(status0, atb::NO_ERROR);
    Mki::SVector<Mki::TensorDesc> InTensorDescs = {
        { Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 5, 16, 96 } },
        { Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 31, 16 } }
    };
    OperationTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);
    opTest.Golden(std::bind(&GatherGolden, std::placeholders::_1, param));
    Mki::Status status1 = opTest.Run(op, InTensorDescs);
    ASSERT_EQ(status1.Ok(), true);
    atb::Status status2 = DestroyOperation(op);
    ASSERT_EQ(status2, atb::NO_ERROR);
}

TEST(TestGatherOperation, TestGather3)
{
    if (GetSingleton<atb::Config>().Is310B()) {
        return;
    }

    atb::infer::GatherParam param;
    param.axis = { 2 };
    atb::Operation *op;
    atb::Status status0 = atb::CreateOperation<atb::infer::GatherParam>(param, &op);
    ASSERT_EQ(status0, atb::NO_ERROR);
    Mki::SVector<Mki::TensorDesc> InTensorDescs = {
        { Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 5, 16, 96 } },
        { Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 31, 16 } }
    };
    OperationTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);
    opTest.Golden(std::bind(&GatherGolden, std::placeholders::_1, param));
    Mki::Status status1 = opTest.Run(op, InTensorDescs);
    ASSERT_EQ(status1.Ok(), true);
    atb::Status status2 = DestroyOperation(op);
    ASSERT_EQ(status2, atb::NO_ERROR);
}

TEST(TestGatherOperation, TestGather4)
{
    atb::infer::GatherParam param;
    param.axis = { 3 };
    atb::Operation *op;
    atb::Status status0 = atb::CreateOperation<atb::infer::GatherParam>(param, &op);
    ASSERT_EQ(status0, atb::NO_ERROR);
    Mki::SVector<Mki::TensorDesc> InTensorDescs = {
        { Mki::TENSOR_DTYPE_FLOAT16, Mki::TENSOR_FORMAT_ND, { 5, 16, 96, 76 } },
        { Mki::TENSOR_DTYPE_INT64, Mki::TENSOR_FORMAT_ND, { 31, 16, 84 } }
    };
    OperationTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.LongRand(0, 3);
    opTest.Golden(std::bind(&GatherGolden, std::placeholders::_1, param));
    Mki::Status status1 = opTest.Run(op, InTensorDescs);
    ASSERT_EQ(status1.Ok(), true);
    atb::Status status2 = DestroyOperation(op);
    ASSERT_EQ(status2, atb::NO_ERROR);
}
