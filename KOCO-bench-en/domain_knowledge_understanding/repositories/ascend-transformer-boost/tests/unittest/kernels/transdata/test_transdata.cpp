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
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/log/log.h>
#include "asdops/params/transdata.h"
#include "kernels/transdata/tiling/transdata_tiling.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;
using namespace torch::indexing;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr float EXTENT_OF_ERROR = 0.001;

constexpr size_t IDX_0 = 0;
constexpr size_t IDX_1 = 1;
constexpr size_t IDX_2 = 2;
constexpr size_t IDX_3 = 3;
constexpr int64_t DEFAULT_ALIGN = 16;

int64_t RoundUp(const int64_t val, const int64_t align = DEFAULT_ALIGN)
{
    if (align == 0) {
        return -1;
    }
    return (val + align - 1) / align * align;
}

void TransdataNdToNzGolden(at::Tensor &inTensor, const SVector<int64_t> &inDims, at::Tensor &outTensor)
{
    // Infer shape
    std::vector<int64_t> auxDims{0, 0, 0, 0};
    auxDims[IDX_0] = inDims[IDX_0];
    auxDims[IDX_1] = RoundUp(inDims[IDX_1]);
    auxDims[IDX_2] = RoundUp(inDims[IDX_2]) / DEFAULT_ALIGN;
    auxDims[IDX_3] = DEFAULT_ALIGN;
    std::vector<int64_t> padDims{0, 0, 0, 0};
    padDims[IDX_1] = RoundUp(inDims[IDX_2]) - inDims[IDX_2];
    padDims[IDX_3] = RoundUp(inDims[IDX_1]) - inDims[IDX_1];
    outTensor = transpose(reshape(constant_pad_nd(inTensor, ToIntArrayRef(padDims)), ToIntArrayRef(auxDims)),
        IDX_1, IDX_2);
    return;
}

Status TransdataNdToNzCompare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutRefTensor;
    TransdataNdToNzGolden(atInTensor, inTensor.desc.dims, atOutRefTensor);
    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}

void TransdataNzToNdGolden(at::Tensor &inTensor, const SVector<int64_t> &inDims, at::Tensor &outTensor,
    const SVector<int64_t> &outDims)
{
    // Infer shape
    std::vector<int64_t> auxDims{0, 0, 0};
    auxDims[IDX_0] = inDims[IDX_0];
    auxDims[IDX_1] = inDims[IDX_2];
    auxDims[IDX_2] = inDims[IDX_1] * inDims[IDX_3];
    outTensor = at::reshape(at::transpose(inTensor, IDX_1, IDX_2), ToIntArrayRef(auxDims))
                    .index({Slice(), Slice(None, outDims[IDX_1]), Slice(None, outDims[IDX_2])});
    return;
}

Status TransdataNzToNdCompare(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    at::Tensor atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor atOutRefTensor;
    TransdataNzToNdGolden(atInTensor, inTensor.desc.dims, atOutRefTensor, outTensor.desc.dims);
    if (!at::allclose(atOutRefTensor, atOutTensor, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}

}

TEST(TestOpTransdataNdToNz, TransdataNdToNzCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 23015, 12573};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNdToNz, TransdataNdToNzCase1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 8, 32};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNdToNz, TransdataNdToNzCase2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 8, 255};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNdToNz, TransdataNdToNzCase3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 30, 8};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNdToNz, TransdataNdToNzCase4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 23015, 12573};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestOpTransdataNzToNd, TransdataNzToNdCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNzToNdCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 2, 32, 16};
    SVector<int64_t> outDims{1, 32, 32};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNzToNd, TransdataNzToNdCase1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNzToNdCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 2, 16, 16};
    SVector<int64_t> outDims{1, 10, 20};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNzToNd, TransdataNzToNdCase2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNzToNdCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{2, 2, 32, 16};
    SVector<int64_t> outDims{2, 32, 32};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNzToNd, TransdataNzToNdCase3)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNzToNdCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{2, 2, 16, 16};
    SVector<int64_t> outDims{2, 10, 20};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpTransdataNzToNd, TransdataNzToNdCase4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNzToNdCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;

    Status status = opTest.Run(opDesc, inTensorDescs);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNzToNd, TestGetBestKernelTransdata0)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
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
TEST(TestOpTransdataNdToNz, TestGetBestKernelTransdata1)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpTransdataNzToNd, TestGetBestKernelTransdata2)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief Transdata wrong
 */
TEST(TestOpTransdataNzToNd, TestGetBestKernelTransdata3)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::UNDEFINED;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNzToNd, TestInferShapeTransdata0)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNdToNz, TestInferShapeTransdata1)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief transdataType wrong
 */
TEST(TestOpTransdataNzToNd, TestInferShapeTransdata2)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::UNDEFINED;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNzToNd, TestCanSupportTransdata0)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNzToNdKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNdToNz, TestCanSupportTransdata1)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNdToNzKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpTransdataNzToNd, TestCanSupportTransdata2)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNzToNdKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpTransdataNzToNd, TestCanSupportTransdata3)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNzToNdKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpTransdataNzToNd, TestCanSupportTransdata4)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNzToNdKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor format wrong
 */
TEST(TestOpTransdataNzToNd, TestCanSupportTransdata5)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 237, 2576, 16};
    SVector<int64_t> outDims{1, 2573, 3792};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::FRACTAL_NZ_TO_ND;
    param.outCrops = {outDims[IDX_1], outDims[IDX_2]};
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNzToNdKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor format wrong
 */
TEST(TestOpTransdataNdToNz, TestCanSupportTransdata6)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_FRACTAL_NZ, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNdToNzKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpTransdataNdToNz, TestCanSupportTransdata7)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNdToNzKernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpTransdataNdToNz, TransdataCommonTiling)
{
    LaunchParam launchParam;
    SVector<int64_t> inDims{1, 23015, 12573};
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims}});
    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;
    launchParam.SetParam(opDesc.specificParam);
 
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("TransdataNdToNzKernel"));
    ASSERT_NE(kernel, nullptr);
    status = kernel->Init(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpTransdataNdToNz, TestTransdataCaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(std::bind(&TransdataNdToNzCompare, ATOL, RTOL, std::placeholders::_1));
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    SVector<TensorDesc> inTensorDescs;
    Mki::SVector<int64_t> inDims{1, 30, 8};
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    SVector<TensorDesc> inTensorDescs1;
    inTensorDescs1.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    SVector<TensorDesc> inTensorDescs2;
    inTensorDescs2.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, inDims});

    Mki::Test::UtOpDesc opDesc{};
    opDesc.opName = "TransdataOperation";
    AsdOps::OpParam::Transdata param{};
    param.transdataType = AsdOps::OpParam::Transdata::ND_TO_FRACTAL_NZ;
    opDesc.specificParam = param;

    Mki::Test::UtOpDesc opDesc1{};
    opDesc1.opName = "TransdataOperation";
    opDesc1.specificParam = param;

    Mki::Test::UtOpDesc opDesc2{};
    opDesc2.opName = "TransdataOperation";
    opDesc2.specificParam = param;

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