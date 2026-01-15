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
#include <future>
#include "test_common.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "op_desc_json.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr float HALF_FLOAT_MIN = 1;
constexpr float HALF_FLOAT_MAX = 65504;
constexpr int64_t RAND_INT64_MIN = 1;
constexpr int64_t RAND_INT64_MAX = 65504;

Status AsStridedF16Golden(OpParam::AsStrided &param, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensorX = context.hostInTensors.at(0);
    at::Tensor atInRefTensorX = at::from_blob(inTensorX.data, ToIntArrayRef(inTensorX.desc.dims), at::kHalf);

    at::Tensor atInRefTensorSize = at::from_blob(param.size.data(), {param.size.size()}, at::kLong);
    at::Tensor atInRefTensorStride = at::from_blob(param.stride.data(), {param.stride.size()}, at::kLong);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    at::Tensor refOutTensor = as_strided(
        atInRefTensorX,
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorSize.data_ptr<int64_t>(),
                                           atInRefTensorSize.data_ptr<int64_t>() + atInRefTensorSize.numel())),
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorStride.data_ptr<int64_t>(),
                                           atInRefTensorStride.data_ptr<int64_t>() + atInRefTensorStride.numel())),
        0).contiguous();
    fp16_t *atOutArray = (fp16_t *)atOutTensor.storage().data_ptr().get();
    fp16_t *atRefOutArray = (fp16_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            MKI_LOG(ERROR) << "golden: " << expect << "  actual: " << actual;
            return Status::FailStatus(-1, "AsStridedGolden Failed");
        }
    }
    return Status::OkStatus();
}

Status AsStridedF32Golden(OpParam::AsStrided &param, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensorX = context.hostInTensors.at(0);
    at::Tensor atInRefTensorX = at::from_blob(inTensorX.data, ToIntArrayRef(inTensorX.desc.dims), at::kFloat);

    at::Tensor atInRefTensorSize = at::from_blob(param.size.data(), {param.size.size()}, at::kLong);
    at::Tensor atInRefTensorStride = at::from_blob(param.stride.data(), {param.stride.size()}, at::kLong);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kFloat);

    at::Tensor refOutTensor = as_strided(
        atInRefTensorX,
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorSize.data_ptr<int64_t>(),
                                           atInRefTensorSize.data_ptr<int64_t>() + atInRefTensorSize.numel())),
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorStride.data_ptr<int64_t>(),
                                           atInRefTensorStride.data_ptr<int64_t>() + atInRefTensorStride.numel())),
        0).contiguous();
    float *atOutArray = (float *)atOutTensor.storage().data_ptr().get();
    float *atRefOutArray = (float *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            MKI_LOG(ERROR) << "golden: " << expect << "  actual: " << actual;
            return Status::FailStatus(-1, "AsStridedGolden Failed");
        }
    }
    return Status::OkStatus();
}

Status AsStridedInt64Golden(OpParam::AsStrided &param, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensorX = context.hostInTensors.at(0);
    at::Tensor atInRefTensorX = at::from_blob(inTensorX.data, ToIntArrayRef(inTensorX.desc.dims), at::kLong);

    at::Tensor atInRefTensorSize = at::from_blob(param.size.data(), {param.size.size()}, at::kLong);
    at::Tensor atInRefTensorStride = at::from_blob(param.stride.data(), {param.stride.size()}, at::kLong);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    at::Tensor refOutTensor = as_strided(
        atInRefTensorX,
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorSize.data_ptr<int64_t>(),
                                           atInRefTensorSize.data_ptr<int64_t>() + atInRefTensorSize.numel())),
        ToIntArrayRef(std::vector<int64_t>(atInRefTensorStride.data_ptr<int64_t>(),
                                           atInRefTensorStride.data_ptr<int64_t>() + atInRefTensorStride.numel())),
        param.offset.at(0)).contiguous();
    int64_t *atOutArray = (int64_t *)atOutTensor.storage().data_ptr().get();
    int64_t *atRefOutArray = (int64_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        int64_t expect = atRefOutArray[i];
        int64_t actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            MKI_LOG(ERROR) << "golden: " << expect << "  actual: " << actual;
            return Status::FailStatus(-1, "AsStridedGolden Failed");
        }
    }
    return Status::OkStatus();
}

// add more case
TEST(TestOpAsStrided, asstridedTest1)
{
    Mki::Test::MkiOpTest opTest;
    float tmp[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    fp16_t inputX[24];
    for (int i = 0; i < 24; i++) {
        inputX[i] = fp16_t(tmp[i]);
    }

    OpParam::AsStrided opParam;
    opParam.size = {3, 2, 4};
    opParam.stride = {4, 12, 1};
    opParam.offset = {0};
    opTest.Golden(std::bind(AsStridedF16Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    int64_t m = 2;
    int64_t n = 3;
    int64_t k = 4;
    Mki::SVector<Tensor> inTensorList = {
        {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}, static_cast<void *>(inputX), sizeof(inputX)}
    };
    Status status = opTest.Run(opDesc, inTensorList);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpAsStrided, asstridedTest2)
{
    Mki::Test::MkiOpTest opTest;
    float tmp[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    fp16_t inputX[24];
    for (int i = 0; i < 24; i++) {
        inputX[i] = fp16_t(tmp[i]);
    }

    OpParam::AsStrided opParam;
    opParam.size = {3, 2, 1};
    opParam.stride = {1, 12, 0};
    opParam.offset = {0};
    opTest.Golden(std::bind(AsStridedF16Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    int64_t m = 2;
    int64_t n = 3;
    Mki::SVector<Tensor> inTensorList = {
        {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, m, m}}, static_cast<void *>(inputX), sizeof(inputX)}
    };
    Status status = opTest.Run(opDesc, inTensorList);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpAsStrided, asstridedTest3)
{
    Mki::Test::MkiOpTest opTest;
    float tmp[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    fp16_t inputX[24];
    for (int i = 0; i < 24; i++) {
        inputX[i] = fp16_t(tmp[i]);
    }

    OpParam::AsStrided opParam;
    opParam.size = {1, 2, 4};
    opParam.stride = {1, 12, 0};
    opParam.offset = {0};
    opTest.Golden(std::bind(AsStridedF16Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    int64_t m = 2;
    int64_t n = 3;
    Mki::SVector<Tensor> inTensorList = {
        {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, m, m}}, static_cast<void *>(inputX), sizeof(inputX)}
    };
    Status status = opTest.Run(opDesc, inTensorList);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpAsStrided, asstridedTest4)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(RAND_INT64_MIN, RAND_INT64_MAX);
    int64_t m = 10, n = 12, k = 14;
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    opTest.Golden(std::bind(AsStridedInt64Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpAsStrided, asstridedTest5)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(RAND_INT64_MIN, RAND_INT64_MAX);
    int64_t m = 10, n = 12, k = 14;
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {2 * k, n * 2 * k, 1};
    opParam.offset = {k};
    opTest.Golden(std::bind(AsStridedInt64Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    Status status = opTest.Run(opDesc, {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, 2 * k}});
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpAsStrided, asstridedTest6)
{
    CHECK_DEVICE_VERSION_ASCEND910B();
    Mki::Test::MkiOpTest opTest;
    float inputX[24] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    OpParam::AsStrided opParam;
    opParam.size = {1, 2, 4};
    opParam.stride = {1, 12, 0};
    opParam.offset = {0};
    opTest.Golden(std::bind(AsStridedF32Golden, opParam, std::placeholders::_1));
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    int64_t m = 2;
    int64_t n = 3;
    Mki::SVector<Tensor> inTensorList = {
        {{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, m, m}}, static_cast<void *>(inputX), sizeof(inputX)}
    };
    Status status = opTest.Run(opDesc, inTensorList);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpAsStrided, TestCanSupport0)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("AsStridedF16Int64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpAsStrided, TestCanSupport1)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("AsStridedF16Int64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpAsStrided, TestCanSupport2)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("AsStridedF16Int64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpAsStrided, TestCanSupport3)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("AsStridedF16Int64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpAsStrided, TestCanSupport4)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("AsStridedInt64Int64Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

TEST(TestOpAsStrided, TestInferShapeAsStrided0)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpAsStrided, TestInferShapeAsStrided1)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_BF16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief inTensor format wrong
 */
TEST(TestOpAsStrided, TestInferShapeAsStrided2)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_NHWC, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief OpParam wrong
 */
TEST(TestOpAsStrided, TestInferShapeAsStrided3)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Activation opParam;
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief OpParam size dim num and stride dim num are different
 */
TEST(TestOpAsStrided, TestInferShapeAsStrided4)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief OpParam offset dim is invalid
 */
TEST(TestOpAsStrided, TestInferShapeAsStrided5)
{
    LaunchParam launchParam;
    int64_t m = 10, n = 12, k = 14;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0, 1};
    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}
/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpAsStrided, TestAsStridedCaseAutoTiling)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(RAND_INT64_MIN, RAND_INT64_MAX);

    Mki::Test::MkiOpTest opTest1;
    opTest1.LongRand(RAND_INT64_MIN, RAND_INT64_MAX);

    Mki::Test::MkiOpTest opTest2;
    opTest2.LongRand(RAND_INT64_MIN, RAND_INT64_MAX);

    int64_t m = 10, n = 12, k = 14;
    OpParam::AsStrided opParam;
    opParam.size = {n, m, k};
    opParam.stride = {k, n * k, 1};
    opParam.offset = {0};

    opTest.Golden(std::bind(AsStridedInt64Golden, opParam, std::placeholders::_1));
    opTest1.Golden(std::bind(AsStridedInt64Golden, opParam, std::placeholders::_1));
    opTest2.Golden(std::bind(AsStridedInt64Golden, opParam, std::placeholders::_1));

    Mki::Test::UtOpDesc opDesc = {"AsStridedOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"AsStridedOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"AsStridedOperation", opParam};

    auto test1 = [&]() {
        return opTest.Run(opDesc, {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k}});
    };

    auto test2 = [&]() {
        return opTest1.Run(opDesc1, {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k}});
    };

    auto test3 = [&]() {
        return opTest2.Run(opDesc2, {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k}});
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