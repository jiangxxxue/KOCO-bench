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
#include <securec.h>
#include <vector>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;
constexpr float HALF_FLOAT_MIN = -60000;
constexpr float HALF_FLOAT_MAX = 60000;
constexpr int8_t RAND_INT8_MIN = 0;
constexpr int8_t RAND_INT8_MAX = 1;

static Status MaskedFillF16BroadcastGolden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kBool);
    auto param = AnyCast<OpParam::Fill>(context.opDesc.specificParam);

    SVector<fp16_t> atInRefTensor3Data;
    for (size_t i = 0; i < param.value.size(); i++) {
        atInRefTensor3Data.push_back(static_cast<fp16_t>(param.value.at(i)));
    }
    at::Tensor atInRefTensor3 = at::from_blob(atInRefTensor3Data.data(), {atInRefTensor3Data.size()}, at::kHalf);
    c10::Half value = *(c10::Half *)atInRefTensor3.data_ptr();
    at::Scalar atInScalar3(value);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor refOutTensor = torch::masked_fill(atInRefTensor1, atInRefTensor2, atInScalar3);
    fp16_t *atOutArray = (fp16_t *)atOutTensor.storage().data_ptr().get();
    fp16_t *atRefOutArray = (fp16_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

static Status MaskedFillInt32Golden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kInt);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kBool);
    auto param = AnyCast<OpParam::Fill>(context.opDesc.specificParam);

    SVector<int32_t> atInRefTensor3Data;
    for (size_t i = 0; i < param.value.size(); i++) {
        MKI_LOG(INFO) << "param.value BEFORE:" << param.value[0];
        atInRefTensor3Data.push_back(static_cast<int32_t>(param.value.at(i)));
        MKI_LOG(INFO) << "param.value AFTER:" << static_cast<int32_t>(param.value[0]);
    }
    at::Tensor atInRefTensor3 = at::from_blob(atInRefTensor3Data.data(), {atInRefTensor3Data.size()}, at::kInt);
    int32_t value = *(int32_t *)atInRefTensor3.data_ptr();
    at::Scalar atInScalar3(value);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);
    at::Tensor refOutTensor = torch::masked_fill(atInRefTensor1, atInRefTensor2, atInScalar3);
    int32_t *atOutArray = (int32_t *)atOutTensor.storage().data_ptr().get();
    int32_t *atRefOutArray = (int32_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        int32_t expect = atRefOutArray[i];
        int32_t actual = atOutArray[i];
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }
    return Status::OkStatus();
}

TEST(TestOpFill, MaskedFillInt32)
{
    Mki::Test::MkiOpTest opTest;
    opTest.IntRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Int8Rand(RAND_INT8_MIN, RAND_INT8_MAX);
    opTest.Golden(&MaskedFillInt32Golden);
    SVector<float> value = {12345};
    OpParam::Fill opParam;
    opParam.withMask = true;
    opParam.value = value;
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    int64_t m = 4, n = 5, k = 3, b = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpFill, MaskedFillCase0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Int8Rand(RAND_INT8_MIN, RAND_INT8_MAX);
    opTest.Golden(&MaskedFillF16BroadcastGolden);
    SVector<float> value = {-10000};
    OpParam::Fill opParam;
    opParam.withMask = true;
    opParam.value = value;
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    int64_t m = 4, n = 5, k = 3, b = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}},
                                        {TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpFill, TestCanSupportMaskedFill0)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief inPutNum wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill1)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief outPutNum wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill2)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief withMask wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill3)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {false, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill4)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill5)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpFill, TestCanSupportMaskedFill6)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("MaskedFillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), false);
}

/**
 * @brief ok
 */
TEST(TestOpFill, TestInferShapeFill0)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief withMask wrong
 */
TEST(TestOpFill, TestInferShapeFill1)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {false, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief ok
 */
TEST(TestOpFill, TestGetBestKernelMaskedFill0)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {4, 5, 5, 2}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_NE(kernel, nullptr);
}

/**
 * @brief inTensor dtype wrong
 */
TEST(TestOpFill, TestGetBestKernelMaskedFill1)
{
    LaunchParam launchParam;
    int64_t m = 4, n = 5, k = 5, b = 2;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_INT8, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {4, 5, 5, 2}}});
    SVector<float> value = {200};
    OpParam::Fill opParam = {true, value, {}};
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(launchParam));
    ASSERT_EQ(kernel, nullptr);
}
