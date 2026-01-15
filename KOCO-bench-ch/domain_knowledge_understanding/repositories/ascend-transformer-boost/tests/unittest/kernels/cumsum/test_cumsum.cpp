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
#include <mki/utils/assert/assert.h>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/cumsum.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "device_version_check.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

namespace {
constexpr float ATOL = 0.1;
constexpr float RTOL = 0.1;
constexpr float HALF_FLOAT_MIN = -10;
constexpr float HALF_FLOAT_MAX = 10;

static Status CumsumGolden(const Mki::Test::GoldenContext &context)
{
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;
    OpParam::Cumsum param = AnyCast<OpParam::Cumsum>(opDesc.specificParam);
    
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    int64_t axis = param.axis[0];
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
    at::Tensor atInTensorFloat = atInRefTensor1.to(at::kFloat);
    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor refOutTensorFloat = torch::cumsum(atInTensorFloat, axis);
    at::Tensor refOutTensor = refOutTensorFloat.to(at::kHalf);
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
}

TEST(TestOpCumsum, TestCumsumF16Int64Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(&CumsumGolden);
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Cumsum opParam;
    opParam.axis = {0};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCumsum, TestCumsumF16Int64Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&CumsumGolden);
    OpParam::Cumsum opParam;
    opParam.axis = {0};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2, k = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(TestOpCumsum, TestCumsumF16Int64Case2)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&CumsumGolden);
    OpParam::Cumsum opParam;
    opParam.axis = {1};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2, k = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
TEST(TestOpCumsum, TestCumsumF16Int64Case3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&CumsumGolden);
    OpParam::Cumsum opParam;
    opParam.axis = {2};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2, k = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCumsum, TestCumsumF16Int64Case5)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    opTest.Golden(&CumsumGolden);
    OpParam::Cumsum opParam;
    opParam.axis = {0};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2, k = 2, b = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief ok
 */
TEST(TestOpCumsum, TestInferShapeCumsum0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 2, n = 2, k = 2, b = 2;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Cumsum opParam;
    opParam.axis = {0};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
}

/**
 * @brief axis size wrong
 */
TEST(TestOpCumsum, TestInferShapeCumsum1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    int64_t m = 2, n = 2, k = 2, b = 2;
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, b}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Cumsum opParam;
    opParam.axis = {0, 1};
    opParam.exclusive = false;
    opParam.reverse = false;
    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), false);
}

/**
 * @brief multi-thread auto tiling test
 */
TEST(TestOpCumsum, TestCumsumF16Int64CaseAutoTiling)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(&CumsumGolden);
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest1;
    opTest1.Golden(&CumsumGolden);
    opTest1.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    Mki::Test::MkiOpTest opTest2;
    opTest2.Golden(&CumsumGolden);
    opTest2.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);

    OpParam::Cumsum opParam;
    opParam.axis = {0};
    opParam.exclusive = false;
    opParam.reverse = false;

    Mki::Test::UtOpDesc opDesc = {"CumsumOperation", opParam};
    Mki::Test::UtOpDesc opDesc1 = {"CumsumOperation", opParam};
    Mki::Test::UtOpDesc opDesc2 = {"CumsumOperation", opParam};
    int64_t m = 2, n = 2;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    SVector<TensorDesc> inTensorDesc1 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};
    SVector<TensorDesc> inTensorDesc2 = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n}}};

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
