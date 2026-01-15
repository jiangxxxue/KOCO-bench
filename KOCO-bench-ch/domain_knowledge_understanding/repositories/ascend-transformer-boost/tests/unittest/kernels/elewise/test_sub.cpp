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
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

static constexpr float ATOL = 0.0001;
static constexpr float RTOL = 0.0001;
static constexpr float HALF_FLOAT_MIN = 1;
static constexpr float HALF_FLOAT_MAX = 65504;
constexpr int64_t LONG_MIN_VALUE = -1000000000;
constexpr int64_t LONG_MAX_VALUE = 1000000000;
using namespace AsdOps;
using namespace Mki;

Status SubF16Golden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kHalf);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kHalf);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    at::Tensor refOutTensor = atInRefTensor1.sub(atInRefTensor2);
    fp16_t *atOutArray = (fp16_t *)atOutTensor.storage().data_ptr().get();
    fp16_t *atRefOutArray = (fp16_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(atRefOutArray[i]);
        float actual = static_cast<float>(atOutArray[i]);
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect, actual, ATOL, RTOL)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}

Status SubInt64Golden(const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor1 = context.hostInTensors.at(0);
    at::Tensor atInRefTensor1 = at::from_blob(inTensor1.data, ToIntArrayRef(inTensor1.desc.dims), at::kLong);
    const Tensor &inTensor2 = context.hostInTensors.at(1);
    at::Tensor atInRefTensor2 = at::from_blob(inTensor2.data, ToIntArrayRef(inTensor2.desc.dims), at::kLong);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kLong);
    at::Tensor refOutTensor = atInRefTensor1.sub(atInRefTensor2);
    int64_t *atOutArray = (int64_t *)atOutTensor.storage().data_ptr().get();
    int64_t *atRefOutArray = (int64_t *)refOutTensor.storage().data_ptr().get(); // golden

    for (int i = 0; i < outTensor.Numel(); i++) {
        int64_t expect = static_cast<int64_t>(atRefOutArray[i]);
        int64_t actual = static_cast<int64_t>(atOutArray[i]);
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect, actual, ATOL, RTOL)) {
            return Status::FailStatus(-1, "float judge not equal");
        }
    }
    return Status::OkStatus();
}

TEST(TestOpElewiseSub, SubF16Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX / 2);
    opTest.Golden(&SubF16Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseSub, SubF16Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX / 2);
    opTest.Golden(&SubF16Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, n, k, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseSub, SubF16Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX / 2);
    opTest.Golden(&SubF16Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t n = 128, k = 4096;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1, n, k}},
                                        {TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {n, 1, k}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseSub, TestCanSupport0)
{
    LaunchParam launchParam;
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddInTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Elewise opParam = {};
    opParam.elewiseType = OpParam::Elewise::ELEWISE_SUB;
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("SubF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

TEST(TestOpElewiseSub, SubInt64Case0)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(&SubInt64Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, 1, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseSub, SubInt64Case1)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(&SubInt64Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t m = 16, n = 40, k = 20, b = 10;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {m, n, k, k, b}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, n, k, 1, b}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpElewiseSub, SubInt64Case2)
{
    Mki::Test::MkiOpTest opTest;
    opTest.LongRand(LONG_MIN_VALUE, LONG_MAX_VALUE);
    opTest.Golden(&SubInt64Golden);
    OpParam::Elewise opParam = {OpParam::Elewise::ELEWISE_SUB};
    Mki::Test::UtOpDesc opDesc = {"ElewiseOperation", opParam};
    int64_t n = 128, k = 4096;
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {1, n, k}},
                                        {TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {n, 1, k}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}