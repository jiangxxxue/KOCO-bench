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
#include <mki/utils/log/log.h>
#include "asdops/params/reduce.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr int32_t RAND_INT32_MIN = 0;
constexpr int32_t RAND_INT32_MAX = 100000;

static Status MatchAtTensorInt32(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    int32_t *result = static_cast<int32_t *>(out.storage().data_ptr().get());
    int32_t *expect = static_cast<int32_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect[i], result[i], atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) + ", expect: " + std::to_string(expect[i]) +
                              ", result: " + std::to_string(result[i]);
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

static Status ReduceMaxGolden(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;

    OpParam::Reduce param = AnyCast<OpParam::Reduce>(opDesc.specificParam);
    auto atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kInt);
    auto atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kInt);

    auto output_tensor = atInTensor;
    std::sort(param.axis.begin(), param.axis.end());
    for (int64_t i = param.axis.size()-1; i >=0; i--) {
        MKI_LOG(INFO) << "axis.size: " <<  param.axis.at(i);
        int64_t axis = param.axis.at(i);
        output_tensor = std::get<0>(torch::max(output_tensor, axis, false));
    }

    return MatchAtTensorInt32(atol, rtol, atOutTensor, output_tensor);
}

TEST(TestOpReduceMax, TestMaxInt32Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceMaxGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(RAND_INT32_MIN, RAND_INT32_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_MAX;
    opParam.axis = {1};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {11008, 4096}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReduceMax, TestMaxInt32Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceMaxGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(RAND_INT32_MIN, RAND_INT32_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_MAX;
    opParam.axis = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {2, 3, 4}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReduceMax, TestMaxInt32Case2) // axis输入错误，导致inferShape失败
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceMaxGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(RAND_INT32_MIN, RAND_INT32_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_MAX;
    opParam.axis = {1, 3};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {100, 300, 400}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), false);
}

TEST(TestOpReduceMax, TestMaxInt32Case3)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceMaxGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.IntRand(RAND_INT32_MIN, RAND_INT32_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_MAX;
    opParam.axis = {1, 3, 5, 7};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};

    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {10, 3, 4, 5, 6, 7, 8, 9}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
