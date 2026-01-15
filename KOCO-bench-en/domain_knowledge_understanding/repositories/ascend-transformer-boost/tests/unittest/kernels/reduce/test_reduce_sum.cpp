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
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/reduce.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "device_version_check.h"

using namespace AsdOps;
using namespace Mki;
constexpr float ATOL = 0.1;
constexpr float RTOL = 0.1;
constexpr float HALF_FLOAT_MIN = 6.2E-5;
constexpr float HALF_FLOAT_MAX = 65504;

static Status MatchAtTensorHalf(float atol, float rtol, at::Tensor &out, at::Tensor &gt)
{
    fp16_t *result = static_cast<fp16_t *>(out.storage().data_ptr().get());
    fp16_t *expect = static_cast<fp16_t *>(gt.storage().data_ptr().get());
    for (int i = 0; i < out.numel(); i++) {
        if (!Mki::Test::FloatUtil::FloatJudgeEqual(expect[i], result[i], atol, rtol)) {
            std::string msg = "pos " + std::to_string(i) +
                              ", expect: " + std::to_string(static_cast<float>(expect[i])) +
                              ", result: " + std::to_string(static_cast<float>(result[i]));
            return Status::FailStatus(-1, msg);
        }
    }
    return Status::OkStatus();
}

static Status ReduceSumGolden(float atol, float rtol, const Mki::Test::GoldenContext &context)
{
    const Tensor &inTensor = context.hostInTensors.at(0);
    const Tensor &outTensor = context.hostOutTensors.at(0);
    const Mki::Test::UtOpDesc &opDesc = context.opDesc;

    OpParam::Reduce param = AnyCast<OpParam::Reduce>(opDesc.specificParam);
    auto atInTensor = at::from_blob(inTensor.data, ToIntArrayRef(inTensor.desc.dims), at::kHalf);
    auto atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);

    auto output_tensor = atInTensor;
    std::sort(param.axis.begin(), param.axis.end());
    for (int64_t i = param.axis.size()-1; i >=0; i--) {
        MKI_LOG(INFO) << "axis.size: " <<  param.axis.at(i);
        int64_t axis = param.axis.at(i);
        output_tensor = torch::sum(output_tensor, axis, false);
    }

    return MatchAtTensorHalf(atol, rtol, atOutTensor, output_tensor);
}

TEST(TestOpReduceSum, TestSumF16Case0)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceSumGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_SUM;
    opParam.axis = {1};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {1003, 1024}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpReduceSum, TestSumF16Case1)
{
    CHECK_DEVICE_VERSION_NOT_ASCEND910A();
    CHECK_DEVICE_VERSION_NOT_ASCEND310B();
    Mki::Test::MkiOpTest opTest;
    opTest.Golden(std::bind(&ReduceSumGolden, ATOL, RTOL, std::placeholders::_1));
    opTest.FloatRand(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    OpParam::Reduce opParam;
    opParam.reduceType = OpParam::Reduce::REDUCE_SUM;
    opParam.axis = {1, 2};
    Mki::Test::UtOpDesc opDesc = {"ReduceOperation", opParam};
    SVector<TensorDesc> inTensorDesc = {{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {10, 50, 100}}};
    Status status = opTest.Run(opDesc, inTensorDesc);
    ASSERT_EQ(status.Ok(), true);
}
