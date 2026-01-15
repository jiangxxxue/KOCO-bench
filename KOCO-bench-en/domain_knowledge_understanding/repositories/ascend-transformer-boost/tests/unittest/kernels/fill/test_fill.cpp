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
#include <random>
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "kernels/fill/tiling/fill_tiling.h"
#include "test_utils/float_util.h"
#include "test_utils/op_test.h"
#include "test_utils/golden.h"
#include "test_common.h"
#include "op_desc_json.h"

using namespace AsdOps;
using namespace Mki;

static constexpr float ATOL = 0.0001;
static constexpr float RTOL = 0.0001;
static constexpr float HALF_FLOAT_MIN = 1;
static constexpr float HALF_FLOAT_MAX = 65504;

Status FillF16Golden(const Mki::Test::GoldenContext &context)
{
    auto param = AnyCast<OpParam::Fill>(context.opDesc.specificParam);

    const Tensor outTensor = context.hostOutTensors.at(0);
    at::Tensor atOutTensor = at::from_blob(outTensor.data, ToIntArrayRef(outTensor.desc.dims), at::kHalf);
    fp16_t *atOutArray = (fp16_t *)atOutTensor.storage().data_ptr().get();

    for (int i = 0; i < outTensor.Numel(); i++) {
        float expect = static_cast<float>(static_cast<fp16_t>(param.value[0])); // golden
        float actual = static_cast<float>(atOutArray[i]);
        bool judge = std::abs(expect - actual) <= (ATOL + RTOL * std::abs(actual));
        if (!judge) {
            return Status::FailStatus(1, "unequal");
        }
    }

    return Status::OkStatus();
}

void GeneteFloatRandVal(SVector<float> &value)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(HALF_FLOAT_MIN, HALF_FLOAT_MAX);
    value.push_back(static_cast<float>(distr(eng)));
}

/**
 * @brief ok
 */
TEST(TestOpFILL, TestCanSupportFill0)
{
    LaunchParam launchParam;
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Fill opParam;
    opParam.value = {0};
    opParam.outDim = {1, 2, 3, 4};
    opParam.outTensorType = TENSOR_DTYPE_FLOAT16;
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->CanSupport(launchParam), true);
}

/**
 * @brief ok
 */
TEST(TestOpFILL, FillTiling)
{
    LaunchParam launchParam;
    launchParam.AddOutTensor({{TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {5, 5}}});
    OpParam::Fill opParam;
    opParam.value = {0};
    opParam.outDim = {1, 2, 3, 4};
    opParam.outTensorType = TENSOR_DTYPE_FLOAT16;
    Mki::Test::UtOpDesc opDesc = {"FillOperation", opParam};
    launchParam.SetParam(opDesc.specificParam);
 
    Mki::Operation *op = Mki::AutoGen::GetOpByName(opDesc.opName);
    Status status = op->InferShape(launchParam);
    ASSERT_EQ(status.Ok(), true);
    auto kernel = std::unique_ptr<Mki::Kernel>(op->GetKernelByName("FillF16Kernel"));
    ASSERT_NE(kernel, nullptr);
    status = kernel->Init(launchParam);
    ASSERT_EQ(status.Ok(), true);
}