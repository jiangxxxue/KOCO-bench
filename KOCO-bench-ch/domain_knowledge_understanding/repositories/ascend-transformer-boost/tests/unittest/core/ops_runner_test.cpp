/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <string>
#include <string.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "atb/infer_op_params.h"
#include "atb/utils/tensor_util.h"
#include <atb/utils/log.h>
#include "mki/utils/file_system/file_system.h"
#include "atb/runner/runner.h"
#include "test_utils/context/memory_context.h"
#include "atb/utils.h"
#include "test_utils/operation_test.h"
#include "atb/operation.h"
#include <torch/torch.h>
#include "test_utils/test_common.h"
#include "test_utils/test_utils.h"
#include "atb/utils/singleton.h"

using namespace atb;
using namespace Mki;
constexpr float ATOL = 0.0001;
constexpr float RTOL = 0.0001;

TEST(TESTOPSRUNNER, UNCOVEREDFUNCTIONS)
{
    setenv("ATB_PROFILING_ENABLE", "1", 1);
    atb::infer::FillParam param = { true, { 1.0f }, { 1, 2, 3 } };
    atb::Operation *op;
    atb::Status status = atb::CreateOperation<atb::infer::FillParam>(param, &op);
    EXPECT_EQ(status, 0);
    VariantPack variantPack;
    variantPack.inTensors.resize(op->GetInputNum());
    variantPack.outTensors.resize(op->GetOutputNum());
    variantPack.inTensors.at(0).desc = { ACL_INT32, ACL_FORMAT_ND, { { 1, 2 }, 2 } };

    variantPack.inTensors.at(0).dataSize = atb::Utils::GetTensorSize(variantPack.inTensors.at(0));
    variantPack.inTensors.at(0).deviceData = malloc(variantPack.inTensors.at(0).dataSize);

    variantPack.inTensors.at(1).desc = { ACL_INT8, ACL_FORMAT_ND, { { 1, 2 }, 2 } };
    variantPack.inTensors.at(1).dataSize = atb::Utils::GetTensorSize(variantPack.inTensors.at(1));
    variantPack.inTensors.at(1).deviceData = malloc(variantPack.inTensors.at(1).dataSize);
    variantPack.outTensors.at(0) = variantPack.inTensors.at(0);

    atb::Context *context = CreateContextAndExecuteStream();

    uint64_t wpsize = 0;
    atb::Status st = op->Setup(variantPack, wpsize, context);
    EXPECT_EQ(st, atb::NO_ERROR);

    // dataSize is invalid
    variantPack.inTensors.at(0).desc.shape.dims[0] = 0;
    st = op->Setup(variantPack, wpsize, context);
    EXPECT_EQ(st, atb::ERROR_INVALID_TENSOR_SIZE);
    variantPack.inTensors.at(0) = variantPack.outTensors.at(0);

    void *workSpace = nullptr;
    if (wpsize > 0) {
        workSpace = GetSingleton<atb::MemoryContext>().GetWorkspaceBuffer(wpsize);
    }

    // deviceData&hostData is null
    st = op->Execute(variantPack, (uint8_t *)workSpace, wpsize, context);
    EXPECT_EQ(st, ERROR_INVALID_PARAM);
    setenv("ATB_PROFILING_ENABLE", "0", 1);
    DestroyOperation(op);
    DestroyContext(context);
}


