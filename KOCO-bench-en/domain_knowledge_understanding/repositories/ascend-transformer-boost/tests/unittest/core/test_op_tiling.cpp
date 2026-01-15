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
#include <cstring>
#include <cstdlib>
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include "atb/utils/tensor_util.h"
#include "test_utils/operation_test.h"
#include "atb/utils/config.h"
#include "atb_torch/utils/utils.h"
#include "atb/utils/current_op_tiling.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "test_utils/test_common.h"
#include <cpp-stub/src/stub.h>
#include "atb/context/context_base.h"
using namespace atb;
using namespace Mki;


bool IsLaunchKernelWithTilingStub()
{
    return false;
}

TEST(TestOpTiling, GetCurrentOpTiling)
{
    Stub stub;
    stub.set(ADDR(ContextBase, GetLaunchWithTilingStatus), IsLaunchKernelWithTilingStub);
    uint64_t tilingBufferSize = 0;
    void *tilingDeviceBuffer = nullptr;
    atb::infer::ElewiseParam param;
    param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Operation *op;
    atb::CreateOperation<atb::infer::ElewiseParam>(param, &op);
    Mki::SVector<Mki::TensorDesc> inTensorDescs = {
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1, 2}},
        {Mki::TENSOR_DTYPE_FLOAT, Mki::TENSOR_FORMAT_ND, {1, 2}}};
    OperationTest opTest;
    Mki::Status status = opTest.Run(op, inTensorDescs);
    EXPECT_EQ(status.Ok(), true);
    atb::GetCurrentOpTiling(tilingDeviceBuffer, tilingBufferSize);
    DestroyOperation(op);
    EXPECT_NE(tilingDeviceBuffer, nullptr);
    EXPECT_NE(tilingBufferSize, 0);
}
