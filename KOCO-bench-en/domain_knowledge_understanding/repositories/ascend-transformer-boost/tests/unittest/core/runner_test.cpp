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
#include "atb/utils/tensor_util.h"
#include <atb/utils/log.h>
#include "mki/utils/file_system/file_system.h"
#include "atb/runner/runner.h"
#include "test_utils/test_utils.h"

using namespace atb;
using namespace Mki;

TEST(TESTRUNNER, UNCOVEREDFUNCTIONS)
{
    setenv("ATB_PROFILING_ENABLE", "1", 1);
    atb::Runner runner("testRunner");
    atb::TensorDesc desc = {ACL_INT32, ACL_FORMAT_ND, {1, 2}};
    atb::Tensor tensor = {desc};
    atb::RunnerVariantPack variantPack = {{tensor}, {tensor}};
    atb::Context *context = nullptr;
    atb::CreateContext(&context);
    variantPack.context = dynamic_cast<atb::ContextBase *>(context);
    atb::Status st = runner.Setup(variantPack);
    EXPECT_EQ(st, 0);
    uint64_t bufferSize = runner.GetIntermediateBufferSize();
    EXPECT_EQ(bufferSize, 0);
    bufferSize = runner.GetTilingBufferSize();
    EXPECT_EQ(bufferSize, 0);
    std::vector<char> hostTilingBuffer = {'a'};
    runner.FillHostTilingBuffer(reinterpret_cast<uint8_t *>(hostTilingBuffer.data()), 1, variantPack.context);
    RunnerVariantPack runnerVariantPack;
    runnerVariantPack.inTensors.resize(1);
    runnerVariantPack.inTensors.at(0).desc = { ACL_INT32, ACL_FORMAT_ND, {1} };
    st = runner.Execute(runnerVariantPack);
    setenv("ATB_PROFILING_ENABLE", "0", 1);
}