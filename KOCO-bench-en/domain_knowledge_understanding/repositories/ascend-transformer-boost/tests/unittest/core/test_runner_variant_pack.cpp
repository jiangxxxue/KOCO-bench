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
#include <securec.h>
#include <gtest/gtest.h>
#include <atb/utils/log.h>
#include <atb/utils/runner_variant_pack.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
 
using namespace atb;
using namespace Mki;

TEST(TestRunnerVariantPack, ToString)
{
    const int tensorCount = 1;
    const uint32_t totalLength = 16384;
    const uint32_t tileNum = 8;
    const uint32_t workspace = 4096;
    const uint32_t intermediate = 2048;

    atb::RunnerVariantPack runnerVariantPack;

    atb::TensorDesc desc;
    desc.dtype = ACL_FLOAT16;
    desc.format = ACL_FORMAT_ND;
    desc.shape.dimNum = 2;
    desc.shape.dims[0] = 1024; // 1024 dims
    desc.shape.dims[1] = 1024; // 1024 dims

    runnerVariantPack.inTensors.resize(tensorCount);
    atb::Tensor &inTensor0 = runnerVariantPack.inTensors.at(0);
    inTensor0.desc = desc;

    runnerVariantPack.outTensors.resize(tensorCount);
    atb::Tensor &outTensor0 = runnerVariantPack.outTensors.at(0);
    outTensor0.desc = desc;

    uint64_t tilingBufferSize = 2 * sizeof(uint32_t); // 2 tilling data
    uint8_t *tilingBuffer = static_cast<uint8_t *>(malloc(tilingBufferSize));
    memcpy(tilingBuffer, &totalLength, sizeof(uint32_t));
    memcpy(tilingBuffer + sizeof(uint32_t), &tileNum, sizeof(uint32_t));
    runnerVariantPack.tilingBufferSize = tilingBufferSize;
    runnerVariantPack.tilingBuffer = tilingBuffer;

    uint64_t workspaceBufferSize = sizeof(uint32_t);
    uint8_t *workspaceBuffer = static_cast<uint8_t *>(malloc(workspaceBufferSize));
    memcpy(workspaceBuffer, &workspace, sizeof(uint32_t));
    runnerVariantPack.workspaceBufferSize = workspaceBufferSize;
    runnerVariantPack.workspaceBuffer = workspaceBuffer;

    uint64_t intermediateBufferSize = sizeof(uint32_t);
    uint8_t *intermediateBuffer = static_cast<uint8_t *>(malloc(intermediateBufferSize));
    memcpy(intermediateBuffer, &intermediate, sizeof(uint32_t));
    runnerVariantPack.intermediateBufferSize = intermediateBufferSize;
    runnerVariantPack.intermediateBuffer = intermediateBuffer;

    atb::Context *context = nullptr;
    int ret = atb::CreateContext(&context);
    ASSERT_EQ(ret, 0);
    ASSERT_NE(context, nullptr);
    runnerVariantPack.context = dynamic_cast<ContextBase*>(context);

    std::string runnerVariantPackString = runnerVariantPack.ToString();
    ATB_LOG(INFO) << runnerVariantPackString;
    ret = atb::DestroyContext(context);
    ASSERT_EQ(ret, 0);
    free(tilingBuffer);
    free(workspaceBuffer);
    free(intermediateBuffer);
}