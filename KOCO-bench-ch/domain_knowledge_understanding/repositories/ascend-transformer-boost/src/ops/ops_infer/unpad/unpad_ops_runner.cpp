/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unpad_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/runner_util.h"
#include "atb/utils/log.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 3;
UnpadOpsRunner::UnpadOpsRunner(const infer::UnpadParam &param)
    : OpsRunner("UnpadOpsRunner", RUNNER_TYPE_UNPAD), param_(param)
{
    ATB_LOG(INFO) << "UnpadOpsRunner::UnpadOpsRunner called:";
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);

    int64_t inTensorNum = 0;
    Mki::Tensor &inputIds = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &cumOffsetsNow = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &tokenNum = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    Mki::Tensor &xRemovePadding = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &cumOffsetsOut = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &paddingOffset = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &unpadNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::Unpad unpadParam;
    unpadNode.opDesc = {0, "UnpadOperation", unpadParam};
    unpadNode.inTensors = {&inputIds, &cumOffsetsNow, &tokenNum, &seqLen};
    unpadNode.outTensors = {&xRemovePadding, &cumOffsetsOut, &paddingOffset};
}

UnpadOpsRunner::~UnpadOpsRunner() {}
} // namespace atb
