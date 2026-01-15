/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pad_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/runner_util.h"
#include "atb/utils/log.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 1;
PadOpsRunner::PadOpsRunner(const infer::PadParam &param) : OpsRunner("PadOpsRunner", RUNNER_TYPE_PAD), param_(param)
{
    ATB_LOG(INFO) << "PadOpsRunner::PadOpsRunner called:";
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);

    int64_t inTensorNum = 0;
    Mki::Tensor &tmpOut = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &paddingOffset = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &inputIds = kernelGraph_.inTensors.at(inTensorNum++);

    Mki::Tensor &out = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &padNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::Pad padParam;
    padNode.opDesc = {0, "PadOperation", padParam};
    padNode.inTensors = {&tmpOut, &paddingOffset, &seqLen, &inputIds};
    padNode.outTensors = {&out};
}

PadOpsRunner::~PadOpsRunner() {}
} // namespace atb
