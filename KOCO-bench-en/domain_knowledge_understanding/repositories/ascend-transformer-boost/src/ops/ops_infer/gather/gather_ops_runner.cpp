

/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
GatherOpsRunner::GatherOpsRunner(const infer::GatherParam &param)
    : OpsRunner("GatherOpsRunner", RUNNER_TYPE_GATHER), param_(param)
{
    ATB_LOG(INFO) << "GatherOpsRunner::GatherOpsRunner called";
    kernelGraph_.inTensors.resize(2); // intersorNum:2
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &yTensor = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &gatherNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Gather gatherNodeParam = {param_.batchDims, {param_.axis}};

    gatherNode.opDesc = {0, "GatherOperation", gatherNodeParam};
    gatherNode.inTensors = {&xTensor, &yTensor};
    gatherNode.outTensors = {&outTensor};
}

GatherOpsRunner::~GatherOpsRunner() {}

} // namespace atb
