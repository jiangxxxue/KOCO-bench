/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_ops_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>

namespace atb {
SoftmaxOpsRunner::SoftmaxOpsRunner(const infer::SoftmaxParam &param)
    : OpsRunner("SoftmaxOpsRunner", RUNNER_TYPE_SOFTMAX), param_(param)
{
    ATB_LOG(INFO) << "SoftmaxOpsRunner::SoftmaxOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &sortNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Softmax asdParam;
    for (std::size_t i = 0; i < param_.axes.size(); ++i) {
        asdParam.axes.push_back(param_.axes[i]);
    }

    sortNode.opDesc = {0, "SoftmaxOperation", asdParam};
    sortNode.inTensors = {&xTensor};
    sortNode.outTensors = {&resultTensor};
}
SoftmaxOpsRunner::~SoftmaxOpsRunner() {}
} // namespace atb