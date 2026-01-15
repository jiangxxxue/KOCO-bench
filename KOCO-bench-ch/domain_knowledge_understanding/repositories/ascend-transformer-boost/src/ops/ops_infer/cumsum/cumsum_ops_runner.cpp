/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cumsum_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
CumsumOpsRunner::CumsumOpsRunner(const infer::CumsumParam &param)
    : OpsRunner("CumsumOpsRunner", RUNNER_TYPE_CUMSUM), param_(param)
{
    ATB_LOG(INFO) << "CumsumOpsRunner::CumsumOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(2); // nodes: 2
    auto &fillNode = kernelGraph_.nodes[0];
    auto &cumsumNode = kernelGraph_.nodes[1];

    fillNode.opDesc = {0, "FillOperation", AsdOps::OpParam::Fill()};
    fillNode.outTensors = {&resultTensor};
    fillNode.inferShapePreFunc = [&](Mki::LaunchParam &launchParam) {
        const auto &inDim = xTensor.desc.dims;
        launchParam.SetParam(AsdOps::OpParam::Fill({false, {0}, inDim}));
    };

    AsdOps::OpParam::Cumsum asdParam;
    asdParam.exclusive = param_.exclusive;
    asdParam.reverse = param_.reverse;
    for (std::size_t i = 0; i < param_.axes.size(); ++i) {
        asdParam.axis.push_back(param_.axes[i]);
    }

    cumsumNode.opDesc = {0, "CumsumOperation", asdParam};
    cumsumNode.inTensors = {&xTensor};
    cumsumNode.outTensors = {&resultTensor};
}

CumsumOpsRunner::~CumsumOpsRunner() {}
} // namespace atb
