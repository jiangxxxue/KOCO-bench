/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rms_norm_backward_ops_runner.h"
#include <asdops/params/params.h>
#include <atb/utils/log.h>
#include <atb/utils/tensor_util.h>

namespace atb {
static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 2;

RmsNormBackwardOpsRunner::RmsNormBackwardOpsRunner(const train::RmsNormBackwardParam &param)
    : OpsRunner("RmsNormBackwardOpsRunner", RUNNER_TYPE_RMS_NORM_BACKWARD), param_(param)
{
    ATB_LOG(INFO) << "RmsNormBackwardOpsRunner::RmsNormBackwardOpsRunner called";
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);

    int64_t inTensorNum = 0;
    Mki::Tensor &tensorDy = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &tensorX = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &tensorRSTD = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &tensorGamma = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    Mki::Tensor &tensorDx = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &tensorDgamma = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormBackwardNode = kernelGraph_.nodes.at(0);

    AsdOps::OpParam::Norm rmsNormBackwardParam = {AsdOps::OpParam::Norm::RMS_NORM_BACKWARD};

    rmsNormBackwardNode.opDesc = {0, "NormOperation", rmsNormBackwardParam};
    rmsNormBackwardNode.inTensors = {&tensorDy, &tensorX, &tensorRSTD, &tensorGamma};
    rmsNormBackwardNode.outTensors = {&tensorDx, &tensorDgamma};
}

RmsNormBackwardOpsRunner::~RmsNormBackwardOpsRunner() {}
} // namespace atb