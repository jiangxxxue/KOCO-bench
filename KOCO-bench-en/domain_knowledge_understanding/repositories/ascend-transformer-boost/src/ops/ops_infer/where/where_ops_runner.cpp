/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "where_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"

namespace atb {
WhereOpsRunner::WhereOpsRunner(const infer::WhereParam &param)
    : OpsRunner("WhereOpsRunner", RUNNER_TYPE_WHERE), param_(param)
{
    static const uint32_t DIM_3 = 3;
    static const uint32_t DIM_4 = 4;

    ATB_LOG(INFO) << "WhereOpsRunner::WhereOpsRunner called.";

    kernelGraph_.inTensors.resize(DIM_3);
    int64_t inTensorNum = 0;
    Mki::Tensor &conditionTensor = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &inputTensor = kernelGraph_.inTensors[inTensorNum++];
    Mki::Tensor &otherTensor = kernelGraph_.inTensors[inTensorNum++];

    kernelGraph_.outTensors.resize(1);
    int64_t outTensorNum = 0;
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.internalTensors.resize(DIM_3);
    int64_t internalTensorNum = 0;
    Mki::Tensor &condLogicalNotTensor = kernelGraph_.internalTensors[internalTensorNum++];
    Mki::Tensor &inputMaskedFillTensor = kernelGraph_.internalTensors[internalTensorNum++];
    Mki::Tensor &otherMaskedFillTensor = kernelGraph_.internalTensors[internalTensorNum++];

    kernelGraph_.nodes.resize(DIM_4);
    int64_t nodeNum = 0;
    auto &logicalNotNode = kernelGraph_.nodes.at(nodeNum++);
    auto &inputMaskedFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &otherMaskedFillNode = kernelGraph_.nodes.at(nodeNum++);
    auto &addNode = kernelGraph_.nodes.at(nodeNum++);

    logicalNotNode.opDesc = {0, "ElewiseOperation",
                             AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_LOGICAL_NOT})};
    logicalNotNode.inTensors = {&conditionTensor};
    logicalNotNode.outTensors = {&condLogicalNotTensor};

    Mki::SVector<float> value = {0};
    Mki::SVector<int64_t> outDim = {0};

    AsdOps::OpParam::Fill maskedFillParam = {true, value, outDim};
    inputMaskedFillNode.opDesc = {0, "FillOperation", maskedFillParam};
    inputMaskedFillNode.inTensors = {&inputTensor, &condLogicalNotTensor};
    inputMaskedFillNode.outTensors = {&inputMaskedFillTensor};

    otherMaskedFillNode.opDesc = {0, "FillOperation", maskedFillParam};
    otherMaskedFillNode.inTensors = {&otherTensor, &conditionTensor};
    otherMaskedFillNode.outTensors = {&otherMaskedFillTensor};

    addNode.opDesc = {0, "ElewiseOperation", AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_ADD})};
    addNode.inTensors = {&inputMaskedFillTensor, &otherMaskedFillTensor};
    addNode.outTensors = {&operationOutTensor};
}

WhereOpsRunner::~WhereOpsRunner() {}
} // namespace atb
