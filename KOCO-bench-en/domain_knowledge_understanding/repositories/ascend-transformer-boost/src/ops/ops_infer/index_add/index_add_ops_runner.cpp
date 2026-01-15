/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "index_add_ops_runner.h"
#include <asdops/params/params.h>
#include <atb/utils/log.h>

namespace atb {
IndexAddOpsRunner::IndexAddOpsRunner(const infer::IndexAddParam &param)
    : OpsRunner("IndexAddOpsRunner", RUNNER_TYPE_INDEXADD), param_(param)
{
    ATB_LOG(INFO) << "IndexAddOpsRunner::IndexAddOpsRunner";
}

IndexAddOpsRunner::~IndexAddOpsRunner() {}

Status IndexAddOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    size_t inTensorNum = 4;
    size_t outTensorNum = 1;
    size_t internalTensorNum = 0;
    size_t nodeNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);
    kernelGraph_.internalTensors.resize(internalTensorNum);
    kernelGraph_.nodes.resize(nodeNum);

    if (param_.indexType == infer::IndexAddParam::INDEX_ADD) {
        return SetupKernelGraphIndexAdd();
    } else if (param_.indexType == infer::IndexAddParam::INDEX_ADD_VALID) {
        return SetupKernelGraphIndexAddValid();
    }

    return NO_ERROR;
}

Status IndexAddOpsRunner::SetupKernelGraphIndexAdd()
{
    ATB_LOG(INFO) << GetLogPrefix() << "IndexAddOpsRunner::SetupKernelGraphIndexAdd";

    size_t inTensorId = 0;
    Mki::Tensor &varTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &indicesTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &updatesTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &alphaTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &indexAddNode = kernelGraph_.nodes.at(nodeId++);

    AsdOps::OpParam::Index indexParam;
    indexParam.axis = param_.axis;
    indexParam.indexType = AsdOps::OpParam::Index::INDEX_ADD;
    indexAddNode.opDesc = {0, "IndexOperation", indexParam};
    indexAddNode.inTensors = {&varTensor, &indicesTensor, &updatesTensor, &alphaTensor};
    indexAddNode.outTensors = {&outTensor};

    return NO_ERROR;
}

Status IndexAddOpsRunner::SetupKernelGraphIndexAddValid()
{
    ATB_LOG(INFO) << GetLogPrefix() << "IndexAddOpsRunner::SetupKernelGraphIndexAddValid";

    size_t inTensorId = 0;
    Mki::Tensor &varTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &indicesTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &updatesTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &validIndicesTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &indexAddValidNode = kernelGraph_.nodes.at(nodeId++);

    AsdOps::OpParam::Index indexParam;
    indexParam.axis = param_.axis;
    indexParam.indexType = AsdOps::OpParam::Index::INDEX_ADD_VALID;
    indexAddValidNode.opDesc = {0, "IndexOperation", indexParam};
    indexAddValidNode.inTensors = {&varTensor, &indicesTensor, &updatesTensor, &validIndicesTensor};
    indexAddValidNode.outTensors = {&outTensor};

    return NO_ERROR;
}
} // namespace atb
