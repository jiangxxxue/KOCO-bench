/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "grouped_matmul_inplace_add_ops_runner.h"
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"

static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;
static constexpr size_t SIZE_4 = 4;

namespace atb {
GroupedMatmulInplaceAddOpsRunner::GroupedMatmulInplaceAddOpsRunner(const infer::GroupedMatmulInplaceAddParam &param)
    : OpsRunner("GroupedMatmulInplaceAddOpsRunner", RUNNER_TYPE_GROUPED_MATMUL_INPLACE_ADD), param_(param)
{
}

GroupedMatmulInplaceAddOpsRunner::~GroupedMatmulInplaceAddOpsRunner() {}

void GroupedMatmulInplaceAddOpsRunner::SetParam(const Mki::Any &param)
{
    infer::GroupedMatmulInplaceAddParam newParam = Mki::AnyCast<infer::GroupedMatmulInplaceAddParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "GroupedMatmulInplaceAddOpsRunner param changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

Status GroupedMatmulInplaceAddOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    kernelGraph_.inTensors.resize(SIZE_4);
    size_t inTensorId = 0;
    Mki::Tensor &x = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weight = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &groupList = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &out = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    Mki::OpDesc groupedMatmulInplaceAddOpDesc;
    groupedMatmulInplaceAddOpDesc.opName = "GmmAddOperation";
    AtbOps::OpParam::GmmAdd groupedMatmulInplaceAddParam;
    groupedMatmulInplaceAddParam.transposeA = true;
    groupedMatmulInplaceAddParam.transposeB = param_.transposeB;

    if (!param_.transposeA) {
        kernelGraph_.internalTensors.resize(1);
        Mki::Tensor &internalTensor = kernelGraph_.internalTensors.at(0);

        kernelGraph_.nodes.resize(SIZE_2);
        KernelGraphNode &groupedMatmulInplaceAddNode = kernelGraph_.nodes.at(1);
        KernelGraphNode &tranposeNode = kernelGraph_.nodes.at(0);

        groupedMatmulInplaceAddOpDesc.specificParam = groupedMatmulInplaceAddParam;
        groupedMatmulInplaceAddNode.opDesc = groupedMatmulInplaceAddOpDesc;
        groupedMatmulInplaceAddNode.inTensors = {&internalTensor, &weight, &groupList, &out};
        groupedMatmulInplaceAddNode.outTensors = {&outTensor};

        AsdOps::OpParam::Transpose transposeParam = {{1, 0}};
        tranposeNode.opDesc = {0, "TransposeOperation", transposeParam};
        tranposeNode.inTensors = {&x};
        tranposeNode.outTensors = {&internalTensor};
    } else {
        kernelGraph_.internalTensors.resize(0);
        kernelGraph_.nodes.resize(1);
        KernelGraphNode &groupedMatmulInplaceAddNode = kernelGraph_.nodes.at(0);

        groupedMatmulInplaceAddOpDesc.specificParam = groupedMatmulInplaceAddParam;
        groupedMatmulInplaceAddNode.opDesc = groupedMatmulInplaceAddOpDesc;
        groupedMatmulInplaceAddNode.inTensors = {&x, &weight, &groupList, &out};
        groupedMatmulInplaceAddNode.outTensors = {&outTensor};
    }

    (void)opsTensorPack;
    return NO_ERROR;
}
} // namespace atb