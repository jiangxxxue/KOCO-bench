/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;
SplitOpsRunner::SplitOpsRunner(const infer::SplitParam &param)
    : OpsRunner("SplitOpsRunner", RUNNER_TYPE_SPLIT), param_(param)
{
    ATB_LOG(INFO) << "SplitOpsRunner::SplitOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &aTensor = kernelGraph_.inTensors.at(0);
    kernelGraph_.nodes.resize(1);
    auto &splitNode = kernelGraph_.nodes.at(0);
    AsdOps::OpParam::Split splitParam;
    splitParam.splitNum = param_.splitNum;
    splitParam.splitDim = param_.splitDim;
    splitParam.splitVDim = {param_.splitDim > 0 ? param_.splitDim : 0};
    Mki::SVector<int32_t> asdSplitSizes;
    TensorUtil::AtbSVector2OpsSVector(param.splitSizes, asdSplitSizes);
    splitParam.splitSize = asdSplitSizes;
    splitNode.opDesc = {0, "SplitOperation", splitParam};
    splitNode.inTensors = {&aTensor};
    if (param_.splitNum == DIM_2) {
        ATB_LOG(INFO) << GetName() << " split by 2";
        kernelGraph_.outTensors.resize(DIM_2);
        Mki::Tensor &operationOutTensor0 = kernelGraph_.outTensors.at(0);
        Mki::Tensor &operationOutTensor1 = kernelGraph_.outTensors.at(1);
        splitNode.outTensors = {&operationOutTensor0, &operationOutTensor1};
    } else {
        ATB_LOG(INFO) << GetName() << " split by 3";
        kernelGraph_.outTensors.resize(DIM_3);
        Mki::Tensor &operationOutTensor0 = kernelGraph_.outTensors.at(0);
        Mki::Tensor &operationOutTensor1 = kernelGraph_.outTensors.at(1);
        Mki::Tensor &operationOutTensor2 = kernelGraph_.outTensors.at(2);
        splitNode.outTensors = {&operationOutTensor0, &operationOutTensor1, &operationOutTensor2};
    }
}

SplitOpsRunner::~SplitOpsRunner() {}
} // namespace atb