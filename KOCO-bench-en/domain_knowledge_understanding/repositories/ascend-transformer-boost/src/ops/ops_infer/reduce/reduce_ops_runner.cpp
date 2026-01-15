/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
ReduceOpsRunner::ReduceOpsRunner(const infer::ReduceParam &param)
    : OpsRunner("ReduceOpsRunner", RUNNER_TYPE_REDUCE), param_(param)
{
    ATB_LOG(INFO) << "ReduceOpsRunner::ReduceOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    AsdOps::OpParam::Reduce opParam = {};
    BuildOpParam(opParam);

    kernelGraph_.nodes.resize(1);
    auto &reduceNode = kernelGraph_.nodes.at(0);
    reduceNode.opDesc = {0, "ReduceOperation", opParam};
    reduceNode.inTensors = {&xTensor};
    reduceNode.outTensors = {&operationOutTensor};
}

ReduceOpsRunner::~ReduceOpsRunner() {}

void ReduceOpsRunner::BuildOpParam(AsdOps::OpParam::Reduce &opParam)
{
    static std::map<infer::ReduceParam::ReduceType, AsdOps::OpParam::Reduce::ReduceType> typeTable = {
        {infer::ReduceParam::ReduceType::REDUCE_MAX, AsdOps::OpParam::Reduce::ReduceType::REDUCE_MAX},
        {infer::ReduceParam::ReduceType::REDUCE_MIN, AsdOps::OpParam::Reduce::ReduceType::REDUCE_MIN},
        {infer::ReduceParam::ReduceType::REDUCE_SUM, AsdOps::OpParam::Reduce::ReduceType::REDUCE_SUM},
    };
    auto it = typeTable.find(param_.reduceType);
    opParam.reduceType = (it == typeTable.end()) ? AsdOps::OpParam::Reduce::ReduceType::REDUCE_UNDEFINED : it->second;

    Mki::SVector<int64_t> axis;
    TensorUtil::AtbSVector2OpsSVector(param_.axis, axis);
    opParam.axis = axis;
}
} // namespace atb