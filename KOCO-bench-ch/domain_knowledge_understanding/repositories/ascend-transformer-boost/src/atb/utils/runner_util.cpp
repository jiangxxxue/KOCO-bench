/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/runner_util.h"
#include <map>
#include <asdops/params/params.h>
#include "atb/utils/log.h"

static constexpr size_t DIM_2 = 2;
static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;

namespace atb {
Mki::OpDesc RunnerUtil::GetActivationNodeOpDesc(infer::ActivationParam activationParam)
{
    std::map<infer::ActivationType, AsdOps::OpParam::Activation::ActivationType> typeTable = {
        {infer::ActivationType::ACTIVATION_RELU, AsdOps::OpParam::Activation::ACTIVATION_RELU},
        {infer::ActivationType::ACTIVATION_GELU, AsdOps::OpParam::Activation::ACTIVATION_GELU},
        {infer::ActivationType::ACTIVATION_FAST_GELU, AsdOps::OpParam::Activation::ACTIVATION_FAST_GELU},
        {infer::ActivationType::ACTIVATION_SWISH, AsdOps::OpParam::Activation::ACTIVATION_SWISH},
        {infer::ActivationType::ACTIVATION_LOG, AsdOps::OpParam::Activation::ACTIVATION_LOG},
        {infer::ActivationType::ACTIVATION_SWIGLU_FORWARD, AsdOps::OpParam::Activation::ACTIVATION_SWIGLU_FORWARD},
        {infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD, AsdOps::OpParam::Activation::ACTIVATION_SWIGLU_BACKWARD},
        {infer::ActivationType::ACTIVATION_SIGMOID, AsdOps::OpParam::Activation::ACTIVATION_SIGMOID},
        {infer::ActivationType::ACTIVATION_FASTER_GELU_FORWARD,
         AsdOps::OpParam::Activation::ACTIVATION_FASTER_GELU_FORWARD},
    };

    std::map<infer::ActivationType, AsdOps::OpParam::Activation::ActivationType>::const_iterator it =
        typeTable.find(activationParam.activationType);
    AsdOps::OpParam::Activation param = {};
    param.activationType = (it == typeTable.end()) ? AsdOps::OpParam::Activation::ACTIVATION_UNDEFINED : it->second;
    param.scale = activationParam.scale;
    param.dim = activationParam.dim;
    param.approx = activationParam.geluMode;
    return {0, "ActivationOperation", param};
}

/**
 * transdata前，保证矩阵为3维，若为2维或合轴后2维，则第0维补1
 * @param transdataNode
 * @param needMergeAxis
 */
void RunnerUtil::TransdataSqueeze(KernelGraphNode &transdataNode, bool needMergeAxis)
{
    size_t viewFuncsSize = transdataNode.inTensors.size();
    if (viewFuncsSize < 1) {
        ATB_LOG(ERROR) << "node inTensorViewFuncs size error!";
        return;
    }
    transdataNode.inTensorViewFuncs.resize(viewFuncsSize);
    transdataNode.inTensorViewFuncs.at(0) = [needMergeAxis](const Mki::SVector<int64_t> &oldDims,
                                                            Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() == SIZE_2) {
            newDims = {1, oldDims.at(0), oldDims.at(1)};
        } else if (oldDims.size() == SIZE_3 && needMergeAxis) {
            newDims = {1, oldDims.at(0) * oldDims.at(1), oldDims.at(2)};
        } else {
            newDims = oldDims;
        }
    };
}

/**
 * 将维度值进行对齐
 * @param dim 维度值
 * @param align 对齐值
 * @return 对齐后的维度值
 */
int64_t RunnerUtil::AlignUp(int64_t dim, int64_t align)
{
    if (align == 0) {
        return -1;
    }
    return (dim + align - 1) / align * align;
}

/**
 * 配置matmulNode的inTensorViewFuncs，根据需求进行合轴或reshape
 * @param matmulNode
 * @param needMergeAxis 是否需要合轴；当X矩阵为3维，且Y矩阵为2维，则需将X矩阵前两维合轴
 * @param needReshape 是否需要进行reshape；当权重矩阵为NZ时，若为2维或3维，进行reshape
 * @param align
 */
void RunnerUtil::ConfigViewFuncs(KernelGraphNode &matmulNode, bool needMergeAxis, bool needReshape, int64_t align)
{
    size_t viewFuncSize = matmulNode.inTensors.size();
    if (viewFuncSize < SIZE_2) {
        ATB_LOG(ERROR) << "node inTensorViewFuncs size error!";
        return;
    }
    matmulNode.inTensorViewFuncs.resize(viewFuncSize);
    matmulNode.inTensorViewFuncs.at(0) = [needMergeAxis](const Mki::SVector<int64_t> &oldDims,
                                                         Mki::SVector<int64_t> &newDims) {
        newDims = oldDims;
        if (needMergeAxis && oldDims.size() == SIZE_3) {
            newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(DIM_2)};
        }
        ATB_LOG_IF(newDims != oldDims, INFO) << " - Merge axis, Before: " << oldDims << "; After: " << newDims;
    };
    if (needReshape) {
        if (align == 0) {
            ATB_LOG(ERROR) << "align should not be 0";
            return;
        }
        matmulNode.inTensorViewFuncs.at(1) = [align](const Mki::SVector<int64_t> &oldDims,
                                                     Mki::SVector<int64_t> &newDims) {
            if (oldDims.size() == SIZE_2) {
                newDims = {1, AlignUp(oldDims.at(1), align) / align, AlignUp(oldDims.at(0), DEFAULT_ALIGN), align};
            } else if (oldDims.size() == SIZE_3) {
                newDims = {oldDims.at(0), AlignUp(oldDims.at(DIM_2), align) / align,
                           AlignUp(oldDims.at(1), DEFAULT_ALIGN), align};
            } else {
                newDims = oldDims;
            }
            ATB_LOG_IF(newDims != oldDims, INFO) << " - Weight reshape, Before: " << oldDims << "; After: " << newDims;
        };
    }
}

bool RunnerUtil::InitGraphRunnerNode(GraphRunner::Node &node, Operation *operation,
                                     const std::vector<int64_t> &operationIds, ContextBase &context)
{
    if (!operation) {
        return false;
    }
    node.op.reset(operation);
    OperationBase *opBase = dynamic_cast<OperationBase *>(operation);
    if (!opBase) {
        ATB_LOG(ERROR) << "dynamic_cast operation to OperationBase failed!";
        return false;
    }
    node.runner = opBase->CreateRunner(context);
    if (!node.runner) {
        ATB_LOG(ERROR) << "CreateRunner failed!";
        return false;
    }
    node.runner->SetRunnerInfo(node.op->GetName(), operationIds);
    return true;
}
} // namespace atb