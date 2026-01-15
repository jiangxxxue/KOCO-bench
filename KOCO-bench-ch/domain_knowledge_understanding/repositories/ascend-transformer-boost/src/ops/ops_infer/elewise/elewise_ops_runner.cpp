/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "atb/utils/log.h"
#include "elewise_ops_runner.h"

namespace atb {
static const uint32_t NUMONE = 1;
static const uint32_t NUMTWO = 2;
static const uint32_t NUMTHREE = 3;
static const uint32_t INDEX_ZERO = 0;
static const uint32_t INDEX_ONE = 1;
static const uint32_t INDEX_TWO = 2;

ElewiseOpsRunner::ElewiseOpsRunner(const infer::ElewiseParam &param)
    : OpsRunner("ElewiseOpsRunner", RUNNER_TYPE_ELEWISE), param_(param)
{
    ATB_LOG(INFO) << "ElewiseOpsRunner::ElewiseOpsRunner called";

    kernelGraph_.nodes.resize(NUMONE);
    auto &elewiseNode = kernelGraph_.nodes.at(INDEX_ZERO);
    if (!SetIntensor(elewiseNode)) {
        return;
    }

    SetOuttensor(elewiseNode);
    ATB_LOG(INFO) << "ElewiseOpsRunner::ElewiseOpsRunner end";
}

ElewiseOpsRunner::~ElewiseOpsRunner() {}

bool ElewiseOpsRunner::SetIntensor(KernelGraphNode &elewiseNode)
{
    uint32_t inTensorNum = GetIntensorSize();
    if (inTensorNum == 0) {
        return false;
    }
    kernelGraph_.inTensors.resize(inTensorNum);
    if (inTensorNum == NUMONE) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        elewiseNode.inTensors = {&aTensor};
    } else if (inTensorNum == NUMTWO) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        Mki::Tensor &bTensor = kernelGraph_.inTensors.at(INDEX_ONE);
        elewiseNode.inTensors = {&aTensor, &bTensor};
    } else if (inTensorNum == NUMTHREE) {
        Mki::Tensor &aTensor = kernelGraph_.inTensors.at(INDEX_ZERO);
        Mki::Tensor &bTensor = kernelGraph_.inTensors.at(INDEX_ONE);
        Mki::Tensor &cTensor = kernelGraph_.inTensors.at(INDEX_TWO);
        elewiseNode.inTensors = {&aTensor, &bTensor, &cTensor};
    } else {
        ATB_LOG(WARN) << "ElewiseOpsRunner::ElewiseOpsRunner inTensorNum: " << inTensorNum;
    }
    return true;
}

void ElewiseOpsRunner::SetOuttensor(KernelGraphNode &elewiseNode)
{
    AsdOps::OpParam::Elewise::ElewiseType opElewiseType = GetOpElwiseType();
    AsdOps::OpParam::Elewise elsewiseParam = {opElewiseType};

    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT) {
        // atb outtensor num, asymmetric:true=>3 false=>2
        size_t resizeNum = param_.quantParam.asymmetric ? NUMTHREE : NUMTWO;
        kernelGraph_.outTensors.resize(resizeNum);
        Mki::Tensor &operationOutTensor0 = kernelGraph_.outTensors.at(INDEX_ZERO);
        Mki::Tensor &operationOutTensor1 = kernelGraph_.outTensors.at(INDEX_ONE);
        Mki::Tensor &operationOutTensor2 =
            param_.quantParam.asymmetric ? kernelGraph_.outTensors.at(INDEX_TWO) : nullTensor_; // 2 : outtensor idx
        elewiseNode.outTensors = {&operationOutTensor0, &operationOutTensor1, &operationOutTensor2};
        elsewiseParam.asymmetric = param_.quantParam.asymmetric;
    } else {
        kernelGraph_.outTensors.resize(INDEX_ONE);
        Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(INDEX_ZERO);
        elewiseNode.outTensors = {&operationOutTensor};
    }

    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_MULS) {
        elsewiseParam.varAttr = param_.mulsParam.varAttr;
    }
    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_QUANT) {
        elsewiseParam.varAttr = 0.0f;
        elsewiseParam.inputScale = param_.quantParam.inputScale;
        elsewiseParam.inputOffset = param_.quantParam.inputOffset;
    }

    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_CAST) {
        elsewiseParam.outTensorType = GetOutTensorType(param_.outTensorType);
    }

    elewiseNode.opDesc = {0, "ElewiseOperation", elsewiseParam};
}

uint32_t ElewiseOpsRunner::GetIntensorSize() const
{
    static std::map<infer::ElewiseParam::ElewiseType, uint32_t> inTensorNumTable = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MULS, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_NEG, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_TANH, NUMONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_ADD, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MUL, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LESS, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, NUMTWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL, NUMTHREE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL, NUMTHREE},
    };
    std::map<infer::ElewiseParam::ElewiseType, uint32_t>::const_iterator it = inTensorNumTable.find(param_.elewiseType);
    return it == inTensorNumTable.end() ? 0 : it->second;
}

AsdOps::OpParam::Elewise::ElewiseType ElewiseOpsRunner::GetOpElwiseType() const
{
    static std::map<infer::ElewiseParam::ElewiseType, AsdOps::OpParam::Elewise::ElewiseType> typeTable = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST, AsdOps::OpParam::Elewise::ELEWISE_CAST},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MULS, AsdOps::OpParam::Elewise::ELEWISE_MULS},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, AsdOps::OpParam::Elewise::ELEWISE_COS},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, AsdOps::OpParam::Elewise::ELEWISE_SIN},
        {infer::ElewiseParam::ElewiseType::ELEWISE_NEG, AsdOps::OpParam::Elewise::ELEWISE_NEG},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, AsdOps::OpParam::Elewise::ELEWISE_QUANT},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, AsdOps::OpParam::Elewise::ELEWISE_LOGICAL_NOT},
        {infer::ElewiseParam::ElewiseType::ELEWISE_ADD, AsdOps::OpParam::Elewise::ELEWISE_ADD},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, AsdOps::OpParam::Elewise::ELEWISE_SUB},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MUL, AsdOps::OpParam::Elewise::ELEWISE_MUL},
        {infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, AsdOps::OpParam::Elewise::ELEWISE_REALDIV},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, AsdOps::OpParam::Elewise::ELEWISE_LOGICAL_AND},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, AsdOps::OpParam::Elewise::ELEWISE_LOGICAL_OR},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LESS, AsdOps::OpParam::Elewise::ELEWISE_LESS},
        {infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, AsdOps::OpParam::Elewise::ELEWISE_GREATER},
        {infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, AsdOps::OpParam::Elewise::ELEWISE_EQUAL},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL,
         AsdOps::OpParam::Elewise::ELEWISE_QUANT_PER_CHANNEL},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL,
         AsdOps::OpParam::Elewise::ELEWISE_DEQUANT_PER_CHANNEL},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT, AsdOps::OpParam::Elewise::ELEWISE_DYNAMIC_QUANT},
        {infer::ElewiseParam::ElewiseType::ELEWISE_TANH, AsdOps::OpParam::Elewise::ELEWISE_TANH},
    };
    std::map<infer::ElewiseParam::ElewiseType, AsdOps::OpParam::Elewise::ElewiseType>::const_iterator it =
        typeTable.find(param_.elewiseType);
    return it == typeTable.end() ? AsdOps::OpParam::Elewise::ELEWISE_CAST : it->second;
}

Mki::TensorDType ElewiseOpsRunner::GetOutTensorType(const aclDataType outType) const
{
    static std::map<aclDataType, Mki::TensorDType> typeTable = {
        {aclDataType::ACL_INT8, Mki::TensorDType::TENSOR_DTYPE_INT8},
        {aclDataType::ACL_FLOAT, Mki::TensorDType::TENSOR_DTYPE_FLOAT},
        {aclDataType::ACL_FLOAT16, Mki::TensorDType::TENSOR_DTYPE_FLOAT16},
        {aclDataType::ACL_INT32, Mki::TensorDType::TENSOR_DTYPE_INT32},
        {aclDataType::ACL_INT64, Mki::TensorDType::TENSOR_DTYPE_INT64},
        {aclDataType::ACL_BF16, Mki::TensorDType::TENSOR_DTYPE_BF16},
    };
    std::map<aclDataType, Mki::TensorDType>::const_iterator it = typeTable.find(outType);
    return it == typeTable.end() ? Mki::TensorDType::TENSOR_DTYPE_UNDEFINED : it->second;
}
} // namespace atb