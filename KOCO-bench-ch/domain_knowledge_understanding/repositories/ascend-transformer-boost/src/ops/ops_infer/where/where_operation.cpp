/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "where_operation.h"
#include "where_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_1 = 1;
static const uint32_t DIM_2 = 2;

template <> Status CreateOperation(const infer::WhereParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new WhereOperation(opParam);
    return NO_ERROR;
}

WhereOperation::WhereOperation(const infer::WhereParam &param) : OperationBase("WhereOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("WhereOperation");
}

WhereOperation::~WhereOperation() {}

uint32_t WhereOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t WhereOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status WhereOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(1).format;
    size_t dimA = inTensorDescs.at(1).shape.dimNum;
    size_t dimB = inTensorDescs.at(2).shape.dimNum;
    size_t maxDim = std::max(dimA, dimB);
    outTensorDescs.at(0).shape.dimNum = maxDim;
    for (size_t i = 0; i < dimA; ++i) {
        outTensorDescs.at(0).shape.dims[i + maxDim - dimA] = inTensorDescs.at(1).shape.dims[i];
    }
    for (size_t i = 0; i < dimB; ++i) {
        size_t index = i + maxDim - dimB;
        if (outTensorDescs.at(0).shape.dims[index] == 0 || outTensorDescs.at(0).shape.dims[index] == 1) {
            outTensorDescs.at(0).shape.dims[index] = inTensorDescs.at(DIM_2).shape.dims[i];
        }
    }
    return NO_ERROR;
}

Status WhereOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return DimsCheck(inTensorDescs.at(DIM_0), inTensorDescs.at(DIM_1), inTensorDescs.at(DIM_2));
}

Status WhereOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (std::max(inTensors.at(DIM_1).desc.shape.dimNum, inTensors.at(DIM_2).desc.shape.dimNum) !=
        outTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dimNum [" << outTensors.at(0).desc.shape.dimNum
                       << "] != max of inTensor dimNum [" << inTensors.at(DIM_1).desc.shape.dimNum << "] and ["
                       << inTensors.at(DIM_2).desc.shape.dimNum << "]";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Status st = DimsCheck(inTensors.at(DIM_0).desc, inTensors.at(DIM_1).desc, inTensors.at(DIM_2).desc);
    if (st != NO_ERROR) {
        return st;
    }
    return OutTensorDimCheck(inTensors, outTensors);
}

Status WhereOperation::DimsCheck(const TensorDesc &condTensorDesc, const TensorDesc &inputTensorDesc,
                                 const TensorDesc &otherTensorDesc) const
{
    Dims broadcastInputDim = {{0}, 0};
    if (!CanBroadcast(condTensorDesc.shape, inputTensorDesc.shape, broadcastInputDim) ||
        !IsDimsEqual(broadcastInputDim, inputTensorDesc.shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The dims of cond should be able to broadcast to x";
        return ERROR_INVALID_TENSOR_DIM;
    }
    Dims broadcastOtherDim = {{0}, 0};
    if (!CanBroadcast(condTensorDesc.shape, otherTensorDesc.shape, broadcastOtherDim) ||
        !IsDimsEqual(broadcastOtherDim, otherTensorDesc.shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The dims of cond should be able to broadcast to y";
        return ERROR_INVALID_TENSOR_DIM;
    }
    Dims broadcastResDim = {{0}, 0};
    if (!CanBroadcast(inputTensorDesc.shape, otherTensorDesc.shape, broadcastResDim)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The dims of x and y should be able to broadcast";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status WhereOperation::OutTensorDimCheck(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Dims broadcastResDim = {{0}, 0};
    if (!CanBroadcast(inTensors.at(DIM_1).desc.shape, inTensors.at(DIM_2).desc.shape, broadcastResDim)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The dims of inTensors can not be broadcast";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!IsDimsEqual(broadcastResDim, outTensors.at(0).desc.shape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The dims of outTensor should be the broadcast result of x and y";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

bool WhereOperation::CanBroadcast(const Dims &dimsA, const Dims &dimsB, Dims &dimsRes) const
{
    size_t dimNumA = dimsA.dimNum;
    size_t dimNumB = dimsB.dimNum;
    size_t dimNumMax = std::max(dimNumA, dimNumB);
    dimsRes.dimNum = dimNumMax;
    while (dimNumA > 0 && dimNumB > 0) {
        int64_t dimA = dimsA.dims[dimNumA - 1];
        int64_t dimB = dimsB.dims[dimNumB - 1];
        if (dimA != dimB && dimA != 1 && dimB != 1) {
            return false;
        } else {
            dimsRes.dims[dimNumMax - 1] = dimA == 1 ? dimB : dimA;
            dimNumMax -= 1;
        }
        dimNumA -= 1;
        dimNumB -= 1;
    }
    while (dimNumA > 0) {
        int64_t dimA = dimsA.dims[dimNumA - 1];
        dimsRes.dims[dimNumA - 1] = dimA;
        dimNumA -= 1;
    }
    while (dimNumB > 0) {
        int64_t dimB = dimsB.dims[dimNumB - 1];
        dimsRes.dims[dimNumB - 1] = dimB;
        dimNumB -= 1;
    }
    return true;
}

bool WhereOperation::IsDimsEqual(const Dims &dimsA, const Dims &dimsB) const
{
    if (dimsA.dimNum != dimsB.dimNum) {
        return false;
    }
    for (size_t i = 0; i < dimsA.dimNum; ++i) {
        if (dimsA.dims[i] != dimsB.dims[i]) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<Runner> WhereOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<WhereOpsRunner>(param_);
}
} // namespace atb