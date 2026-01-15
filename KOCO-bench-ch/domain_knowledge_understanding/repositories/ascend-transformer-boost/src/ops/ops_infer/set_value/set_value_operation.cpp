/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_value_operation.h"
#include "set_value_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 2;
static const int32_t OUT_TENSOR_NUM = 0;

template <> Status CreateOperation(const infer::SetValueParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new SetValueOperation(opParam);
    return NO_ERROR;
}

SetValueOperation::SetValueOperation(const infer::SetValueParam &param)
    : OperationBase("SetValueOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SetValueOperation");
}

SetValueOperation::~SetValueOperation() {}

uint32_t SetValueOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SetValueOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SetValueOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status SetValueOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &dstTensorDesc = inTensorDescs.at(0);
    const TensorDesc &srcTensorDesc = inTensorDescs.at(1);
    Status status = DimNumCheckImpl(dstTensorDesc, srcTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    status = DimsCheckImpl(dstTensorDesc, srcTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    return NO_ERROR;
}

Status SetValueOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    const TensorDesc &dstTensorDesc = inTensors.at(0).desc;
    const TensorDesc &srcTensorDesc = inTensors.at(1).desc;
    Status status = DimNumCheckImpl(dstTensorDesc, srcTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    status = DimsCheckImpl(dstTensorDesc, srcTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "outTensorSize:" << outTensors.size();
    return NO_ERROR;
}

Status SetValueOperation::DimNumCheckImpl(const TensorDesc &dstTensorDesc, const TensorDesc &srcTensorDesc) const
{
    uint64_t dstDimNum = dstTensorDesc.shape.dimNum;
    uint64_t srcDimNum = srcTensorDesc.shape.dimNum;
    if (dstDimNum != srcDimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << " DimNumCheckImpl failed - src dimNum should equal to dst dimNunm";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (dstDimNum != param_.strides.size() || dstDimNum != param_.starts.size() || dstDimNum != param_.ends.size()) {
        ATB_LOG(ERROR)
            << GetLogPrefix()
            << " DimNumCheckImpl failed - param_ strides & starts & ends dimNunm should equal to src/dst dimNunm";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status SetValueOperation::DimsCheckImpl(const TensorDesc &dstTensorDesc, const TensorDesc &srcTensorDesc) const
{
    uint64_t dstDimNum = dstTensorDesc.shape.dimNum;
    for (size_t i = 0; i < dstDimNum; i++) {
        if (srcTensorDesc.shape.dims[i] > dstTensorDesc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << " DimCheckImpl failed - src dim should not more than dst dim";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    for (size_t i = 0; i < param_.strides.size(); i++) {
        if (param_.strides[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "DimCheckImpl failed - all stride should be 1";
            return ERROR_INVALID_PARAM;
        }
    }
    for (size_t i = 0; i < param_.starts.size(); i++) {
        if (param_.starts.at(i) < 0 || param_.ends.at(i) > dstTensorDesc.shape.dims[i] ||
            param_.starts.at(i) >= param_.ends.at(i) ||
            (param_.ends.at(i) - param_.starts.at(i) - 1) / param_.strides.at(i) + 1 != srcTensorDesc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << " DimCheckImpl failed - param_ starts[i] is larger than ends[i] or copy shape is "
                              "different from src shape ";
            return ERROR_INVALID_PARAM;
        }
    }
    size_t count = 0;
    for (size_t i = 1; i < dstDimNum; i++) {
        if (dstTensorDesc.shape.dims[i] != srcTensorDesc.shape.dims[i]) {
            count++;
            if (count > 1) {
                ATB_LOG(ERROR) << GetLogPrefix()
                               << "DimCheckImpl failed - src and dst at most have two different dimensions, and one of "
                                  "them is the "
                                  "highest dimension";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    if (count == 0) {
        ATB_LOG(ERROR)
            << GetLogPrefix()
            << "DimCheckImpl failed - If src and dst have only one different dimension, this dimension cannot be the "
               "highest dimension ex.support[a,b,c]->[a',b,c]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> SetValueOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SetValueOpsRunner>(param_);
}

nlohmann::json SetValueOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
