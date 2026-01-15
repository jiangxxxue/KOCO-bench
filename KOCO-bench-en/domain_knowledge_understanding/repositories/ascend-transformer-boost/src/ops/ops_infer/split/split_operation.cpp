/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "split_operation.h"
#include "split_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t OUT_TENSOR_NUM_BISECT = 2;
static const uint32_t OUT_TENSOR_NUM_TRISECT = 3;
static const uint32_t OUT_TENSOR_INDEX2 = 2;

template <> Status CreateOperation(const infer::SplitParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_OPERATION_ADDR;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.splitNum > 3 || opParam.splitNum <= 1) { // splitNum:2 or 3
        ATB_LOG(ERROR) << "SplitOperation splitNum should be 2 or 3, now splitNum is " << opParam.splitNum;
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) SplitOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

SplitOperation::SplitOperation(const infer::SplitParam &param) : OperationBase("SplitOperation"), param_(param)
{
    std::string opIrKey = param_.splitNum == OUT_TENSOR_NUM_BISECT ? "SplitOperationBISECT" : "SplitOperationTRISECT";
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}

SplitOperation::~SplitOperation() {}

uint32_t SplitOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 1;
    return inTensorNum;
}

uint32_t SplitOperation::GetOutputNum() const
{
    if (param_.splitNum == 2) { // splitNum: 2
        return OUT_TENSOR_NUM_BISECT;
    } else {
        return OUT_TENSOR_NUM_TRISECT;
    }
}

Status SplitOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    int32_t dim = param_.splitDim;
    Status paramCheckStatus = ParamCheck(xTensorDesc, dim);
    if (paramCheckStatus != NO_ERROR) {
        return paramCheckStatus;
    }
    if (param_.splitSizes.size() > 0) {
        std::size_t size = param_.splitSizes.size();
        for (std::size_t i = 0; i < size; i++) {
            outTensorDescs.at(i) = xTensorDesc;
            outTensorDescs.at(i).shape.dims[param_.splitDim] = param_.splitSizes.at(i);
        }
    } else {
        outTensorDescs.at(0) = xTensorDesc;
        outTensorDescs.at(0).shape.dims[dim] = xTensorDesc.shape.dims[dim] / param_.splitNum;
        outTensorDescs.at(1) = outTensorDescs.at(0);
        if (param_.splitNum == 3) {                      // splitNum: 3
            outTensorDescs.at(2) = outTensorDescs.at(0); // dim: 2
        }
    }
    return NO_ERROR;
}

Status SplitOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    int32_t dim = param_.splitDim;
    return ParamCheck(inTensors.at(0).desc, dim);
}

Status SplitOperation::ParamCheck(const TensorDesc &xTensorDesc, int32_t &dim) const
{
    if (dim < 0) {
        dim = static_cast<int32_t>(xTensorDesc.shape.dimNum) + dim;
    }
    if (dim < 0 || static_cast<int32_t>(xTensorDesc.shape.dimNum) <= dim) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Incorrect splitDim.";
        return ERROR_INVALID_PARAM;
    }
    if (param_.splitSizes.size() > 0) {
        if (param_.splitDim < 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "splitDim cannot be less than zero when use splitV";
            return ERROR_INVALID_PARAM;
        }
        if (param_.splitNum != static_cast<int32_t>(param_.splitSizes.size())) {
            ATB_LOG(ERROR) << GetLogPrefix() << "splitNum is not equal to the length of splitSizes";
            return ERROR_INVALID_PARAM;
        }
        int64_t splitSizeSum = 0;
        std::size_t size = param_.splitSizes.size();
        for (std::size_t i = 0; i < size; i++) {
            if (param_.splitSizes.at(i) < 1) { // The minimum value of the element is 1
                ATB_LOG(ERROR) << GetLogPrefix() << "the elements in splitSizes is less than 1";
                return ERROR_INVALID_PARAM;
            }
            splitSizeSum += param_.splitSizes.at(i);
        }
        if (splitSizeSum != xTensorDesc.shape.dims[param_.splitDim]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "the sum of splitSizes is not equal to the size of split dimension";
            return ERROR_INVALID_PARAM;
        }
    } else {
        if (xTensorDesc.shape.dims[dim] % param_.splitNum != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dims[splitDim] mod splitNum is not zero";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> SplitOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SplitOpsRunner>(param_);
}

nlohmann::json SplitOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb