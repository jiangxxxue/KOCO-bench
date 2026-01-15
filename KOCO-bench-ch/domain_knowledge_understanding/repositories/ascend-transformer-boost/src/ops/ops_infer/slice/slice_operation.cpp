/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_operation.h"
#include "atb/utils/log.h"
#include "slice_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::SliceParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new SliceOperation(opParam);
    return NO_ERROR;
}

SliceOperation::SliceOperation(const infer::SliceParam &param) : OperationBase("SliceOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SliceOperation");
}

SliceOperation::~SliceOperation() {}

uint32_t SliceOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SliceOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SliceOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);

    const SVector<int64_t> &offsets = param_.offsets;
    const SVector<int64_t> &size = param_.size;
    for (size_t i = 0; i < param_.size.size(); i++) {
        int64_t offsetValue = offsets.at(i);
        int64_t sizeValue = size.at(i);
        int64_t xDim = inTensorDescs.at(0).shape.dims[i];
        if (offsetValue < 0) {
            offsetValue = offsetValue + xDim;
        }
        if (sizeValue == -1) {
            sizeValue = xDim - offsetValue;
        }

        outTensorDescs.at(0).shape.dims[i] = sizeValue;
    }

    return NO_ERROR;
}

Status SliceOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return ParamCheck(inTensorDescs.at(0));
}

Status SliceOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status status = ParamCheck(inTensors.at(0).desc);
    if (status != NO_ERROR) {
        return status;
    }

    const SVector<int64_t> &offsets = param_.offsets;
    const SVector<int64_t> &size = param_.size;
    for (size_t i = 0; i < param_.size.size(); i++) {
        int64_t offsetValue = offsets.at(i);
        int64_t sizeValue = size.at(i);
        int64_t xDim = inTensors.at(0).desc.shape.dims[i];
        if (offsetValue < 0) {
            offsetValue = offsetValue + xDim;
        }
        if (sizeValue == -1) {
            sizeValue = xDim - offsetValue;
        }

        if (outTensors.at(0).desc.shape.dims[i] != sizeValue) {
            ATB_LOG(ERROR) << "outTensor dim and param size are different at: " << i
                           << " dim : " << outTensors.at(0).desc.shape.dims[i] << " param size: " << sizeValue;
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    return NO_ERROR;
}

Status SliceOperation::ParamCheck(TensorDesc inTensorDesc) const
{
    size_t sliceSize = param_.size.size();
    auto xTensorDims = inTensorDesc.shape.dimNum;
    if (sliceSize != param_.offsets.size()) {
        ATB_LOG(ERROR) << "SliceOperation InferShapeImpl failed: offsets length must be equal to size length, "
                       << "the offsets length is " << param_.offsets.size() << ". the size length is "
                       << param_.size.size();
        return ERROR_INVALID_PARAM;
    }
    if (sliceSize != xTensorDims) {
        ATB_LOG(ERROR) << "SliceOperation InferShapeImpl failed: size length must be equal to the input tensor dimNum, "
                       << "the size length is " << param_.size.size() << ". the input tensor dimNum is "
                       << inTensorDesc.shape.dimNum;
        return ERROR_INVALID_PARAM;
    }

    const SVector<int64_t> &offsets = param_.offsets;
    const SVector<int64_t> &size = param_.size;
    for (size_t i = 0; i < param_.size.size(); i++) {
        int64_t offsetValue = offsets.at(i);
        int64_t sizeValue = size.at(i);
        int64_t xDim = inTensorDesc.shape.dims[i];
        if (offsetValue < -xDim) {
            ATB_LOG(ERROR) << "SliceOperation InferShapeImpl: wrong offset: " << offsetValue;
            return ERROR_INVALID_PARAM;
        }
        if (offsetValue < 0) {
            offsetValue = offsetValue + xDim;
        }
        if (sizeValue == -1) {
            sizeValue = xDim - offsetValue;
        } else if (sizeValue < -1) {
            ATB_LOG(ERROR) << "SliceOperation InferShapeImpl: Wrong size: " << sizeValue;
            return ERROR_INVALID_PARAM;
        }
        if (std::numeric_limits<int64_t>::max() - offsetValue < sizeValue) {
            ATB_LOG(ERROR) << "SliceOperation InferShapeImpl: Calculate the total size overflow: " << offsetValue
                           << " size: " << sizeValue;
            return ERROR_INVALID_PARAM;
        }
        if (offsetValue + sizeValue > xDim) {
            ATB_LOG(ERROR) << "SliceOperation InferShapeImpl: Wrong offsets or size, offsets:" << offsetValue
                           << " size: " << sizeValue;
            return ERROR_INVALID_PARAM;
        }
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> SliceOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SliceOpsRunner>(param_);
}

nlohmann::json SliceOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb