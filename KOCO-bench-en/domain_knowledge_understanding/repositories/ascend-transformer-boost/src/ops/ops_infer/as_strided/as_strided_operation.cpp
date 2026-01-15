/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "as_strided_operation.h"
#include <limits>
#include "as_strided_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::AsStridedParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new AsStridedOperation(opParam);
    return NO_ERROR;
}

AsStridedOperation::AsStridedOperation(const infer::AsStridedParam &param)
    : OperationBase("AsStridedOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AsStridedOperation");
}

AsStridedOperation::~AsStridedOperation() {}

uint32_t AsStridedOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AsStridedOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AsStridedOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = ACL_FORMAT_ND;
    std::size_t size = param_.size.size();
    for (std::size_t i = 0; i < size; i++) {
        outTensorDescs.at(0).shape.dims[i] = param_.size.at(i);
    }
    outTensorDescs.at(0).shape.dimNum = size;
    return NO_ERROR;
}

Status AsStridedOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return ParamCheck(inTensorDescs.at(0));
}

Status AsStridedOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (outTensors.at(0).desc.shape.dimNum != param_.size.size()) {
        ATB_LOG(ERROR) << "outTensor size and param size are inconsistent."
                       << " OutTensor dimNum: " << outTensors.at(0).desc.shape.dimNum
                       << "param size: " << param_.size.size();
        return ERROR_INVALID_TENSOR_SIZE;
    }

    for (uint64_t i = 0; i < outTensors.at(0).desc.shape.dimNum; i++) {
        if (outTensors.at(0).desc.shape.dims[i] != param_.size.at(i)) {
            ATB_LOG(ERROR) << "outTensor dim and param dim are inconsistent at dim" << i
                           << " param size: " << param_.size.at(i)
                           << " outTensors dim: " << outTensors.at(0).desc.shape.dims[i];
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return ParamCheck(inTensors.at(0).desc);
}

std::shared_ptr<Runner> AsStridedOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<AsStridedOpsRunner>(param_);
}

Status AsStridedOperation::ParamCheck(TensorDesc inTensorDesc) const
{
    if (param_.size.size() > MAX_DIM || param_.size.size() != param_.stride.size() || param_.offset.size() != 1 ||
        param_.offset.at(0) < 0) {
        ATB_LOG(ERROR) << "Wrong params: size: " << param_.size << " stride: " << param_.stride
                       << " offset: " << param_.offset;
        return ERROR_INVALID_PARAM;
    }

    // 总偏移不能超过输入tensor大小
    int64_t numel = static_cast<int64_t>(Utils::GetTensorNumel(inTensorDesc));
    int64_t distance = 0;
    for (uint64_t i = 0; i < param_.size.size(); i++) {
        if (param_.size.at(i) <= 0 || param_.stride.at(i) < 0) {
            ATB_LOG(ERROR) << "Wrong size and stride, size: " << param_.size << " stride: " << param_.stride;
            return ERROR_INVALID_PARAM;
        }
        if (param_.size.at(i) != 1 &&
            std::numeric_limits<int64_t>::max() / (param_.size.at(i) - 1) < param_.stride.at(i)) {
            ATB_LOG(ERROR) << "Calculate offset at dim " << i << " overflow, "
                           << "size: " << param_.size.at(i) << " stride: " << param_.stride.at(i);
            return ERROR_INVALID_PARAM;
        }
        int64_t curOffset = (param_.size.at(i) - 1) * param_.stride.at(i);
        if (std::numeric_limits<int64_t>::max() - curOffset < distance) {
            ATB_LOG(ERROR) << "Calculate the total offset overflow! Wrong params: size: " << param_.size
                           << " stride: " << param_.stride << " offset: " << param_.offset;
            return ERROR_INVALID_PARAM;
        }
        distance += curOffset;
    }

    if (std::numeric_limits<int64_t>::max() - param_.offset.at(0) < distance) {
        ATB_LOG(ERROR) << "Calculate the total offset overflow! Wrong params: size: " << param_.size
                       << " stride: " << param_.stride << " offset: " << param_.offset;
        return ERROR_INVALID_PARAM;
    }

    if (param_.offset.at(0) + distance >= numel) {
        ATB_LOG(ERROR) << "Wrong params, over max length: size: " << param_.size << " stride: " << param_.stride
                       << " offset: " << param_.offset;
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

nlohmann::json AsStridedOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb