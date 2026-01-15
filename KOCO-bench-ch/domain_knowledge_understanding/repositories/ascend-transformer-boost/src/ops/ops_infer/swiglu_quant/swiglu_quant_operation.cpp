/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swiglu_quant_operation.h"
#include <algorithm>
#include "swiglu_quant_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/utils_internal.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 2;
static const uint64_t INPUT_TENSOR_DIM_NUM = 2;
static const int64_t HIDDEN_SIZE_DIM_BASE = 32;
static const int64_t HIDDEN_SIZE_BYTE_SIZE_FACTOR = 2;
static const int64_t MAX_UB_SIZE = 192LL * 1024LL; // 192KB
static const int64_t UB_FACTOR = 25;

template <> Status CreateOperation(const infer::SwigluQuantParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "SwigluQuant only supports 800I A2/A3 inference products";
        return ERROR_INVALID_PARAM;
    }
    ATB_LOG(INFO) << "CreateOperation SwigluQuantParam quantType: " << opParam.quantType;
    if (opParam.quantType != infer::SwigluQuantParam::QUANT_TYPE_PER_TOKEN) {
        ATB_LOG(ERROR) << "param quantType should be QUANT_TYPE_PER_TOKEN";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) SwigluQuantOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

SwigluQuantOperation::SwigluQuantOperation(const infer::SwigluQuantParam &param)
    : OperationBase("SwigluQuantOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SwigluQuantOperation");
}

SwigluQuantOperation::~SwigluQuantOperation() {}

uint32_t SwigluQuantOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SwigluQuantOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SwigluQuantOperation::DimCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t dimNum = inTensorDesc.shape.dimNum;
    if (dimNum != INPUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << "dimNum of inTensor should be 2, but inTensor dimNum is : " << dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return NO_ERROR;
}

Status SwigluQuantOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs.at(0));
    if (st != NO_ERROR) {
        return st;
    }
    int64_t hiddenSize = inTensorDescs.at(0).shape.dims[INPUT_TENSOR_DIM_NUM - 1];
    if (hiddenSize % HIDDEN_SIZE_BYTE_SIZE_FACTOR != 0) {
        ATB_LOG(ERROR) << "Expected inTensor dims[1] to be a multiple of 2, but got: " << hiddenSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    hiddenSize /= 2; // 2: inTensor dim: [ntokens, 2 * hidden_size]
    int64_t dataItemSize = static_cast<int64_t>(UtilsInternal::GetDataTypeSize(inTensorDescs.at(0).dtype));
    if (dataItemSize == 0) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t dimBase = HIDDEN_SIZE_DIM_BASE / dataItemSize;
    if (UtilsInternal::AlignUp(hiddenSize, dimBase) >= MAX_UB_SIZE / UB_FACTOR) {
        ATB_LOG(ERROR) << "Expected 32-byte-up-aligned 25 * hiddenSize * item byte size < 192KB, but got hiddenSize: "
                       << hiddenSize << ", with dtype: " << inTensorDescs.at(0).dtype;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status SwigluQuantOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> targetOutTensorDescs = {};
    targetOutTensorDescs.reserve(OUT_TENSOR_NUM);
    targetOutTensorDescs.resize(OUT_TENSOR_NUM);
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    InferShapeImpl(inTensorDescs, targetOutTensorDescs);
    for (size_t i = 0; i < OUT_TENSOR_NUM; ++i) {
        if (!TensorUtil::TensorDescEqual(outTensors.at(i).desc, targetOutTensorDescs.at(i))) {
            ATB_LOG(ERROR) << "invalid outTensor[" << i << "] shape ";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status SwigluQuantOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = ACL_INT8;
    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).dtype = ACL_FLOAT;
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1] / 2; // 2: chunk num
    outTensorDescs.at(1).shape.dimNum = 1;
    outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    return NO_ERROR;
}

std::shared_ptr<Runner> SwigluQuantOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SwigluQuantOpsRunner>(param_);
}

nlohmann::json SwigluQuantOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb