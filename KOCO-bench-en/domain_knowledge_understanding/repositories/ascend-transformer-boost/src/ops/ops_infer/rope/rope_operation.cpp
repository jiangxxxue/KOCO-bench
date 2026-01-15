/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_operation.h"
#include "atb/utils/config.h"
#include "rope_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 5;
static const int32_t OUT_TENSOR_NUM = 2;
static const int32_t ROTARY_COEFF_TWO = 2;
static const int32_t ROTARY_COEFF_FOUR = 4;
static const int32_t PARAM_COS = 2;

template <> Status CreateOperation(const infer::RopeParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new (std::nothrow) RopeOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RopeOperation::RopeOperation(const infer::RopeParam &param) : OperationBase("RopeOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RopeOperation");
}

RopeOperation::~RopeOperation() {}

uint32_t RopeOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RopeOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RopeOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                     SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = inTensorDescs.at(1);
    return NO_ERROR;
}

Status RopeOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }

    return ParamCheck(inTensorDescs);
}

Status RopeOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }

    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }

    ATB_LOG(DEBUG) << "outTensors Size:" << outTensors.size();
    return ParamCheck(inTensorDescs);
}

Status RopeOperation::ParamCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.rotaryCoeff != ROTARY_COEFF_TWO && param_.rotaryCoeff != ROTARY_COEFF_FOUR &&
        param_.rotaryCoeff != inTensorDescs.at(PARAM_COS).shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Wrong rotaryCoeff: " << param_.rotaryCoeff;
        return ERROR_INVALID_PARAM;
    }

    if (param_.cosFormat != 0 && param_.cosFormat != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Wrong Param, cosFormat: " << param_.cosFormat;
        return ERROR_INVALID_PARAM;
    }

    return NO_ERROR;
}

Status RopeOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if ((inTensorDescs.at(0).shape.dimNum != 2 && inTensorDescs.at(0).shape.dimNum != 4) || // dimNum: 2, 4
        (inTensorDescs.at(1).shape.dimNum != 2 && inTensorDescs.at(1).shape.dimNum != 4) || // dimNum: 2, 4
        inTensorDescs.at(2).shape.dimNum != 2 || inTensorDescs.at(3).shape.dimNum != 2 ||   // dimNum: 2, index: 3
        inTensorDescs.at(4).shape.dimNum != 1) {                                            // index: 4
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor dimNum is not support, tensor[0-1] only support 2 or 4 dimNum,"
                       << "tensor[2-3] only support 2 dimNum, tensor[4] only support 1 dimNum"
                       << "inTensor[0-4] dimNum: " << inTensorDescs.at(0).shape.dimNum << ", "
                       << inTensorDescs.at(1).shape.dimNum << ", " << inTensorDescs.at(2).shape.dimNum // index: 2
                       << ", " << inTensorDescs.at(3).shape.dimNum                                     // index: 3
                       << ", " << inTensorDescs.at(4).shape.dimNum;                                    // index: 4
        return ERROR_INVALID_TENSOR_DIM;
    }

    int64_t ntokens = inTensorDescs.at(0).shape.dimNum == 2 ?
                          inTensorDescs.at(0).shape.dims[0] :
                          (inTensorDescs.at(0).shape.dims[0] * inTensorDescs.at(0).shape.dims[1]);
    int64_t kDimNum = inTensorDescs.at(1).shape.dimNum;
    if ((kDimNum == 2 && inTensorDescs.at(1).shape.dims[0] != ntokens) || // dimNum: 2
        (kDimNum == 4 &&                                                  // dimNum: 4
         (inTensorDescs.at(1).shape.dims[0] * inTensorDescs.at(1).shape.dims[1]) != ntokens)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "ntokens of q/k should be same. "
                       << "Q : " << TensorUtil::TensorDescToString(inTensorDescs.at(0))
                       << ", K: " << TensorUtil::TensorDescToString(inTensorDescs.at(1));
        return ERROR_INVALID_TENSOR_SIZE;
    }

    if (inTensorDescs.at(2).shape.dims[0] != inTensorDescs.at(3).shape.dims[0] || // index: 2, 3
        inTensorDescs.at(2).shape.dims[1] != inTensorDescs.at(3).shape.dims[1]    // index: 2, 3
    ) {
        ATB_LOG(ERROR) << GetLogPrefix() << "cos sin shape should be same. "
                       << "cos : " << TensorUtil::TensorDescToString(inTensorDescs.at(2))   // index: 2
                       << ", sin: " << TensorUtil::TensorDescToString(inTensorDescs.at(3)); // index: 3
        return ERROR_INVALID_TENSOR_SIZE;
    }

    return HiddenSizeCheck(inTensorDescs);
}

Status RopeOperation::HiddenSizeCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t hiddenSizeQ = inTensorDescs.at(0).shape.dimNum == 4 ?
                              inTensorDescs.at(0).shape.dims[2] * inTensorDescs.at(0).shape.dims[3] :
                              inTensorDescs.at(0).shape.dims[1];
    int64_t hiddenSizeK = inTensorDescs.at(1).shape.dimNum == 4 ?
                              inTensorDescs.at(1).shape.dims[2] * inTensorDescs.at(1).shape.dims[3] :
                              inTensorDescs.at(1).shape.dims[1];
    if (hiddenSizeK == 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSizeK should not be 0.";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (hiddenSizeQ % hiddenSizeK != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "hiddenSizeQ should be an integer multiple of hiddenSizeK, Q: "
                       << TensorUtil::TensorDescToString(inTensorDescs.at(0))
                       << ", K : " << TensorUtil::TensorDescToString(inTensorDescs.at(1))
                       << ", cos : " << TensorUtil::TensorDescToString(inTensorDescs.at(2)) // index: 2
                       << ", sin: " << TensorUtil::TensorDescToString(inTensorDescs.at(3)); // index: 3
        return ERROR_INVALID_TENSOR_SIZE;
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> RopeOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_ROPE);
    Runner *runner = pool.MallocRunner<RopeOpsRunner, infer::RopeParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<RopeOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json RopeOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
