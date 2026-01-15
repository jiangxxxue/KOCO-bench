/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_operation.h"
#include <sstream>
#include <cstring>
#include <mki/utils/env/env.h>
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "linear_ops_runner.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static constexpr uint32_t IN_TENSOR_NUM_2 = 2;
static constexpr uint32_t IN_TENSOR_NUM_3 = 3;
static constexpr uint32_t IN_TENSOR_NUM_4 = 4;
static constexpr uint32_t OUT_TENSOR_NUM = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr uint64_t DIM_NUM_2 = 2;
static constexpr uint64_t DIM_NUM_3 = 3;
static constexpr uint64_t DIM_NUM_4 = 4;

namespace atb {
bool MatmulParamCheck(const infer::LinearParam &opParam, ExternalError &error)
{
    if (opParam.quantMode != atb::infer::LinearParam::QUANT_UNDEFINED) {
        error.errorData = OperationUtil::ConcatInfo(error.errorData, "quantMode = ", opParam.quantMode);
        error.errorDesc = "QuantMode is not the default value. When outDataType is undefind, quantMode needs to be undefind.";
        error.solutionDesc = "Please check the quantMode value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    if (opParam.enAccum) {
        error.errorData = OperationUtil::ConcatInfo(error.errorData, "enAccum = ", opParam.enAccum);
        if (!GetSingleton<Config>().Is910B()) {
            error.errorDesc = "Platform is not Atlas 800I A2 inference product, enAccum should be false,";
            error.solutionDesc = "Please check the platform and value of input params.";
            ATB_LOG(ERROR) << error;
            return false;
        }
        if (opParam.hasBias) {
            error.errorDesc = "enAccum and hasBias can not both be true!";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", hasBias = ", opParam.hasBias);
            ATB_LOG(ERROR) << error;
            return false;
        }
    }
    return true;
}

bool MatmulEinParamCheck(const infer::LinearParam &opParam, ExternalError &error)
{
    if (opParam.enAccum) {
        error.errorDesc = "When MatmulEin is used, enAccum should be false,";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    } else if (opParam.transposeA) {
        error.errorDesc = "When MatmulEin is used, transposeA should be false,";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    } else if (opParam.hasBias) {
        error.errorDesc = "When MatmulEin is used, hasBias should be false,";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    } else if (opParam.outDataType != ACL_DT_UNDEFINED) {
        error.errorDesc = "When MatmulEin is used, outDataType should be ACL_DT_UNDEFINED,";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    } else if (!GetSingleton<Config>().Is910B()) {
        error.errorDesc = "Platform is not Atlas 800I A2 inference product, MatmulType should be 0,";
        error.solutionDesc = "Please check the platform and value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    return true;
}

bool MatmulDequantFloat16ParamCheck(const infer::LinearParam &opParam, ExternalError &error)
{
    if (!GetSingleton<Config>().Is910B()) {
        error.errorDesc = "Platform is not Atlas 800I A2 inference product, outDataType is ACL_FLOAT16, ";
        error.solutionDesc = "Please check the platform and value of input params.";
        if (opParam.transposeA) {
            error.errorDesc = error.errorDesc + "transposeA should be false.";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", transposeA = ", opParam.transposeA);
            ATB_LOG(ERROR) << error;
            return false;
        }
        if (!opParam.transposeB) {
            error.errorDesc = error.errorDesc + "transposeB should be true.";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", transposeB = ", opParam.transposeB);
            ATB_LOG(ERROR) << error;
            return false;
        }
        if (!opParam.hasBias && !GetSingleton<Config>().Is310B()) {
            error.errorDesc = error.errorDesc + "hasBias should be true.";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", hasBias = ", opParam.hasBias);
            ATB_LOG(ERROR) << error;
            return false;
        }
        if (opParam.quantMode == infer::LinearParam::PER_TOKEN) {
            error.errorDesc = error.errorDesc + "quantMode should be QUANT_UNDEFINED or PER_CHANNEL.";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", quantMode = ", opParam.quantMode);
            ATB_LOG(ERROR) << error;
            return false;
        }
    } else if (opParam.quantMode == infer::LinearParam::PER_TOKEN && opParam.hasBias) {
        error.errorDesc = "PER_TOKEN quantMode is not supported when hasBias is true";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    return true;
}

bool MatmulDequantBf16ParamCheck(const infer::LinearParam &opParam, ExternalError &error)
{
    (void)opParam;
    if (!GetSingleton<Config>().Is910B()) {
        error.errorDesc = "Platform is not Atlas 800I A2 inference product, outDataType should not be ACL_BF16.";
        error.solutionDesc = "Please check the platform and value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    } else if (opParam.quantMode == infer::LinearParam::PER_TOKEN && opParam.hasBias) {
        error.errorDesc = "PER_TOKEN quantMode is not supported when hasBias is true";
        error.solutionDesc = "Please check the value of input params.";
        ATB_LOG(ERROR) << error;
        return false;
    }
    return true;
}

bool MatmulUndefindCheck(const infer::LinearParam &opParam, ExternalError &error)
{
    switch (opParam.outDataType) {
        case ACL_DT_UNDEFINED:
            {
                if (!MatmulParamCheck(opParam, error)) {
                    return false;
                }
                break;
            }
        case ACL_FLOAT16:
            {
                if (opParam.enAccum) {
                    error.errorDesc = "outDataType is not ACL_DT_UNDEFINED, enAccum should be false.";
                    error.errorData = OperationUtil::ConcatInfo(error.errorData, ", enAccum = ", opParam.enAccum);
                    ATB_LOG(ERROR) << error;
                    return false;
                }
                if (!MatmulDequantFloat16ParamCheck(opParam, error)) {
                    return false;
                }
                break;
            }
        case ACL_BF16:
            {
                if (opParam.enAccum) {
                    error.errorDesc = "outDataType is not ACL_DT_UNDEFINED, enAccum should be false.";
                    error.errorData = OperationUtil::ConcatInfo(error.errorData, ", enAccum = ", opParam.enAccum);
                    ATB_LOG(ERROR) << error;
                    return false;
                }
                if (!MatmulDequantBf16ParamCheck(opParam, error)) {
                    return false;
                }
                break;
            }
        default:
            {
                error.errorDesc = "outDataType should be ACL_DT_UNDEFINED/ACL_FLOAT16/ACL_BF16.";
                ATB_LOG(ERROR) << error;
                return false;
            }
    }
    return true;
}

template <> Status CreateOperation(const infer::LinearParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ExternalError error;
    error.errorType = ERROR_INVALID_PARAM;
    error.errorData = OperationUtil::ConcatInfo("outDataType = ", opParam.outDataType);
    error.solutionDesc = "Please check the value of input params.";
    if (opParam.matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        if (!MatmulEinParamCheck(opParam, error)) {
            return ERROR_INVALID_PARAM;
        }
    } else if (opParam.matmulType == infer::LinearParam::MATMUL_UNDEFINED) {
        if (!MatmulUndefindCheck(opParam, error)) {
            return ERROR_INVALID_PARAM;
        }
    } else {
        error.errorDesc = "matmulType uses an nonexistent type.";
        error.errorData = OperationUtil::ConcatInfo(error.errorData, ", matmulType = ", opParam.matmulType);

        ATB_LOG(ERROR) << error;
        return ERROR_INVALID_PARAM;
    }

    *operation = new (std::nothrow) LinearOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LinearOperation::LinearOperation(const infer::LinearParam &param) : OperationBase("LinearOperation"), param_(param)
{
    commonCheckParam_ = param_;
    std::stringstream opIrKey;
    opIrKey << "LinearOperationMatmul";
    if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        opIrKey << "EinSum";
        opIrKey << (param_.hasBias ? "ElewiseAdd" : "");
    } else if (param_.outDataType == ACL_DT_UNDEFINED) {
        if (param_.enAccum) {
            opIrKey << "Accum";
        } else {
            opIrKey << (param_.hasBias ? "WithBias" : "");
            opIrKey << (GetSingleton<Config>().Is910B() ? "Atlas800IA2" : "NotAtlas800IA2");
        }
    } else {
        opIrKey << (param_.hasBias ? "DequantWithBias" : "Dequant");
        if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
            opIrKey << "PerToken";
        }
        if (param_.outDataType == ACL_FLOAT16) {
            opIrKey << "Float16";
            opIrKey << (GetSingleton<Config>().Is910B()                   ? "Atlas800IA2" :
                        GetSingleton<Config>().Is910A() && param_.hasBias ? "AtlasTrain" :
                                                                            "NotAtlas800IA2");
        } else if (param_.outDataType == ACL_BF16) {
            opIrKey << "Bf16";
            opIrKey << (GetSingleton<Config>().Is910B() ? "Atlas800IA2" : "NotAtlas800IA2");
        }
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey.str());
}

LinearOperation::~LinearOperation() {}

uint32_t LinearOperation::GetInputNum() const
{
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        if (param_.hasBias || param_.enAccum) {
            return IN_TENSOR_NUM_3;
        } else {
            return IN_TENSOR_NUM_2;
        }
    } else {
        if (param_.hasBias) {
            return IN_TENSOR_NUM_4;
        } else if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
            return IN_TENSOR_NUM_4;
        } else {
            return IN_TENSOR_NUM_3;
        }
    }
}

uint32_t LinearOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status LinearOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        if (param_.enAccum) {
            outTensorDescs.at(0).dtype = ACL_FLOAT;
        }
    } else {
        outTensorDescs.at(0).dtype = param_.outDataType;
    }
    int64_t m = OperationUtil::GetXTensorM(inTensorDescs.at(0), param_.transposeA, param_.matmulType);
    int64_t n = OperationUtil::GetYTensorN(inTensorDescs.at(1), param_.transposeB);
    uint64_t xDimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).shape.dims[xDimNum - DIM_2] = m;
    outTensorDescs.at(0).shape.dims[xDimNum - 1] = n;
    if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM) {
        outTensorDescs.at(0).shape.dims[0] =
            param_.transposeA ? inTensorDescs.at(0).shape.dims[DIM_2] : inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[DIM_2] = n;
    }
    return NO_ERROR;
}

Status LinearOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status LinearOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }
    return OutTensorCheck(inTensorDescs, outTensors);
}

std::shared_ptr<Runner> LinearOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<LinearOpsRunner>(param_);
}

nlohmann::json LinearOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

Status LinearOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    size_t inTensorId = 0;
    const TensorDesc &xTensorDesc = inTensorDescs.at(inTensorId++);
    const TensorDesc &weightTensorDesc = inTensorDescs.at(inTensorId++);
    if (GetSingleton<Config>().Is310B() && xTensorDesc.dtype == aclDataType::ACL_FLOAT16 && param_.transposeA) {
        ATB_LOG(ERROR) << "Atlas 200I/500 A2 inference product does not support transposeA=true!";
        return ERROR_INVALID_PARAM;
    }
    Status status = NO_ERROR;
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        status = XWeightCheck(xTensorDesc, weightTensorDesc, DEFAULT_ALIGN);
        if (status != NO_ERROR) {
            return status;
        }
        if (param_.hasBias) {
            const TensorDesc &biasTensorDesc = inTensorDescs.at(inTensorId++);
            status = BiasCheck(biasTensorDesc, weightTensorDesc);
            if (status != NO_ERROR) {
                return status;
            }
        } else if (param_.enAccum) {
            const TensorDesc &accumTensorDesc = inTensorDescs.at(inTensorId++);
            status = AccumCheck(accumTensorDesc, xTensorDesc, weightTensorDesc);
            if (status != NO_ERROR) {
                return status;
            }
        }
    } else {
        status = XWeightCheck(xTensorDesc, weightTensorDesc, INT8_ALIGN);
        if (status != NO_ERROR) {
            return status;
        }
        if (param_.hasBias) {
            const TensorDesc &biasTensorDesc = inTensorDescs.at(inTensorId++);
            status = BiasCheck(biasTensorDesc, weightTensorDesc);
            if (status != NO_ERROR) {
                return status;
            }
        }
        const TensorDesc &deqScaleTensorDesc = inTensorDescs.at(inTensorId++);
        if (param_.quantMode == infer::LinearParam::PER_TOKEN) {
            const TensorDesc &perTokenDeqScaleTensorDesc = inTensorDescs.at(inTensorId++);
            status = PerTokenDeqScaleCheck(deqScaleTensorDesc, weightTensorDesc, xTensorDesc, perTokenDeqScaleTensorDesc);
        } else {
            status = DeqScaleCheck(deqScaleTensorDesc, weightTensorDesc);
        }
        if (status != NO_ERROR) {
            return status;
        }
    }
    return NO_ERROR;
}

Status LinearOperation::XWeightCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc,
                                     int64_t align) const
{
    if (!XWeightDimNumCheck(xTensorDesc, weightTensorDesc)) {
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (!XWeightBatchCheck(xTensorDesc, weightTensorDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!XWeightKEqualCheck(xTensorDesc, weightTensorDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!WeightAlignCheck(weightTensorDesc, align)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status LinearOperation::BiasCheck(const TensorDesc &biasTensorDesc, const TensorDesc &weightTensorDesc) const
{
    ExternalError error;

    if (biasTensorDesc.shape.dimNum != 1 && biasTensorDesc.shape.dimNum != DIM_NUM_2) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor2 dim num should be 1 or 2,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dimNum = ", biasTensorDesc.shape.dimNum);
        error.solutionDesc = "Please check shape of inTensor2.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    error.errorType = ERROR_INVALID_TENSOR_DIM;
    int64_t biasBatch = biasTensorDesc.shape.dimNum == 1 ? 1 : biasTensorDesc.shape.dims[0];
    int64_t weightBatch = OperationUtil::GetTensorBatch(weightTensorDesc, atb::infer::LinearParam::MATMUL_UNDEFINED);
    if (biasBatch != weightBatch) {
        error.errorDesc = "value of inTensor2 batch and value of inTensor1 batch should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor2 batch = ", biasBatch, ", inTensor1 batch = ", weightBatch);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    int64_t biasN = biasTensorDesc.shape.dims[biasTensorDesc.shape.dimNum - 1];
    int64_t weightN = OperationUtil::GetYTensorN(weightTensorDesc, param_.transposeB);
    if (biasN != weightN) {
        error.errorDesc = "value of inTensor2 n and value of inTensor1 n should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 n = ", biasN, ", inTensor1 n = ", weightN);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LinearOperation::AccumCheck(const TensorDesc &accumTensorDesc, const TensorDesc &xTensorDesc,
                                   const TensorDesc &weightTensorDesc) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.solutionDesc = "Please check shape of inTensors";

    if (accumTensorDesc.shape.dimNum != xTensorDesc.shape.dimNum) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor2 dim num and inTensor0 dim num should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 dimNum = ", accumTensorDesc.shape.dimNum,
                                                    ", inTensor0 dimNum = ", xTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t accumBatch = OperationUtil::GetTensorBatch(accumTensorDesc, param_.matmulType);
    int64_t xBatch = OperationUtil::GetTensorBatch(xTensorDesc, param_.matmulType);
    if (accumBatch != xBatch) {
        error.errorDesc = "value of inTensor2 batch and inTensor0 batch should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 batch = ", accumBatch, ", inTensor0 batch = ", xBatch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t accumM = accumTensorDesc.shape.dims[accumTensorDesc.shape.dimNum - DIM_2];
    int64_t xM = OperationUtil::GetXTensorM(xTensorDesc, param_.transposeA, param_.matmulType);
    if (accumM != xM) {
        error.errorDesc = "value of inTensor2 m and inTensor0 m should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 m = ", accumM, ", inTensor0 m = ", xM);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t accumN = accumTensorDesc.shape.dims[accumTensorDesc.shape.dimNum - 1];
    int64_t weightN = OperationUtil::GetYTensorN(weightTensorDesc, param_.transposeB);
    if (accumN != weightN) {
        error.errorDesc = "value of inTensor2 n and inTensor1 n should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor2 n = ", accumN, ", inTensor1 n = ", weightN);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LinearOperation::DeqScaleCheck(const TensorDesc &deqScaleTensorDesc, const TensorDesc &weightTensorDesc) const
{
    ExternalError error;
    error.solutionDesc = "Please check the shape of inTensors.";
    if (deqScaleTensorDesc.shape.dimNum != 1 && deqScaleTensorDesc.shape.dimNum != DIM_NUM_2) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor3 dim num should be 1 or 2,";
        error.errorData = OperationUtil::ConcatInfo("inTensor3 dimNum = ", deqScaleTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    error.errorType = ERROR_INVALID_TENSOR_DIM;
    int64_t deqScaleBatch = deqScaleTensorDesc.shape.dimNum == 1 ? 1 : deqScaleTensorDesc.shape.dims[0];
    int64_t weightBatch = OperationUtil::GetTensorBatch(weightTensorDesc, param_.matmulType);
    if (deqScaleBatch != weightBatch) {
        error.errorDesc = "value of inTensor3 batch and value of inTensor1 batch should be equal,";
        error.errorData =
            OperationUtil::ConcatInfo("inTensor3 batch = ", deqScaleBatch, ", inTensor1 batch = ", weightBatch);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    int64_t deqScaleN = deqScaleTensorDesc.shape.dims[deqScaleTensorDesc.shape.dimNum - 1];
    int64_t weightN = OperationUtil::GetYTensorN(weightTensorDesc, param_.transposeB);
    if (deqScaleN != weightN) {
        error.errorDesc = "value of inTensor3 n and value of inTensor1 n should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor3 n = ", deqScaleN, ", inTensor1 n = ", weightN);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LinearOperation::PerTokenDeqScaleCheck(const TensorDesc &deqScaleTensorDesc, const TensorDesc &weightTensorDesc,
                                              const TensorDesc &xTensorDesc, const TensorDesc &perTokendeqScaleTensorDesc) const
{
    ExternalError error;
    error.solutionDesc = "Please check the shape of inTensors.";
    if (deqScaleTensorDesc.shape.dimNum != 1 || perTokendeqScaleTensorDesc.shape.dimNum != 1) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "inTensor3 and inTensor4 dim num should be 1,";
        error.errorData = OperationUtil::ConcatInfo("inTensor3 dimNum = ", deqScaleTensorDesc.shape.dimNum);
        error.errorData = OperationUtil::ConcatInfo("inTensor4 dimNum = ", perTokendeqScaleTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }

    error.errorType = ERROR_INVALID_TENSOR_DIM;

    int64_t deqScaleN = deqScaleTensorDesc.shape.dims[0];
    int64_t perTokendeqScaleM = perTokendeqScaleTensorDesc.shape.dims[0];
    int64_t weightN = OperationUtil::GetYTensorN(weightTensorDesc, param_.transposeB);
    int64_t xM = OperationUtil::GetXTensorM(xTensorDesc, param_.transposeA, param_.matmulType);
    if (deqScaleN != weightN) {
        error.errorDesc = "value of inTensor3 n and value of inTensor1 n should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor3 n = ", deqScaleN, ", inTensor1 n = ", weightN);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    if (perTokendeqScaleM != xM) {
        error.errorDesc = "value of inTensor4 m and value of inTensor1 m should be equal,";
        error.errorData = OperationUtil::ConcatInfo("inTensor4 m = ", perTokendeqScaleM, ", inTensor1 m = ", xM);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

Status LinearOperation::OutTensorCheck(const SVector<TensorDesc> &inTensorDescs,
                                       const SVector<Tensor> &outTensors) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    const TensorDesc &weightTensorDesc = inTensorDescs.at(1);
    const TensorDesc &outTensorDesc = outTensors.at(0).desc;
    ExternalError error;
    if (outTensorDesc.shape.dimNum != xTensorDesc.shape.dimNum) {
        error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
        error.errorDesc = "outTensor dimNum and inTensor0 dimNum should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor dimNum = ", outTensorDesc.shape.dimNum,
                                                    ", inTensor0 dimNum = ", xTensorDesc.shape.dimNum);
        error.solutionDesc = "Please check shape of outTensor.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t outBatch = OperationUtil::GetTensorBatch(outTensorDesc, param_.matmulType);
    int64_t xBatch = OperationUtil::GetTensorBatch(xTensorDesc, param_.matmulType);
    if (outBatch != xBatch) {
        error.errorType = ERROR_INVALID_TENSOR_DIM;
        error.errorDesc = "value of outTensor batch and inTensor0 batch should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor batch = ", outBatch, ", inTensor0 batch = ", xBatch);
        error.solutionDesc = "Please check shape of outTensor.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t outM = OperationUtil::GetOutTensorM(outTensorDesc, param_.matmulType);
    int64_t xM = OperationUtil::GetXTensorM(xTensorDesc, param_.transposeA, param_.matmulType);
    if (outM != xM) {
        error.errorType = ERROR_INVALID_TENSOR_DIM;
        error.errorDesc = "value of outTensor m and inTensor0 m should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor m = ", outM, ", inTensor0 m = ", xM);
        error.solutionDesc = "Please check shape of outTensor.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    int64_t outN = OperationUtil::GetOutTensorN(outTensorDesc);
    int64_t weightN = OperationUtil::GetYTensorN(weightTensorDesc, param_.transposeB);
    if (outN != weightN) {
        error.errorType = ERROR_INVALID_TENSOR_DIM;
        error.errorDesc = "value of outTensor n and inTensor1 n should be equal,";
        error.errorData = OperationUtil::ConcatInfo("outTensor n = ", outN, ", inTensor1 n = ", weightN);
        error.solutionDesc = "Please check shape of outTensor.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return error.errorType;
    }
    return NO_ERROR;
}

bool LinearOperation::XWeightDimNumCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.solutionDesc = "Please check format and shape of inTensors.";

    if (xTensorDesc.shape.dimNum != DIM_NUM_2 && xTensorDesc.shape.dimNum != DIM_NUM_3) {
        error.errorDesc = "inTensor0 dim num should be 2 or 3,";
        error.errorData = OperationUtil::ConcatInfo("inTensor0 dimNum = ", xTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return false;
    }

    if (weightTensorDesc.format == ACL_FORMAT_ND) {
        if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM && weightTensorDesc.shape.dimNum != DIM_NUM_3) {
            error.errorDesc = "When EIN is used inTensor1 dim num should be 3,";
            error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        } else if (weightTensorDesc.shape.dimNum != DIM_NUM_2 && weightTensorDesc.shape.dimNum != DIM_NUM_3) {
            error.errorDesc = "inTensor1 dim num should be 2 or 3,";
            error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        }
    } else if (weightTensorDesc.format == ACL_FORMAT_FRACTAL_NZ) {
        if (param_.matmulType == infer::LinearParam::MATMUL_EIN_SUM && weightTensorDesc.shape.dimNum != DIM_NUM_4) {
            error.errorDesc = "When EIN is used inTensor1 dim num should be 4,";
            error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        } else if (weightTensorDesc.shape.dimNum != DIM_NUM_2 && weightTensorDesc.shape.dimNum != DIM_NUM_3 &&
                   weightTensorDesc.shape.dimNum != DIM_NUM_4) {
            error.errorDesc = "inTensor1 dim num should be 2 or 3 or 4,";
            error.errorData = OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        }
    }
    if (xTensorDesc.shape.dimNum == DIM_NUM_2 && weightTensorDesc.shape.dimNum == DIM_NUM_3) {
        error.errorDesc = "inTensor1 dim num is 3, inTensor0 dim num should not be 2,";
        error.errorData =
            OperationUtil::ConcatInfo("enAccum = ", param_.enAccum, ", inTensor0 dimNum = ", xTensorDesc.shape.dimNum,
                                      ", inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return false;
    } else if (!PerTokenXWeightDimNumCheck(xTensorDesc, weightTensorDesc)) {
        return false;
    }
    return true;
}

bool LinearOperation::PerTokenXWeightDimNumCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    error.solutionDesc = "Please check format and shape of inTensors.";
    if (param_.quantMode == infer::LinearParam::PER_TOKEN && xTensorDesc.shape.dimNum == DIM_NUM_3 && weightTensorDesc.shape.dimNum == DIM_NUM_2) {
        error.errorDesc = "When quantMode is PER_TOKEN and inTensor0 dim num is 3, inTensor1 dim num cannot be 2,";
        error.errorData = OperationUtil::ConcatInfo("quantMode = ", param_.quantMode, ", inTensor0 dimNum = ", xTensorDesc.shape.dimNum,
                                                    ", inTensor1 dimNum = ", weightTensorDesc.shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return false;
    }
    return true;
}

bool LinearOperation::XWeightBatchCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const
{
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    int64_t xBatch = OperationUtil::GetTensorBatch(xTensorDesc, param_.matmulType);
    int64_t weightBatch = OperationUtil::GetTensorBatch(weightTensorDesc, atb::infer::LinearParam::MATMUL_UNDEFINED);
    if (xBatch != weightBatch) {
        if (param_.transposeA) {
            error.errorDesc = "transposeA is true, values of inTensor0 batch and inTensor1 batch should be equal,";
            error.errorData = OperationUtil::ConcatInfo(
                "transposeA: ", param_.transposeA, ", inTensor0 batch = ", xBatch, ", inTensor1 batch = ", weightBatch);
            error.solutionDesc = "Please check the value of params and shape of inTensors.";
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        }
        if (weightTensorDesc.shape.dimNum == DIM_NUM_3) {
            error.errorDesc = "inTensor1 dimNum is 3, values of inTensor0 batch and inTensor1 batch should be equal,";
            error.errorData =
                OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum,
                                          "inTensor0 batch = ", xBatch, ", inTensor1 batch = ", weightBatch);
            error.solutionDesc = "Please check the shape of inTensors.";
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        }
        if (weightBatch != 1) {
            error.errorDesc =
                "inTensor1 dimNum is not 3, value of inTensor1 batch should be 1 or equal to value of inTensor0 batch,";
            error.errorData =
                OperationUtil::ConcatInfo("inTensor1 dimNum = ", weightTensorDesc.shape.dimNum,
                                          "inTensor0 batch = ", xBatch, ", inTensor1 batch = ", weightBatch);
            error.solutionDesc = "Please check the shape of inTensors.";
            ATB_LOG(ERROR) << GetLogPrefix() << error;
            return false;
        }
    }
    return true;
}

bool LinearOperation::XWeightKEqualCheck(const TensorDesc &xTensorDesc, const TensorDesc &weightTensorDesc) const
{
    int64_t xK = OperationUtil::GetXTensorK(xTensorDesc, param_.transposeA);
    int64_t weightK = OperationUtil::GetYTensorK(weightTensorDesc, param_.transposeB);
    if (xK == weightK) {
        return true;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    error.errorDesc = "value of inTensor0 k and value of inTensor1 k should be equal,";
    error.errorData = OperationUtil::ConcatInfo("inTensor0 k = ", xK, ", inTensor1 k = ", weightK);
    error.solutionDesc = "Please check the shape of inTensors.";
    ATB_LOG(ERROR) << GetLogPrefix() << error;
    return false;
}

bool LinearOperation::WeightAlignCheck(const TensorDesc &weightTensorDesc, int64_t align) const
{
    if (weightTensorDesc.shape.dimNum != DIM_NUM_4) {
        return true;
    }
    ExternalError error;
    error.errorType = ERROR_INVALID_TENSOR_DIM;
    int64_t dim2 = weightTensorDesc.shape.dims[DIM_2];
    if (dim2 % DEFAULT_ALIGN != 0) {
        error.errorDesc = "inTensor1 dim2 should be divisible by 16,";
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim2 = ", dim2);
        error.solutionDesc = "Please check the shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return false;
    }
    int64_t dim3 = weightTensorDesc.shape.dims[DIM_3];
    if (dim3 != align) {
        error.errorDesc = OperationUtil::ConcatInfo("inTensor1 dim3 should be ", align, ",");
        error.errorData = OperationUtil::ConcatInfo("inTensor1 dim3 = ", dim3);
        error.solutionDesc = "Please check the value of params and shape of inTensors.";
        ATB_LOG(ERROR) << GetLogPrefix() << error;
        return false;
    }
    return true;
}
} // namespace atb