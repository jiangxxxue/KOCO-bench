/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_operation.h"
#include <cmath>
#include "rms_norm_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT_SIX = 6;
static const uint32_t IN_TENSOR_COUNT_FIVE = 5;
static const uint32_t IN_TENSOR_COUNT_FOUR = 4;
static const uint32_t IN_TENSOR_COUNT_THREE = 3;
static const uint32_t IN_TENSOR_COUNT_TWO = 2;
static const uint32_t OUT_TENSOR_COUNT_TWO = 2;
static const uint32_t OUT_TENSOR_COUNT_THREE = 3;
static const uint32_t IN_TENSOR_X = 0;
static const uint32_t IN_TENSOR_GAMMA = 1;
static const uint32_t IN_TENSOR_SCALE = 3;
static const uint32_t IN_TENSOR_OFFSET = 4;
static const uint32_t IN_TENSOR_SCALE_PRENORM = 4;
static const uint32_t IN_TENSOR_OFFSET_PRENORM = 5;
static const uint32_t OUT_TENSOR_X = 0;
static const uint32_t OUT_TENSOR_QAUNT = 1;
static const uint32_t OUT_TENSOR_RSTD = 1;
static const uint32_t ALIGNMENT = 32;
static const uint32_t DYNAMIC_QUANT_MAX_TENSOR_LENGTH = 12288;
static const uint32_t TENSOR_DIM_ONE = 1;
static const uint32_t DIM_SHAPE_ONE = 1;
static const uint32_t QUANT_TENSOR_COUNT = 2;
static const uint32_t GAMMA_DIM_ONE = 1;
static const uint32_t GAMMA_DIM_TWO = 2;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.

bool EpsilonCheck(const infer::RmsNormParam &opParam)
{
    if (std::fabs(opParam.normParam.epsilon) < THRESHOLD || std::fabs(opParam.preNormParam.epsilon) < THRESHOLD ||
        std::fabs(opParam.normParam.layerNormEps) < THRESHOLD || std::fabs(opParam.postNormParam.epsilon) < THRESHOLD) {
        return false;
    }
    return true;
}

bool CheckRmsNormFor310B(const infer::RmsNormParam &opParam, atb::ExternalError &error)
{
    if (opParam.layerType == infer::RmsNormParam::RMS_NORM_NORM &&
        (opParam.normParam.quantType == infer::QUANT_UNQUANT
         || opParam.normParam.quantType == infer::QUANT_INT8)) {
        return true;
    } else {
        error.errorDesc = "Atlas 500 A2 product only supports RMS_NORM_NORM layerType "
                          "in QUANT_UNQUANT、QUANT_INT8 quantType";
        error.errorData = OperationUtil::ConcatInfo(error.errorData, ", layerType = ", opParam.layerType,
                                                    ", quantType = ", opParam.normParam.quantType);
        ATB_LOG(ERROR) << error;
        return false;
    }
}

bool RmsNormTypeCheck(const infer::RmsNormParam &opParam)
{
    atb::ExternalError error;
    error.errorType = ERROR_INVALID_PARAM;
    if (GetSingleton<Config>().Is910A()) {
        if (opParam.layerType == infer::RmsNormParam::RMS_NORM_NORM &&
            (opParam.normParam.quantType == infer::QUANT_UNQUANT ||
             opParam.normParam.quantType == infer::QUANT_INT8)) {
            return true;
        } else {
            error.errorDesc = "layerType and quantType combination is invalid";
            error.errorData = OperationUtil::ConcatInfo(error.errorData, ", layerType = ", opParam.layerType,
                                                        ", quantType = ", opParam.normParam.quantType);
            ATB_LOG(ERROR) << error;
            return false;
        }
    }
    if (GetSingleton<Config>().Is310B()) {
        return CheckRmsNormFor310B(opParam, error);
    }
    if (!GetSingleton<Config>().Is910B()) {
        if (opParam.postNormParam.quantType == infer::QUANT_INT8) {
            ATB_LOG(ERROR) << "RMS_NORM_POSTNORM layerType & QUANT_INT8 quantType\
                only supports Atlas 800I A2 product and Atlas A3 product!";
            return false;
        }
    }
    error.errorDesc = "layerType and quantType combination is invalid";
    error.errorData = OperationUtil::ConcatInfo(error.errorData, ", layerType = ", opParam.layerType);
    if (opParam.layerType == infer::RmsNormParam::RMS_NORM_NORM) {
        if (opParam.normParam.quantType == infer::QUANT_UNQUANT || opParam.normParam.quantType == infer::QUANT_INT8) {
            return true;
        }
        error.errorData = OperationUtil::ConcatInfo(error.errorData, ", quantType = ", opParam.normParam.quantType);
    } else if (opParam.layerType == infer::RmsNormParam::RMS_NORM_PRENORM) {
        if (opParam.preNormParam.quantType == infer::QUANT_UNQUANT ||
            opParam.preNormParam.quantType == infer::QUANT_INT8) {
            return true;
        }
        error.errorData = OperationUtil::ConcatInfo(error.errorData, ", quantType = ", opParam.preNormParam.quantType);
    } else if (opParam.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM) {
        if (opParam.postNormParam.quantType == infer::QUANT_UNQUANT ||
            opParam.postNormParam.quantType == infer::QUANT_INT8) {
            return true;
        }
        error.errorData = OperationUtil::ConcatInfo(error.errorData, ", quantType = ", opParam.postNormParam.quantType);
    }
    error.solutionDesc = "Please check the document for valid layerType and quantType.";
    ATB_LOG(ERROR) << error;
    return false;
}

template <> Status CreateOperation(const infer::RmsNormParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    OP_PARAM_RSV_CHECK(opParam.normParam);
    OP_PARAM_RSV_CHECK(opParam.preNormParam);
    OP_PARAM_RSV_CHECK(opParam.postNormParam);
    ATB_LOG(INFO) << "CreateOperation with RmsNormParam: " << OpParamToJson(opParam);
    if (!EpsilonCheck(opParam)) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (!RmsNormTypeCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (opParam.layerType == infer::RmsNormParam::RMS_NORM_NORM &&
        opParam.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
        ATB_LOG(ERROR) << "dynamicQuantType not support DYNAMIC_QUANT_ASYMMETRIC";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) RmsNormOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RmsNormOperation::RmsNormOperation(const infer::RmsNormParam &param) : OperationBase("RmsNormOperation"), param_(param)
{
    is910b_ = GetSingleton<Config>().Is910B();
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_UNQUANT) {
            operationIr_ =
                param_.normParam.rstd ?
                    GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationNORMAndQUANTUNDEFINEDWithRSTD") :
                    GetSingleton<AtbOperationIrCfg>().GetOperationIr(
                        "RmsNormOperationNORMAndQUANTUNDEFINEDWithoutRSTD");
        } else if (param_.normParam.quantType == infer::QUANT_INT8) {
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
                operationIr_ =
                    GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationNORMAndQUANTParamInTensor");
            } else if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC) {
                operationIr_ =
                    GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationNORMAndSymmetricDynamicQuant");
            } else if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
                operationIr_ =
                    GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationNORMAndAsymmetricDynamicQuant");
            }
        } else {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "param_.normParam.quantType is invalid, type:" << param_.normParam.quantType;
        }
    } else if ((param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM)) {
        if (param_.preNormParam.quantType == infer::QUANT_INT8) {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPRENORMAndQUANT");
        } else if (param_.preNormParam.quantType == infer::QUANT_UNQUANT) {
            operationIr_ = param_.preNormParam.hasBias ?
                               GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPRENORM") :
                               GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPRENORMNoBias");
        } else {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "param_.normParam.quantType is invalid, type:" << param_.preNormParam.quantType;
        }
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_INT8) {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPOSTNORMAndQUANT");
        } else if (param_.postNormParam.quantType == infer::QUANT_UNQUANT) {
            operationIr_ = param_.postNormParam.hasBias ?
                               GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPOSTNORM") :
                               GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormOperationPOSTNORMNoBias");
        } else {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "param_.normParam.quantType is invalid, type:" << param_.postNormParam.quantType;
        }
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "param_.layerType is invalid, type:" << param_.layerType;
    }
}

RmsNormOperation::~RmsNormOperation() {}

uint32_t RmsNormOperation::GetInputNum() const
{
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_UNQUANT) {
            return IN_TENSOR_COUNT_TWO;
        }
        if (param_.normParam.quantType == infer::QUANT_INT8) {
            return param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED ? IN_TENSOR_COUNT_FIVE :
                                                                                         IN_TENSOR_COUNT_THREE;
        }
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
               param_.preNormParam.quantType == infer::QUANT_INT8) {
        return IN_TENSOR_COUNT_SIX;
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM &&
               param_.postNormParam.quantType == infer::QUANT_INT8) {
        return IN_TENSOR_COUNT_FIVE;
    } else if ((param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM &&
                param_.postNormParam.quantType == infer::QUANT_UNQUANT) ||
               (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
                param_.preNormParam.quantType == infer::QUANT_UNQUANT)) {
        return param_.preNormParam.hasBias || param_.postNormParam.hasBias ? IN_TENSOR_COUNT_FOUR :
                                                                             IN_TENSOR_COUNT_THREE;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "RmsNormType Undefined";
    }
    return 0;
}

uint32_t RmsNormOperation::GetOutputNum() const
{
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM &&
        param_.normParam.quantType == infer::QUANT_UNQUANT) {
        return param_.normParam.rstd ? OUT_TENSOR_COUNT_TWO : 1;
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM &&
               param_.normParam.quantType == infer::QUANT_INT8) {
        if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
            return 1;
        }
        if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC) {
            return OUT_TENSOR_COUNT_TWO;
        }
        if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
            return OUT_TENSOR_COUNT_THREE;
        }
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
               (param_.preNormParam.quantType == infer::QUANT_INT8 ||
                param_.preNormParam.quantType == infer::QUANT_UNQUANT)) {
        return OUT_TENSOR_COUNT_TWO;
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_UNQUANT) {
            return 1;
        }
        if (param_.postNormParam.quantType == infer::QUANT_INT8) {
            return OUT_TENSOR_COUNT_TWO;
        }
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "RmsNormType Undefined";
    return 0;
}

Status RmsNormOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM && param_.normParam.quantType == infer::QUANT_INT8) {
        outTensorDescs.at(0).dtype = ACL_INT8;
        if (param_.normParam.dynamicQuantType != infer::DYNAMIC_QUANT_UNDEFINED) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
            outTensorDescs.at(1).shape.dimNum--;
            outTensorDescs.at(1).dtype = ACL_FLOAT;
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
                outTensorDescs.at(2) = outTensorDescs.at(1); // 2: offset
            }
        }
    }
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM && param_.normParam.rstd) {
        outTensorDescs.at(OUT_TENSOR_RSTD) = inTensorDescs.at(IN_TENSOR_X);
        outTensorDescs.at(OUT_TENSOR_RSTD).dtype = ACL_FLOAT;
        size_t xDimNum = inTensorDescs.at(IN_TENSOR_X).shape.dimNum;
        size_t gammaDimNum = inTensorDescs.at(IN_TENSOR_GAMMA).shape.dimNum;
        for (size_t i = 0; i < gammaDimNum; i++) {
            outTensorDescs[OUT_TENSOR_RSTD].shape.dims[xDimNum - gammaDimNum + i] = 1;
        }
    }
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
        param_.preNormParam.quantType == infer::QUANT_INT8) {
        outTensorDescs.at(0).dtype = ACL_INT8;
        outTensorDescs.at(1) = inTensorDescs.at(0);
    }
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
        param_.preNormParam.quantType == infer::QUANT_UNQUANT) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
    }
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM &&
        param_.postNormParam.quantType == infer::QUANT_INT8) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(0).dtype = ACL_INT8;
    }
    return NO_ERROR;
}

Status RmsNormOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!DtypeCheck(inTensorDescs.at(IN_TENSOR_X).dtype)) {
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM) {
        if (param_.normParam.rstd) {
            if (!NormForwardInferShapeCheckImpl(inTensorDescs)) {
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else {
            if (!NormInferShapeCheckImpl(inTensorDescs)) {
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    } else if ((param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM &&
                param_.postNormParam.quantType == infer::QUANT_UNQUANT) ||
               (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM &&
                param_.preNormParam.quantType == infer::QUANT_UNQUANT)) {
        if (!AddRMSNormInferShapeCheckImpl(inTensorDescs)) {
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM &&
            param_.postNormParam.quantType == infer::QUANT_INT8) {
            if (!PostnormInferShapeCheckImpl(inTensorDescs)) {
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else if (!PrenormInferShapeCheckImpl(inTensorDescs)) {
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (!CheckLastDims(inTensorDescs)) {
        bool isPostQuant = (param_.postNormParam.quantType == infer::QUANT_INT8);
        return isPostQuant ? ERROR_INVALID_TENSOR_DIM : ERROR_INVALID_TENSOR_SIZE;
    }
    return ParamCheck();
}

Status RmsNormOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    aclDataType targetType = inTensorDescs[IN_TENSOR_X].dtype;
    if (param_.normParam.quantType == infer::QUANT_INT8 || param_.preNormParam.quantType == infer::QUANT_INT8 ||
        param_.postNormParam.quantType == infer::QUANT_INT8) {
        targetType = ACL_INT8;
    }
    Status st = CheckOutTensorSame(inTensorDescs[IN_TENSOR_X], outTensors[OUT_TENSOR_X].desc, targetType);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.preNormParam.quantType == infer::QUANT_INT8 || param_.postNormParam.quantType == infer::QUANT_INT8) {
        targetType = inTensorDescs[IN_TENSOR_X].dtype;
        st = CheckOutTensorSame(inTensorDescs[IN_TENSOR_X], outTensors[OUT_TENSOR_QAUNT].desc, targetType);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (param_.normParam.rstd) {
        st = CheckRstdOutTensor(inTensorDescs, outTensors[OUT_TENSOR_RSTD].desc);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return InferShapeCheckImpl(inTensorDescs);
}

Status RmsNormOperation::ParamCheck() const
{
    if (!CheckModelType() || !CheckPrecisionMode() || !CheckHasBias() || !CheckRstd()) {
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

bool RmsNormOperation::DtypeCheck(aclDataType inputDtype) const
{
    if (param_.normParam.rstd && inputDtype != ACL_FLOAT) {
        ATB_LOG(ERROR) << GetLogPrefix() << "rstd mode only supports float type";
        return false;
    }
    if (param_.normParam.precisionMode == infer::RmsNormParam::HIGH_PERFORMANCE_MODE && inputDtype != ACL_FLOAT16) {
        ATB_LOG(ERROR) << GetLogPrefix() << "HIGH_PERFORMANCE_MODE only supports float16 type";
        return false;
    }
    return true;
}

bool RmsNormOperation::CheckModelType() const
{
    if (param_.normParam.modelType != infer::RmsNormParam::LLAMA_MODEL) {
        if (param_.normParam.quantType != infer::QUANT_UNQUANT) {
            ATB_LOG(ERROR) << GetLogPrefix() << "modelType setting is not supported in quant";
            return false;
        }
        if (param_.normParam.precisionMode == infer::RmsNormParam::HIGH_PERFORMANCE_MODE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "modelType setting is not supported in HIGH_PERFORMANCE_MODE";
            return false;
        }
        if (param_.normParam.rstd) {
            ATB_LOG(ERROR) << GetLogPrefix() << "modelType setting is not supported in rstd";
            return false;
        }
    }
    return true;
}

bool RmsNormOperation::CheckPrecisionMode() const
{
    if (param_.normParam.precisionMode == infer::RmsNormParam::HIGH_PERFORMANCE_MODE) {
        if (param_.normParam.quantType != infer::QUANT_UNQUANT) {
            ATB_LOG(ERROR) << GetLogPrefix() << "HIGH_PERFORMANCE_MODE is not supported in quant";
            return false;
        }
        if (param_.normParam.rstd) {
            ATB_LOG(ERROR) << GetLogPrefix() << "HIGH_PERFORMANCE_MODE is not supported in rstd";
            return false;
        }
    }
    return true;
}

bool RmsNormOperation::CheckHasBias() const
{
    if (param_.preNormParam.hasBias) {
        if (param_.preNormParam.quantType != infer::QUANT_UNQUANT) {
            ATB_LOG(ERROR) << GetLogPrefix() << "hasBias option is not supported in quant";
            return false;
        }
    }
    if (param_.postNormParam.hasBias) {
        if (param_.postNormParam.quantType != infer::QUANT_UNQUANT) {
            ATB_LOG(ERROR) << GetLogPrefix() << "hasBias option is not supported in quant";
            return false;
        }
    }
    return true;
}

bool RmsNormOperation::CheckRstd() const
{
    if (param_.normParam.rstd) {
        if (param_.normParam.quantType != infer::QUANT_UNQUANT) {
            ATB_LOG(ERROR) << GetLogPrefix() << "rstd option is not supported in quant";
            return false;
        }
        if (!is910b_) {
            ATB_LOG(ERROR) << GetLogPrefix() << "only Atlas 800I A2 inference product devices support rstd";
            return false;
        }
    }
    return true;
}

std::shared_ptr<Runner> RmsNormOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RmsNormOpsRunner>(param_);
}

bool RmsNormOperation::NormInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc xTensorDesc = inTensorDescs.at(0);
    TensorDesc gammaTensorDesc = inTensorDescs.at(1);
    uint64_t xDimNum = xTensorDesc.shape.dimNum;
    uint64_t gammaDimNum = gammaTensorDesc.shape.dimNum;
    if (xTensorDesc.shape.dims[xDimNum - 1] != gammaTensorDesc.shape.dims[gammaDimNum - 1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 lastDim == inTensor1 lastDim";
        return false;
    }
    for (uint64_t i = 0; i < gammaTensorDesc.shape.dimNum - 1; i++) {
        if (gammaTensorDesc.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor1 dimsExceptLastDim == 1";
            return false;
        }
    }
    if (param_.normParam.quantType == infer::QUANT_INT8 &&
        param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
        TensorDesc betaTensorDesc = inTensorDescs.at(2);
        if (TensorDescsDiff(gammaTensorDesc, betaTensorDesc)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dims error, inTensor1 dims == inTensor2 dims";
            return false;
        }
        // 兼容intensor传入quant参数
        const TensorDesc &scaleTensorDesc = inTensorDescs.at(IN_TENSOR_SCALE);
        const TensorDesc &offsetTensorDesc = inTensorDescs.at(IN_TENSOR_OFFSET);
        if (!QuantTensorsCheck(scaleTensorDesc, offsetTensorDesc)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "quant tensor shape error";
            return false;
        }
    }
    return true;
}

bool RmsNormOperation::NormForwardInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    auto xDim = inTensorDescs.at(IN_TENSOR_X).shape.dims;
    auto gammaDim = inTensorDescs.at(IN_TENSOR_GAMMA).shape.dims;
    int32_t xDimNum = inTensorDescs.at(IN_TENSOR_X).shape.dimNum;
    int32_t gammaDimNum = inTensorDescs.at(IN_TENSOR_GAMMA).shape.dimNum;
    int32_t startDim = xDimNum - gammaDimNum;
    if (startDim < 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma's dimnum invalid";
        return false;
    }
    for (int32_t i = startDim; i < xDimNum; i++) {
        if (xDim[i] != gammaDim[i - startDim]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "gamma shape invalid";
            return false;
        }
    }
    return true;
}

bool RmsNormOperation::PrenormInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc xTensorDesc = inTensorDescs.at(0);
    TensorDesc resInTensorDesc = inTensorDescs.at(1);
    TensorDesc gammaTensorDesc = inTensorDescs.at(2);
    TensorDesc betaTensorDesc = inTensorDescs.at(3);
    if (TensorDescsDiff(xTensorDesc, resInTensorDesc) || TensorDescsDiff(gammaTensorDesc, betaTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 == inTensor1, inTensor2 == inTensor3";
        return false;
    }
    if (xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1] !=
        gammaTensorDesc.shape.dims[gammaTensorDesc.shape.dimNum - 1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 lastDim == inTensor2 lastDim";
        return false;
    }
    for (uint64_t i = 0; i < gammaTensorDesc.shape.dimNum - 1; i++) {
        if (gammaTensorDesc.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor2 dimsExceptLastDim == 1";
            return false;
        }
    }
    const TensorDesc &scaleTensorDesc = inTensorDescs.at(IN_TENSOR_SCALE_PRENORM);
    const TensorDesc &offsetTensorDesc = inTensorDescs.at(IN_TENSOR_OFFSET_PRENORM);
    if (!QuantTensorsCheck(scaleTensorDesc, offsetTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "quant tensor shape error";
        return false;
    }
    return true;
}

bool RmsNormOperation::PostnormInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc xTensorDesc = inTensorDescs.at(0);
    TensorDesc resInTensorDesc = inTensorDescs.at(1);
    TensorDesc gammaTensorDesc = inTensorDescs.at(2);
    if (TensorDescsDiff(xTensorDesc, resInTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 == inTensor1, inTensor2 == inTensor3";
        return false;
    }
    if (xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1] !=
        gammaTensorDesc.shape.dims[gammaTensorDesc.shape.dimNum - 1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 lastDim == inTensor2 lastDim";
        return false;
    }
    if (gammaTensorDesc.shape.dimNum != GAMMA_DIM_TWO &&
        (gammaTensorDesc.shape.dimNum != GAMMA_DIM_ONE && param_.postNormParam.quantType == infer::QUANT_INT8)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, gammainTensor dimsExceptDimNum == 1 or 2";
        return false;
    }
    for (uint64_t i = 0; i < gammaTensorDesc.shape.dimNum - 1; i++) {
        if (gammaTensorDesc.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor2 dimsExceptLastDim == 1";
            return false;
        }
    }
    const TensorDesc &scaleTensorDesc = inTensorDescs.at(IN_TENSOR_SCALE);
    const TensorDesc &offsetTensorDesc = inTensorDescs.at(IN_TENSOR_OFFSET);
    if (!QuantTensorsCheck(scaleTensorDesc, offsetTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "quant tensor shape error";
        return false;
    }
    return true;
}

bool RmsNormOperation::AddRMSNormInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t idx = 0;
    bool hasBias = param_.preNormParam.hasBias || param_.postNormParam.hasBias;
    TensorDesc xTensorDesc = inTensorDescs.at(idx++);
    TensorDesc biasTensorDesc;
    if (hasBias) {
        biasTensorDesc = inTensorDescs.at(idx++);
    }
    TensorDesc resInTensorDesc = inTensorDescs.at(idx++);
    TensorDesc gammaTensorDesc = inTensorDescs.at(idx++);
    if (TensorDescsDiff(xTensorDesc, resInTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 == inTensor1, inTensor2 == inTensor3";
        return false;
    }
    if (!GammaBetaTensorCheck(xTensorDesc, gammaTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gammaTensorDesc error";
        return false;
    }
    if (hasBias && !GammaBetaTensorCheck(xTensorDesc, biasTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "biasTensorDesc error";
        return false;
    }
    return true;
}

bool TensorDescsDiff(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2)
{
    if (tensorDesc1.dtype != tensorDesc2.dtype || tensorDesc1.format != tensorDesc2.format ||
        tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        return true;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            return true;
        }
    }
    return false;
}

Status RmsNormOperation::CheckOutTensorSame(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2,
                                            const aclDataType &targetType) const
{
    if (targetType != tensorDesc2.dtype) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dtype is invalid";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (tensorDesc1.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor format is invalid";
        return ERROR_INVALID_TENSOR_FORMAT;
    }
    if (tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dimNum is invalid";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dims are invalid";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status RmsNormOperation::CheckRstdOutTensor(const SVector<TensorDesc> &inTensorDescs,
                                            const TensorDesc &rsdtTensorDesc) const
{
    TensorDesc xTensorDesc = inTensorDescs[IN_TENSOR_X];
    TensorDesc gammaTensorDesc = inTensorDescs[IN_TENSOR_GAMMA];
    uint64_t xDimNum = xTensorDesc.shape.dimNum;
    uint64_t gammaDimNum = gammaTensorDesc.shape.dimNum;
    for (uint64_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - gammaDimNum && xTensorDesc.shape.dims[i] != rsdtTensorDesc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor rstd dim does not match the shape of x inTensor.";
            return ERROR_INVALID_TENSOR_DIM;
        } else if (i >= xDimNum - gammaDimNum && rsdtTensorDesc.shape.dims[i] != TENSOR_DIM_ONE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "size of dim which larger than xDimNum - gammaNum should be 1.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

bool RmsNormOperation::GammaBetaTensorCheck(const TensorDesc &xTensorDesc, const TensorDesc &tensorDesc2) const
{
    int embedDim = xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1];
    if (xTensorDesc.dtype != tensorDesc2.dtype || xTensorDesc.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma and bias should have the same dtype and format as x";
        return false;
    }
    if (tensorDesc2.shape.dims[tensorDesc2.shape.dimNum - 1] != embedDim) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor last dim is not equal to embed_dim";
        return false;
    }
    for (uint64_t i = 0; i < tensorDesc2.shape.dimNum - 1; i++) {
        if (tensorDesc2.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor non-last dim is not equal to 1";
            return false;
        }
    }
    return true;
}

// 兼容intensor传入quant参数
bool RmsNormOperation::QuantTensorsCheck(const TensorDesc &scaleTensorDesc, const TensorDesc &offsetTensorDesc) const
{
    if (scaleTensorDesc.shape.dimNum != TENSOR_DIM_ONE || offsetTensorDesc.shape.dimNum != TENSOR_DIM_ONE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "scaleTensor and offsetTensor dim number should be one";
        return false;
    }
    if (scaleTensorDesc.shape.dims[0] != DIM_SHAPE_ONE || offsetTensorDesc.shape.dims[0] != DIM_SHAPE_ONE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "scaleTensor and offsetTensor dim size should be one";
        return false;
    }
    return true;
}

bool RmsNormOperation::CheckLastDims(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t inTensorCount = GetInputNum();
    if (param_.normParam.quantType == infer::QUANT_INT8 || param_.preNormParam.quantType == infer::QUANT_INT8 ||
        param_.postNormParam.quantType == infer::QUANT_INT8) {
        inTensorCount -= QUANT_TENSOR_COUNT;
    }
    uint64_t dimNum = inTensorDescs.at(IN_TENSOR_X).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(IN_TENSOR_X).shape.dims[dimNum - 1];
    uint32_t dtypeSize = inTensorDescs.at(IN_TENSOR_X).dtype == ACL_FLOAT ? sizeof(uint32_t) : sizeof(uint16_t);
    if (param_.postNormParam.quantType == infer::QUANT_INT8) {
        if ((static_cast<uint64_t>(lastDim) % static_cast<uint64_t>(ALIGNMENT) != 0)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is not support, should align 32";
            return false;
        }
    }
    if ((static_cast<uint64_t>(lastDim) % static_cast<uint64_t>(ALIGNMENT / dtypeSize) != 0)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is not support, should align 32";
        return false;
    }
    if (param_.normParam.quantType == infer::QUANT_INT8) {
        if ((inTensorDescs.at(IN_TENSOR_X).shape.dims[dimNum - 1]) > DYNAMIC_QUANT_MAX_TENSOR_LENGTH &&
            param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Length of the last dim is invalid, it should be no larger than "
                           << DYNAMIC_QUANT_MAX_TENSOR_LENGTH << ".";
            return false;
        }
    }
    for (uint32_t i = 1; i < inTensorCount; i++) {
        dimNum = inTensorDescs.at(i).shape.dimNum;
        if ((inTensorDescs.at(i).shape.dims[dimNum - 1]) != lastDim) {
            ATB_LOG(ERROR) << GetLogPrefix() << "last dims of inTensors does not match";
            return false;
        }
    }
    return true;
}

nlohmann::json RmsNormOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
