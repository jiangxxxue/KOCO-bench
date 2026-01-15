/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_operation.h"
#include <cmath>
#include <sstream>
#include "layer_norm_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT_THREE = 3;
static const uint32_t IN_TENSOR_COUNT_FOUR = 4;
static const uint32_t IN_TENSOR_COUNT_FIVE = 5;
static const uint32_t IN_TENSOR_COUNT_SIX = 6;
static const uint32_t OUT_TENSOR_COUNT_TWO = 2;
static const uint32_t OUT_TENSOR_COUNT_THREE = 3;
static const uint32_t IN_TENSOR_SCALE_NORMNORM = 3;
static const uint32_t IN_TENSOR_OFFSET_NORMNORM = 4;
static const uint32_t IN_TENSOR_SCALE_POSTNORM = 4;
static const uint32_t IN_TENSOR_OFFSET_POSTNORM = 5;
static const uint32_t IN_TENSOR_GAMMA_PRE_POSTNORM = 2;
static const uint32_t X_TENSOR = 0;
static const uint32_t RESIDUAL_IN_TENSOR = 1;
static const uint32_t GAMMA_IN_TENSOR = 2;
static const uint32_t BETA_IN_TENSOR = 3;
static const uint32_t GAMMA_IN_TENSOR_NORM = 1;
static const uint32_t BETA_IN_TENSOR_NORM = 2;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_2 = 2;
static const uint32_t SIZE_1 = 1;
static const uint32_t ALIGNMENT = 32;
static const uint32_t QUANT_TENSOR_COUNT = 2;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.
static const uint32_t DYNAMIC_QUANT_MAX_TENSOR_LENGTH = 12288;

Status NormParamCheck(infer::LayerNormParam::NormParam normParam)
{
    if (normParam.quantType != infer::QUANT_UNQUANT && normParam.quantType != infer::QUANT_INT8) {
        ATB_LOG(ERROR) << "quantType is not supported, only support QUANT_UNQUANT/QUANT_INT8";
        return ERROR_INVALID_PARAM;
    }
    if (std::fabs(normParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
        ATB_LOG(ERROR) << "dynamicQuantType not support DYNAMIC_QUANT_ASYMMETRIC";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status PreNormParamCheck(infer::LayerNormParam::PreNormParam preNormParam)
{
    if (preNormParam.quantType != infer::QUANT_UNQUANT) {
        ATB_LOG(ERROR) << "prelayernorm does not support quant";
        return ERROR_INVALID_PARAM;
    }
    if (std::fabs(preNormParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (preNormParam.opMode != 0) {
        ATB_LOG(ERROR) << "Invalid opMode, only supports 0.";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status PostNormParamCheck(infer::LayerNormParam::PostNormParam postNormParam)
{
    if (postNormParam.quantType != infer::QUANT_UNQUANT && postNormParam.quantType != infer::QUANT_INT8) {
        ATB_LOG(ERROR) << "quantType is not supported, only support QUANT_UNQUANT/QUANT_INT8";
        return ERROR_INVALID_PARAM;
    }
    if (std::fabs(postNormParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (postNormParam.opMode != 0) {
        ATB_LOG(ERROR) << "Invalid opMode, only supports 0.";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

template <> Status CreateOperation(const infer::LayerNormParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    OP_PARAM_RSV_CHECK(opParam.normParam);
    OP_PARAM_RSV_CHECK(opParam.preNormParam);
    OP_PARAM_RSV_CHECK(opParam.postNormParam);
    Status paramCheckRet = NO_ERROR;
    if (opParam.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        paramCheckRet = NormParamCheck(opParam.normParam);
        if (paramCheckRet != NO_ERROR) {
            return paramCheckRet;
        }
    } else if (opParam.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        paramCheckRet = PreNormParamCheck(opParam.preNormParam);
        if (paramCheckRet != NO_ERROR) {
            return paramCheckRet;
        }
    } else if (opParam.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        paramCheckRet = PostNormParamCheck(opParam.postNormParam);
        if (paramCheckRet != NO_ERROR) {
            return paramCheckRet;
        }
    } else {
        ATB_LOG(ERROR) << "layerType is not supported, only support NORM/PRENORM/POSTNORM";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) LayerNormOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LayerNormOperation::LayerNormOperation(const infer::LayerNormParam &param)
    : OperationBase("LayerNormOperation"), param_(param)
{
    std::stringstream opIrKeySs;
    opIrKeySs << "LayerNormOperation";
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        opIrKeySs << "Norm";
        if (param_.normParam.quantType == infer::QUANT_INT8) {
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
                opIrKeySs << "Quant";
            } else if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC) {
                opIrKeySs << "SymmetricDynamicQuant";
            } else if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
                opIrKeySs << "AsymmetricDynamicQuant";
            }
        }
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        opIrKeySs << "Prenorm";
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        opIrKeySs << "Postnorm";
        if (param_.postNormParam.quantType == infer::QUANT_INT8) {
            opIrKeySs << "Quant";
        }
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeySs.str());
}

LayerNormOperation::~LayerNormOperation() {}

uint32_t LayerNormOperation::GetInputNum() const
{
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_UNQUANT) {
            return IN_TENSOR_COUNT_THREE;
        } else if (param_.normParam.quantType == infer::QUANT_INT8) {
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
                return IN_TENSOR_COUNT_FIVE;
            } else {
                return IN_TENSOR_COUNT_THREE;
            }
        }
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        return IN_TENSOR_COUNT_FOUR;
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        return param_.postNormParam.quantType == infer::QUANT_INT8 ? IN_TENSOR_COUNT_SIX : IN_TENSOR_COUNT_FOUR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LayerNormType Undefined";
    return 0;
}

uint32_t LayerNormOperation::GetOutputNum() const
{
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_INT8) {
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC) {
                return OUT_TENSOR_COUNT_TWO;
            } else if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
                return OUT_TENSOR_COUNT_THREE;
            }
        }
        return 1;
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        return OUT_TENSOR_COUNT_TWO;
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_UNQUANT) {
            return 1;
        } else {
            return OUT_TENSOR_COUNT_TWO;
        }
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LayerNormType Undefined";
    return 0;
}

Status LayerNormOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM && param_.normParam.quantType == infer::QUANT_INT8) {
        outTensorDescs.at(0).dtype = ACL_INT8;
        if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
            return NO_ERROR;
        }
        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dimNum--;
        outTensorDescs.at(1).dtype = ACL_FLOAT;
        if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
            outTensorDescs.at(2) = outTensorDescs.at(1); // 2: offset
        }
    }
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
    }
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_INT8) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
            outTensorDescs.at(1).dtype = ACL_INT8;
        }
    }
    return NO_ERROR;
}

Status LayerNormOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    const TensorDesc &gammaTensorDesc = param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM ?
                                            inTensorDescs.at(1) :
                                            inTensorDescs.at(IN_TENSOR_GAMMA_PRE_POSTNORM);
    Status checkStatus = ParamCheck(xTensorDesc, gammaTensorDesc);
    if (checkStatus != NO_ERROR) {
        return checkStatus;
    }
    checkStatus = InTensorsDimCheck(inTensorDescs);
    if (checkStatus != NO_ERROR) {
        return checkStatus;
    }
    checkStatus = LastDimCheck(inTensorDescs);
    if (checkStatus != NO_ERROR) {
        return checkStatus;
    }
    if ((param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM &&
         param_.normParam.quantType == infer::QUANT_INT8 &&
         param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) ||
        (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM &&
         param_.postNormParam.quantType == infer::QUANT_INT8)) {
        if (!QuantTensorCheck(inTensorDescs)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "quant tensor shape error";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    return NO_ERROR;
}

Status LayerNormOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    const TensorDesc &xTensorDesc = inTensors.at(0).desc;
    const TensorDesc &gammaTensorDesc = param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM ?
                                            inTensors.at(1).desc :
                                            inTensors.at(IN_TENSOR_GAMMA_PRE_POSTNORM).desc;
    Status paramCheckStatus = ParamCheck(xTensorDesc, gammaTensorDesc);
    if (paramCheckStatus != NO_ERROR) {
        return paramCheckStatus;
    }
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> outTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);
    Status lastDimCheckStatus = LastDimCheck(inTensorDescs);
    if (lastDimCheckStatus != NO_ERROR) {
        return lastDimCheckStatus;
    }
    uint32_t out_num = GetOutputNum();
    SVector<TensorDesc> targetOutTensorDescs = {};
    targetOutTensorDescs.reserve(out_num);
    targetOutTensorDescs.resize(out_num);
    InferShapeImpl(inTensorDescs, targetOutTensorDescs);
    for (size_t i = 0; i < out_num; ++i) {
        if (!TensorUtil::TensorDescEqual(outTensorDescs.at(i), targetOutTensorDescs.at(i))) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dims of outTensors does not match";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status LayerNormOperation::ParamCheck(const TensorDesc &xTensorDesc, const TensorDesc &gammaTensorDesc) const
{
    int32_t beginNormAxis = param_.normParam.beginNormAxis;
    uint64_t xTensorDimNum = xTensorDesc.shape.dimNum;
    uint64_t gammaTensorDimNum = gammaTensorDesc.shape.dimNum;
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM &&
        param_.normParam.quantType == infer::QUANT_UNQUANT) {
        if (gammaTensorDimNum > xTensorDimNum) {
            ATB_LOG(ERROR) << GetLogPrefix() << "gammaTensor dimNum should be smaller or equal to xTensor.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if ((beginNormAxis >= 0 && static_cast<uint64_t>(beginNormAxis) + gammaTensorDimNum != xTensorDimNum) ||
            (beginNormAxis < 0 && static_cast<uint64_t>(-beginNormAxis) != gammaTensorDimNum)) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "Invalid param: beginNormAxis,"
                              "beginNormAxis should not larger than xTensorDimNum or beginNormAxis can be negative,"
                              "but beginNormAxis+xTensorDimNum cannot be negative.";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> LayerNormOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<LayerNormOpsRunner>(param_);
}

nlohmann::json LayerNormOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

bool LayerNormOperation::QuantTensorCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t scaleIndex = 0;
    uint32_t offsetIndex = 0;
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        scaleIndex = IN_TENSOR_SCALE_NORMNORM;
        offsetIndex = IN_TENSOR_OFFSET_NORMNORM;
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        scaleIndex = IN_TENSOR_SCALE_POSTNORM;
        offsetIndex = IN_TENSOR_OFFSET_POSTNORM;
    }
    const TensorDesc scaleInTensorDesc = inTensorDescs.at(scaleIndex);
    const TensorDesc offsetInTensorDesc = inTensorDescs.at(offsetIndex);
    if (scaleInTensorDesc.shape.dimNum != 1 || offsetInTensorDesc.shape.dimNum != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim numbers of quant param intensors should be one";
        return false;
    }
    if (scaleInTensorDesc.shape.dims[0] != 1 || offsetInTensorDesc.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim sizes of quant param intensors should be one";
        return false;
    }
    return true;
}

Status LayerNormOperation::LastDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t dimNum = inTensorDescs.at(0).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(0).shape.dims[dimNum - 1];
    // only layernorm no quant has no restriction on alignment
    if (!(param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM &&
          param_.normParam.quantType == infer::QUANT_UNQUANT)) {
        if ((static_cast<uint64_t>(lastDim) % static_cast<uint64_t>(ALIGNMENT / sizeof(uint16_t))) != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is not support, should align 32";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    uint32_t inTensorCount = GetInputNum();
    if ((param_.normParam.quantType == infer::QUANT_INT8 &&
         param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) ||
        param_.postNormParam.quantType == infer::QUANT_INT8) {
        inTensorCount -= QUANT_TENSOR_COUNT;
    }
    for (uint32_t i = 1; i < inTensorCount; i++) {
        dimNum = inTensorDescs.at(i).shape.dimNum;
        if ((inTensorDescs.at(i).shape.dims[dimNum - 1]) != lastDim) {
            ATB_LOG(ERROR) << GetLogPrefix() << "last dims of inTensors does not match";
            return ERROR_INVALID_TENSOR_SIZE;
        }
    }
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM &&
        param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC &&
        lastDim > DYNAMIC_QUANT_MAX_TENSOR_LENGTH) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Length of the last dim is invalid, it should be no larger than "
                       << DYNAMIC_QUANT_MAX_TENSOR_LENGTH << ".";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    return NO_ERROR;
}

Status LayerNormOperation::InTensorsDimCheck(const SVector<TensorDesc> inTensorDescs) const
{
    Status result = NO_ERROR;
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM &&
        param_.normParam.quantType == infer::QUANT_INT8 &&
        param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC &&
        inTensorDescs.at(0).shape.dimNum < 2) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim numbers of inTensor[0] should be greater than one";
            return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM ||
        param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        result = TensorCheck::TensorDescsEqual(inTensorDescs.at(X_TENSOR), inTensorDescs.at(RESIDUAL_IN_TENSOR));
        if (result != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Residual tensor should be the same as x tensor.";
            return result;
        }
    }
    return result;
}
} // namespace atb