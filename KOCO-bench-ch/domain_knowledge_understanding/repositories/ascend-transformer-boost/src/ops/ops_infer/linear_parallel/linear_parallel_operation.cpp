/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_parallel_operation.h"
#include <sstream>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "linear_parallel_graph_runner.h"
#include "linear_parallel_aclnn_runner.h"
#include "linear_parallel_lcoc_runner.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM_WITH_RESIDUAL = 3;
static const uint32_t IN_TENSOR_NUM_WITHOUT_RESIDUAL = 2;
static const uint32_t IN_TENSOR_NUM_WITH_MOE = 2;
static const uint32_t EXTRA_IN_TENSOR_NUM_WITH_QUANT = 2;
static const uint32_t MOE_IN_TENSOR_NUM_REMOVE_OFFSET = 1;
static const uint32_t EXTRA_IN_TENSOR_NUM_WITH_PER_TOKEN_QUANT = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM_WITH_MID = 2;
static const uint32_t LOCAL_EXPERT_NUMS_MIN = 1;
static const uint32_t LOCAL_EXPERT_NUMS_MAX = 16;
static const uint32_t IN_TENSOR_DIM_NUM = 2;
static const uint32_t RESIDUAL_TENSOR_INDEX_3 = 3;
static const uint32_t RESIDUAL_TENSOR_INDEX_4 = 4;
static const uint32_t MAX_OUTPUT_SIZE = 204800;
static const uint32_t MAX_K = 24000;

static bool AllToAllvcAllGatherGmmOutTensorCheck(const SVector<TensorDesc> &inTensorDescs,
                                                 const TensorDesc &outTensorDesc, const std::string &logPrefix)
{
    uint64_t inTensorXDimNum = inTensorDescs.at(0).shape.dimNum;
    if (inTensorXDimNum != IN_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << logPrefix << "invalid intensor dimNum";
        return false;
    }
    if (inTensorXDimNum != outTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR) << logPrefix << "outTensor dimNum [" << outTensorDesc.shape.dimNum << "] and inTensor0 dimNum ["
                       << inTensorXDimNum << "] should be equal";
        return false;
    }
    return true;
}

bool CheckType(const infer::LinearParallelParam &opParam, Status &isOK)
{
    if (opParam.type <= atb::infer::LinearParallelParam::ParallelType::UNDEFINED ||
        opParam.type >= atb::infer::LinearParallelParam::ParallelType::MAX) {
        ATB_LOG(ERROR) << "LinearParallel type:" << opParam.type << " is invalid ParallelType";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.type != atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR && opParam.keepIntermediate) {
        ATB_LOG(ERROR) << "LinearParallel backend lcoc only ALL_GATHER_LINEAR support keepIntermediate is true";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN &&
        opParam.type != atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM &&
        opParam.type != atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC &&
        opParam.type != atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE) {
        ATB_LOG(ERROR) << "When LinearParallel backend is lcoc, "
                       << "only ALLTOALLVC_ALL_GATHER_GMM, GMM_REDUCE_SCATTER_ALLTOALLVC and LINEAR_ALL_REDUCE"
                       << " support quantType[QUANT_TYPE_PER_TOKEN]";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
        opParam.type == atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC) {
        if (opParam.moeInfo.epSize * opParam.moeInfo.tpSize != opParam.rankSize) {
            ATB_LOG(ERROR) << "ep * tp should equal to rankSize";
            isOK = ERROR_INVALID_PARAM;
            return true;
        }
        if (opParam.moeInfo.localExpertNums < static_cast<int16_t>(LOCAL_EXPERT_NUMS_MIN) ||
            opParam.moeInfo.localExpertNums > static_cast<int16_t>(LOCAL_EXPERT_NUMS_MAX)) {
            ATB_LOG(ERROR) << "localExpertNums should between 1 and 16";
            isOK = ERROR_INVALID_PARAM;
            return true;
        }
        if (opParam.quantType != atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED &&
            opParam.quantType != atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL &&
            opParam.quantType != atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN) {
            ATB_LOG(ERROR) << "LinearParallel type:" << opParam.type
                           << " support QuantType is QUANT_TYPE_UNDEFINED/QUANT_TYPE_PER_CHANNEL/QUANT_TYPE_PER_TOKEN";
            isOK = ERROR_INVALID_PARAM;
            return true;
        }
    }
    return false;
}

bool CheckTypeMc2(const infer::LinearParallelParam &opParam, Status &isOK)
{
    if (opParam.hasResidual) {
        ATB_LOG(ERROR) << "When LinearParallel backend is mc2, not support residual";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.commMode != atb::infer::CommMode::COMM_MULTI_THREAD) {
        ATB_LOG(ERROR) << "When LinearParallel backend is mc2, only support commMode[COMM_MULTI_THREAD]";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.quantType != atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNDEFINED ||
        opParam.quantType != atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT) {
        ATB_LOG(ERROR)
            << "When LinearParallel backend is mc2, only support quantType[QUANT_TYPE_UNDEFINED][QUANT_TYPE_UNQUANT]";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    if (opParam.type != infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER) {
        ATB_LOG(ERROR)
            << "When LinearParallel backend is mc2, only support type[LINEAR_REDUCE_SCATTER], LinearParallel type:"
            << opParam.type << " is invalid ParallelType";
        isOK = ERROR_INVALID_PARAM;
        return true;
    }
    return false;
}

template <> Status CreateOperation(const infer::LinearParallelParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (OperationUtil::DistributedInitCheck<infer::LinearParallelParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "LinearParallelOperation DistributedInitCheck failed.";
        return ERROR_INVALID_PARAM;
    }
    int rankSize = opParam.rankSize;
    if ((opParam.rankSize <= 0 || (rankSize & (rankSize - 1)) != 0) && opParam.backend == "lcoc") {
        ATB_LOG(ERROR) << "LinearParallel rankSize support power of 2 but got [" << opParam.rankSize << "]";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend != "hccl" && opParam.backend != "lccl" && opParam.backend != "lcoc" &&
        opParam.backend != "mc2") {
        ATB_LOG(ERROR) << "LinearParallel backend support hccl/lccl/lcoc/mc2 but get [" << opParam.backend << "]";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend == "lcoc") {
        Status isOk;
        if (CheckType(opParam, isOk))
            return isOk;
    }
    if (opParam.backend == "mc2") {
        Status isOk;
        if (CheckTypeMc2(opParam, isOk)) {
            return isOk;
        }
        isOk = LinearParallelAclnnRunner::LoadMethodMatmulReduceScatter();
        if (isOk != NO_ERROR) {
            return isOk;
        }
    }
    *operation = new (std::nothrow) LinearParallelOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LinearParallelOperation::LinearParallelOperation(const infer::LinearParallelParam &param)
    : OperationBase("LinearParallelOperation"), param_(param)
{
    commonCheckParam_ = param_;
    std::stringstream opIrKeySs;
    std::string withStr = param_.hasResidual ? "With" : "Without";
    opIrKeySs << "LinearParallelOperation" << withStr << "Residual";
    if (param_.backend == "lcoc" || param_.backend == "mc2") {
        opIrKeySs << (param_.backend == "lcoc" ? "Lcoc" : "Mc2");
        if (param_.keepIntermediate) {
            opIrKeySs << "KeepIntermediate";
        }
        if (param_.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
            param_.type == atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC) {
            opIrKeySs << "Moe";
        }
        if (commonCheckParam_.isQuant) {
            opIrKeySs << "Quant";
            if (param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN) {
                opIrKeySs << "PerToken";
            }
        }
    }
    std::string opIrKey = opIrKeySs.str();
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}

LinearParallelOperation::~LinearParallelOperation() {}

uint32_t LinearParallelOperation::GetInputNum() const
{
    bool isMoe = false;
    if (param_.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
        param_.type == atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC) {
        isMoe = true;
    }
    uint32_t inTensorNum = param_.hasResidual ? IN_TENSOR_NUM_WITH_RESIDUAL : IN_TENSOR_NUM_WITHOUT_RESIDUAL;
    if (isMoe) {
        inTensorNum += IN_TENSOR_NUM_WITH_MOE;
    }
    if (commonCheckParam_.isQuant) {
        if (isMoe) {
            inTensorNum -= MOE_IN_TENSOR_NUM_REMOVE_OFFSET;
        }
        inTensorNum += EXTRA_IN_TENSOR_NUM_WITH_QUANT;
        if (param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_TOKEN) {
            inTensorNum += EXTRA_IN_TENSOR_NUM_WITH_PER_TOKEN_QUANT;
        }
    }
    return inTensorNum;
}

uint32_t LinearParallelOperation::GetOutputNum() const
{
    return param_.keepIntermediate ? OUT_TENSOR_NUM_WITH_MID : OUT_TENSOR_NUM;
}

Status LinearParallelOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.backend != "lcoc" && param_.backend != "mc2") {
        return OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
    }
    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return InferShapeLinearReduceScatter(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return InferShapeAllGatherLinear(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return InferShapeAllGatherLinearReduceScatter(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
        case atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            return InferShapeAllToAllvcAllGatherGmm(inTensorDescs, outTensorDescs);
        default:
            ATB_LOG(ERROR) << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}


Status LinearParallelOperation::InferShapeLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                              SVector<TensorDesc> &outTensorDescs) const
{
    Status st = OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    if (outTensorDescs.at(0).shape.dims[0] == 1) {
        if (outTensorDescs.at(0).shape.dims[1] % param_.rankSize != 0) {
            return ERROR_INVALID_TENSOR_DIM;
        }
        outTensorDescs.at(0).shape.dims[1] /= param_.rankSize;
    } else {
        if (outTensorDescs.at(0).shape.dims[0] % param_.rankSize != 0) {
            return ERROR_INVALID_TENSOR_DIM;
        }
        outTensorDescs.at(0).shape.dims[0] /= param_.rankSize;
    }
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeAllGatherLinear(const SVector<TensorDesc> &inTensorDescs,
                                                          SVector<TensorDesc> &outTensorDescs) const
{
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    Status st = OperationUtil::MatmulInferShape(newInTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    if (outTensorDescs.size() == OUT_TENSOR_NUM_WITH_MID) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dims[0] *= param_.rankSize;
    }
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeAllGatherLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                                       SVector<TensorDesc> &outTensorDescs) const
{
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    Status st = OperationUtil::MatmulInferShape(newInTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    outTensorDescs.at(0).shape.dims[0] /= param_.twoDimTPInfo.rsDim;
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs,
                                                                 SVector<TensorDesc> &outTensorDescs) const
{
    Status st = OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
    if (st != NO_ERROR) {
        return st;
    }
    uint32_t maxOutputSize = inTensorDescs.at(inTensorDescs.size() - 1).shape.dims[0];
    if (maxOutputSize > MAX_OUTPUT_SIZE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "maxOutputSize is too large." << maxOutputSize;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    uint32_t k = inTensorDescs.at(0).shape.dims[1];
    if (k > MAX_K) {
        ATB_LOG(ERROR) << GetLogPrefix() << "K is too large. should be 1 to 24000, now is " << k;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(inTensorDescs.size() - 1).shape.dims[0];
    return NO_ERROR;
}

Status LinearParallelOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.backend != "lcoc" && param_.backend != "mc2") {
        return InferShapeCheckLinearAllReduce(inTensorDescs);
    }
    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return InferShapeCheckLinearAllReduce(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return InferShapeCheckLinearReduceScatter(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return InferShapeCheckAllGatherLinear(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return InferShapeCheckAllGatherLinearReduceScatter(inTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
        case atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            return InferShapeCheckAllToAllvcAllGatherGmm(inTensorDescs);
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}

Status LinearParallelOperation::CheckResidual(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.hasResidual) {
        int64_t n = OperationUtil::GetYTensorN(inTensorDescs.at(1), param_.transWeight);
        size_t residualTensorId = inTensorDescs.size();
        if (param_.type == atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM ||
            param_.type == atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC) {
            residualTensorId -= commonCheckParam_.isQuant ? RESIDUAL_TENSOR_INDEX_4 : RESIDUAL_TENSOR_INDEX_3;
        } else {
            residualTensorId -= 1;
        }
        const TensorDesc &residual = inTensorDescs.at(residualTensorId);
        int64_t needFirstDim = 1;
        if (!OperationUtil::LinearBiasDeqCheck(residual, GetLogPrefix(), n, needFirstDim, residualTensorId)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Check residual tensor failed.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}


Status LinearParallelOperation::InferShapeCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    bool isQuant = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT &&
                   param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
    if (isQuant && inTensorDescs.at(3).dtype == ACL_FLOAT && param_.outDataType == ACL_FLOAT16) {
        ATB_LOG(ERROR) << GetLogPrefix() << "when perChannelScale's type is float, "
                       << "outputDataType do not support float16_t";
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }

    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    bool isQuant = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT &&
                   param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
    if (isQuant && inTensorDescs.at(3).dtype == ACL_FLOAT && param_.outDataType == ACL_FLOAT16) {
        ATB_LOG(ERROR) << GetLogPrefix() << "when perChannelScale's type is float, "
                       << "outputDataType do not support float16_t";
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }

    if (param_.backend == "mc2") {
        if (inTensorDescs.at(0).shape.dimNum != IN_TENSOR_DIM_NUM) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dimNum should be equal to 2";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        if (inTensorDescs.at(1).shape.dimNum != IN_TENSOR_DIM_NUM) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be equal to 2";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
        int64_t xTensorK = OperationUtil::GetXTensorK(inTensorDescs.at(0));
        if (xTensorK < 256 || xTensorK >= 65535) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 k [" << xTensorK
                           << "] should be an integer between [256 ~ 65535)";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckAllGatherLinear(const SVector<TensorDesc> &inTensorDescs) const
{
    bool isQuant = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT &&
                   param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
    if (isQuant && inTensorDescs.at(3).dtype == ACL_FLOAT && param_.outDataType == ACL_FLOAT16) {
        ATB_LOG(ERROR) << GetLogPrefix() << "when perChannelScale's type is float, "
                       << "outputDataType do not support float16_t";
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }

    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    ATB_LOG(INFO) << GetLogPrefix() << "matmul input dim0: " << newInTensorDescs.at(0).shape.dims[0];
    if (!OperationUtil::MatmulInTensorDescsCheck(newInTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckAllGatherLinearReduceScatter(
    const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.twoDimTPInfo.rsDim * param_.twoDimTPInfo.agDim != param_.rankSize) {
        ATB_LOG(ERROR) << "agDim * rsDim should equal to rankSize";
        return ERROR_INVALID_PARAM;
    }
    if (param_.twoDimTPInfo.rsDim == 0) {
        ATB_LOG(ERROR) << "rsDim can't be 0";
        return ERROR_INVALID_PARAM;
    }
    SVector<TensorDesc> newInTensorDescs;
    newInTensorDescs = inTensorDescs;
    newInTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    ATB_LOG(INFO) << GetLogPrefix() << "matmul input dim0: " << newInTensorDescs.at(0).shape.dims[0];
    if (!OperationUtil::MatmulInTensorDescsCheck(newInTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t xTensorM = OperationUtil::GetXTensorM(inTensorDescs.at(0));
    if (xTensorM * param_.twoDimTPInfo.agDim % param_.twoDimTPInfo.rsDim != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 m [" << xTensorM << "] times agDim ["
                       << param_.twoDimTPInfo.agDim
                       << "] should be an integer multiple of rsDim :" << param_.twoDimTPInfo.rsDim;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::InferShapeCheckAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs) const
{
    bool isQuant = param_.quantType > infer::LinearParallelParam::QuantType::QUANT_TYPE_UNQUANT &&
                   param_.quantType < infer::LinearParallelParam::QuantType::QUANT_TYPE_MAX;
    if (isQuant && inTensorDescs.at(2).dtype == ACL_FLOAT && param_.outDataType == ACL_FLOAT16) {
        ATB_LOG(ERROR) << GetLogPrefix() << "when perChannelScale's type is float, "
                       << "outputDataType do not support float16_t";
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }

    int64_t globalTokensPerExpertMatrixM = OperationUtil::GetXTensorM(inTensorDescs.at(inTensorDescs.size() - 2));
    int64_t globalTokensPerExpertMatrixK = OperationUtil::GetXTensorK(inTensorDescs.at(inTensorDescs.size() - 2));
    int64_t expertNums = param_.moeInfo.epSize * param_.moeInfo.localExpertNums;
    if (globalTokensPerExpertMatrixM != param_.moeInfo.epSize || globalTokensPerExpertMatrixK != expertNums) {
        ATB_LOG(ERROR) << GetLogPrefix() << "globalTokensPerExpertMatrix m [" << globalTokensPerExpertMatrixM
                       << "] should equal to ep [" << param_.moeInfo.epSize << "] and globalTokensPerExpertMatrix k ["
                       << globalTokensPerExpertMatrixK << "] should equal to ep*localExpertNums [" << expertNums << "]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckResidual(inTensorDescs);
}

Status LinearParallelOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);

    switch (param_.type) {
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE:
        case atb::infer::LinearParallelParam::ParallelType::PURE_LINEAR:
            return SetupCheckLinearAllReduce(inTensorDescs, outTensorDescs.at(0));
        case atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER:
            return SetupCheckLinearReduceScatter(inTensorDescs, outTensorDescs.at(0));
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR:
            return SetupCheckAllGatherLinear(inTensorDescs, outTensorDescs);
        case atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR_REDUCE_SCATTER:
            return SetupCheckAllGatherLinearReduceScatter(inTensorDescs, outTensorDescs.at(0));
        case atb::infer::LinearParallelParam::ParallelType::ALLTOALLVC_ALL_GATHER_GMM:
        case atb::infer::LinearParallelParam::ParallelType::GMM_REDUCE_SCATTER_ALLTOALLVC:
            return SetupCheckAllToAllvcAllGatherGmm(inTensorDescs, outTensorDescs.at(0));
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "not support type:" << param_.type;
            return ERROR_INVALID_PARAM;
    }
}

Status LinearParallelOperation::SetupCheckLinearAllReduce(const SVector<TensorDesc> &inTensorDescs,
                                                          const TensorDesc &outTensorDesc) const
{
    if (inTensorDescs.at(0).dtype == ACL_INT8 &&
        param_.quantType == atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_GROUP) {
        ATB_LOG(ERROR) << GetLogPrefix() << "In the W8A8 scenario, per_group is not supported.";
        return ERROR_INVALID_PARAM;
    }
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearParallelOperation::SetupCheckLinearReduceScatter(const SVector<TensorDesc> &inTensorDescs,
                                                              TensorDesc &outTensorDesc) const
{
    if (InferShapeCheckLinearReduceScatter(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.outDataType != ACL_DT_UNDEFINED && outTensorDesc.dtype != param_.outDataType) {
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }
    if (inTensorDescs.at(0).shape.dims[0] == 1) {
        outTensorDesc.shape.dims[1] *= param_.rankSize;
    } else {
        outTensorDesc.shape.dims[0] *= param_.rankSize;
    }
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearParallelOperation::SetupCheckAllGatherLinear(SVector<TensorDesc> &inTensorDescs,
                                                          const SVector<TensorDesc> &outTensorDescs) const
{
    if (InferShapeCheckAllGatherLinear(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.outDataType != ACL_DT_UNDEFINED && outTensorDescs.at(0).dtype != param_.outDataType) {
        return ERROR_INVALID_TENSOR_INI_MATCH;
    }
    inTensorDescs.at(0).shape.dims[0] *= param_.rankSize;
    if (!OperationUtil::MatmulOutTensorCheck(outTensorDescs.at(0), inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensorDescs.size() == OUT_TENSOR_NUM_WITH_MID) {
        if (outTensorDescs.at(1).shape.dims[0] != inTensorDescs.at(0).shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor1 dim0 does not match (rankSize * inTensor0 dim0)";
        }
    }
    return NO_ERROR;
}

Status LinearParallelOperation::SetupCheckAllGatherLinearReduceScatter(SVector<TensorDesc> &inTensorDescs,
                                                                       TensorDesc &outTensorDesc) const
{
    if (InferShapeCheckAllGatherLinearReduceScatter(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    inTensorDescs.at(0).shape.dims[0] *= param_.twoDimTPInfo.agDim;
    outTensorDesc.shape.dims[0] *= param_.twoDimTPInfo.rsDim;
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearParallelOperation::SetupCheckAllToAllvcAllGatherGmm(const SVector<TensorDesc> &inTensorDescs,
                                                                 const TensorDesc &outTensorDesc) const
{
    if (InferShapeCheckAllToAllvcAllGatherGmm(inTensorDescs) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return AllToAllvcAllGatherGmmOutTensorCheck(inTensorDescs, outTensorDesc, GetLogPrefix()) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

nlohmann::json LinearParallelOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> LinearParallelOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    if (param_.backend == "hccl" || param_.backend == "lccl") {
        return std::make_shared<LinearParallelGraphRunner>(param_, *contextBase);
    } else if (param_.backend == "lcoc") {
        return std::make_shared<LinearParallelLcocRunner>(param_, context);
    } else if (param_.backend == "mc2") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<LinearParallelAclnnRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<LinearParallelAclnnRunner>(param_, param_.hcclComm);
        }
    }
    return std::shared_ptr<Runner>();
}
} // namespace atb
