/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ring_mla_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "ring_mla_ops_runner.h"

namespace {
static const uint32_t BASE_IN_TENSOR_NUM = 7;  // query1, query2, key1, key2, value, mask, seqLen
static const uint32_t RING_OPTIONAL_IN_TENSOR_NUM = 2; // prevOut, prevLse
static const uint32_t BASE_OUT_TENSOR_NUM = 2; // output, softmaxLse
// dimNum
static const uint32_t QKV_DIM_NUM = 3; // [sum(seqlen), headNum, headSize]
static const uint32_t LSE_DIM_NUM = 2; // [headNum, qNTokens]
// inTensor, outTensor index
static const uint32_t IN_QUERY_SPLIT1_INDEX = 0;
static const uint32_t IN_QUERY_SPLIT2_INDEX = IN_QUERY_SPLIT1_INDEX + 1;
static const uint32_t IN_KEY_SPLIT1_INDEX = IN_QUERY_SPLIT1_INDEX + 2;
static const uint32_t IN_KEY_SPLIT2_INDEX = IN_QUERY_SPLIT1_INDEX + 3;
static const uint32_t IN_VALUE_INDEX = IN_KEY_SPLIT2_INDEX + 1;
static const uint32_t IN_MASK_INDEX = IN_KEY_SPLIT2_INDEX + 2;
static const uint32_t IN_SEQLEN_INDEX = IN_KEY_SPLIT2_INDEX + 3;
static const uint32_t IN_PREV_OUT_INDEX = IN_KEY_SPLIT2_INDEX + 4;
static const uint32_t IN_PREV_LSE_INDEX = IN_KEY_SPLIT2_INDEX + 5;
static const uint32_t OUT_PREV_OUT_INDEX = 0;
static const uint32_t OUT_PREV_LSE_INDEX = 1;
// query, key, value dim index
static const uint32_t QKV_N_TOKENS_IDX = 0;
static const uint32_t QKV_HEAD_NUM_IDX = 1;
static const uint32_t QKV_HEAD_SIZE_IDX = 2;
static const uint32_t QK_SPLIT1_HEAD_SIZE = 128;
static const uint32_t QK_SPLIT2_HEAD_SIZE = 64;
// lse dim index
static const uint32_t LSE_N_TOKENS_IDX = 1;
static const uint32_t LSE_HEAD_NUM_IDX = 0;
// seqlen, mask index
static const uint32_t SEQLEN_BATCH_IDX = 0;
}; // namespace
namespace atb {
bool ParamCheck(const infer::RingMLAParam &opParam, ExternalError &extError)
{
    if (opParam.calcType != infer::RingMLAParam::CalcType::CALC_TYPE_DEFAULT &&
        opParam.calcType != infer::RingMLAParam::CalcType::CALC_TYPE_FISRT_RING) {
        extError.errorDesc = "Ring MLA expects calcType to be one of CALC_TYPE_DEFAULT, CALC_TYPE_FISRT_RING.";
        extError.errorData = OperationUtil::ConcatInfo("But got param.calcType = ", opParam.calcType);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.headNum <= 0) {
        extError.errorDesc = "Ring MLA expects headNum to be greater than zero!";
        extError.errorData = OperationUtil::ConcatInfo("But got param.headNum = ", opParam.headNum);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.kvHeadNum < 0) {
        extError.errorDesc = "Ring MLA expects kvHeadNum to be no less than zero!";
        extError.errorData = OperationUtil::ConcatInfo("But got param.kvHeadNum = ", opParam.kvHeadNum);
        return false;
    }
    if (opParam.kvHeadNum > 0 && opParam.headNum % opParam.kvHeadNum != 0) {
        extError.errorDesc = "Ring MLA expects headNum to be divisible by kvHeadNum.";
        extError.errorData = OperationUtil::ConcatInfo("But got param.headNum = ", opParam.headNum,
                                                       "param.kvHeadNum = ", opParam.kvHeadNum);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.headNum < opParam.kvHeadNum) {
        extError.errorDesc = "Ring MLA expects headNum >= kvHeadNum";
        extError.errorData = OperationUtil::ConcatInfo("But got param.headNum = ", opParam.headNum,
                                                       "param.kvHeadNum = ", opParam.kvHeadNum);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.maskType != infer::RingMLAParam::MaskType::NO_MASK &&
        opParam.maskType != infer::RingMLAParam::MaskType::MASK_TYPE_TRIU) {
        extError.errorDesc = "Ring MLA expects maskType as one of NO_MASK, MASK_TYPE_TRIU";
        extError.errorData = OperationUtil::ConcatInfo("But got param.maskType = ", opParam.maskType);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.inputLayout != infer::TYPE_BSND) {
        extError.errorDesc = "Ring MLA only supports inputLayout as TYPE_BSND";
        extError.errorData = OperationUtil::ConcatInfo("But got param.inputLayout = ", opParam.inputLayout);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    if (opParam.kernelType != infer::RingMLAParam::KernelType::KERNELTYPE_HIGH_PRECISION) {
        extError.errorDesc = "Ring MLA only supports kernelType as KERNELTYPE_HIGH_PRECISION";
        extError.errorData = OperationUtil::ConcatInfo("But got param.kernelType = ", opParam.kernelType);
        ATB_LOG(ERROR) << extError;
        return false;
    }
    return true;
}

template <> Status CreateOperation(const infer::RingMLAParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation RingMLAParam headNum: " << opParam.headNum << ", kvHeadNum: " << opParam.kvHeadNum
                  << ", qkScale: " << opParam.qkScale << ", calcType: " << opParam.calcType
                  << ", kernelType: " << opParam.kernelType << ", maskType: " << opParam.maskType
                  << ", inputLayout:" << opParam.inputLayout;
    ExternalError extError;
    extError.errorType = ERROR_INVALID_PARAM;
    extError.solutionDesc = "Please check the RingMLA op params.";
    if (!GetSingleton<Config>().Is910B()) {
        extError.errorDesc = "Ring MLA only support 800I A2/A3 inference product";
        ATB_LOG(ERROR) << extError;
        return ERROR_INVALID_PARAM;
    }
    if (!ParamCheck(opParam, extError)) {
        return extError.errorType;
    }
    *operation = new (std::nothrow) RingMLAOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}


RingMLAOperation::RingMLAOperation(const infer::RingMLAParam &opParam)
    : OperationBase("RingMLAOperation"), param_(opParam)
{
    std::string opKey = "RingMLAOperation";
    if (opParam.calcType == infer::RingMLAParam::CALC_TYPE_DEFAULT) { // CALC_TYPE_DEFAULT:  输入lse，o，输出 lse，o
        isInputSoftmaxLse_ = true;
        opKey += "InputSoftmaxLse";
    }
    ATB_LOG(INFO) << GetLogPrefix() << "Ring MLA created with key: " << opKey;
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opKey);
}


RingMLAOperation::~RingMLAOperation() {}

uint32_t RingMLAOperation::GetInputNum() const
{
    if (isInputSoftmaxLse_) {
        return BASE_IN_TENSOR_NUM + RING_OPTIONAL_IN_TENSOR_NUM;
    }
    return BASE_IN_TENSOR_NUM;
}

uint32_t RingMLAOperation::GetOutputNum() const
{
    return BASE_OUT_TENSOR_NUM;
}

bool RingMLAOperation::DimNumCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const
{
    // IR: query1, query2, key1, key2, value, mask, (prevLse), (prevOut)
    for (int i = 0; i < 5; ++i) {                              // 5: query1, query2, key1, key2, value
        if (inTensorDescs.at(i).shape.dimNum != QKV_DIM_NUM) { // QKV_DIM_NUM: [nTokens, headNum, headSize]
            extError.errorDesc = OperationUtil::ConcatInfo("DimNum of inTensors[", i, "] should be ", QKV_DIM_NUM);
            extError.errorData =
                OperationUtil::ConcatInfo(", but got inTensors[", i, "] dimNum: ", inTensorDescs.at(i).shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return false;
        }
    }
    if (param_.maskType != infer::RingMLAParam::MaskType::NO_MASK) {
        if (inTensorDescs.at(IN_MASK_INDEX).shape.dimNum != 2 && // 2: mask: [maxSeqLen, maxSeqLen]
            inTensorDescs.at(IN_MASK_INDEX).shape.dimNum != 3) { // 3: mask: [batch, maxSeqLen, maxSeqLen]
            extError.errorDesc = "dimNum of mask should be 2 or 3!";
            extError.errorData =
                OperationUtil::ConcatInfo(", but got mask dimNum: ", inTensorDescs.at(IN_MASK_INDEX).shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return false;
        }
    }

    if (inTensorDescs.at(IN_SEQLEN_INDEX).shape.dimNum != 1 && // 1: [batch]
        inTensorDescs.at(IN_SEQLEN_INDEX).shape.dimNum != 2) { // 2: [2, batch]
        extError.errorDesc = "dimNum of seqlen should be 1 or 2!";
        extError.errorData =
            OperationUtil::ConcatInfo(", but got seqlen dimNum: ", inTensorDescs.at(IN_SEQLEN_INDEX).shape.dimNum);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    if (isInputSoftmaxLse_) {
        if (inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dimNum != QKV_DIM_NUM) { // 3: [nTokens, headNum, headSize]
            extError.errorDesc = "dimNum of prevOut should be 3!";
            extError.errorData = OperationUtil::ConcatInfo("But got prevOut dimNum: ",
                                                           inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return false;
        }
        if (inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dimNum != LSE_DIM_NUM) { // 2: [headNum, nTokens]
            extError.errorDesc = "dimNum of prevLse should be 2!";
            extError.errorData =
                OperationUtil::ConcatInfo("But got prevLse dimNum: ", inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dimNum);
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return false;
        }
    }
    return true;
}

bool RingMLAOperation::QSplitDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const
{
    if (inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != QK_SPLIT1_HEAD_SIZE) {
        extError.errorDesc = OperationUtil::ConcatInfo("headSize of querySplit1 must be ", QK_SPLIT1_HEAD_SIZE);
        extError.errorData = OperationUtil::ConcatInfo(
            "But got querySplit1[2] headSize: ", inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    if (inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != QK_SPLIT2_HEAD_SIZE) {
        extError.errorDesc = OperationUtil::ConcatInfo("headSize of querySplit2 must be ", QK_SPLIT2_HEAD_SIZE);
        extError.errorData = OperationUtil::ConcatInfo(
            "But got querySplit2[2] headSize: ", inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t qNTokens = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_N_TOKENS_IDX];
    if (inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_N_TOKENS_IDX] != qNTokens) {
        extError.errorDesc = "querySplit1[0] nTokens must be same as querySplit2[0] nTokens.";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got querySplit1[0] nTkens: ", qNTokens,
            ", querySplit2[0] nTkens: ", inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_N_TOKENS_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t qHeadNum = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_HEAD_NUM_IDX];
    if (inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_HEAD_NUM_IDX] != qHeadNum) {
        extError.errorDesc = "querySplit1[1] must be same as querySplit2[1].";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got querySplit1[1] headNum: ", qHeadNum,
            ", querySplit2[1] headNum: ", inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_HEAD_NUM_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    return true;
}

bool RingMLAOperation::KSplitDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const
{
    if (inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != QK_SPLIT1_HEAD_SIZE) {
        extError.errorDesc = OperationUtil::ConcatInfo("headSize of keySplit1 must be ", QK_SPLIT1_HEAD_SIZE);
        extError.errorData = OperationUtil::ConcatInfo(
            "But got keySplit1[2] headSize: ", inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    if (inTensorDescs.at(IN_KEY_SPLIT2_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != QK_SPLIT2_HEAD_SIZE) {
        extError.errorDesc = OperationUtil::ConcatInfo("headSize of keySplit1 must be ", QK_SPLIT2_HEAD_SIZE);
        extError.errorData = OperationUtil::ConcatInfo(
            "But got keySplit1[2] headSize: ", inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t kvNTokens = inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_N_TOKENS_IDX];
    if (inTensorDescs.at(IN_KEY_SPLIT2_INDEX).shape.dims[QKV_N_TOKENS_IDX] != kvNTokens) {
        extError.errorDesc = "keySplit1[0] nTokens must be same as keySplit2[0] nTokens,";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got keySplit1[0] nTokens: ", kvNTokens,
            ", keySplit2[0] nTokens: ", inTensorDescs.at(IN_KEY_SPLIT2_INDEX).shape.dims[QKV_N_TOKENS_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t kvHeadNum = inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_HEAD_NUM_IDX];
    if (inTensorDescs.at(IN_KEY_SPLIT2_INDEX).shape.dims[QKV_HEAD_NUM_IDX] != kvHeadNum) {
        extError.errorDesc = "keySplit1[1] headNum must be same as keySplit2[1] headNum,";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got keySplit1[1] headNum: ", kvHeadNum,
            ", keySplit2[1] headNum: ", inTensorDescs.at(IN_QUERY_SPLIT2_INDEX).shape.dims[QKV_HEAD_NUM_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    return true;
}

Status RingMLAOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    ExternalError extError;
    extError.errorType = ERROR_INVALID_TENSOR_DIM_NUM;
    extError.solutionDesc = "Please check the shape of inTensors.";
    if (!DimNumCheck(inTensorDescs, extError)) {
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    // qkv shape: [q/kv nTokens, q/kv HeadNum, qk/v headSize]
    extError.errorType = ERROR_INVALID_TENSOR_DIM;
    extError.solutionDesc = "Please check the shape of querySplit1, querySplit2, keySplit1, keySplit2 and value.";
    if (!QSplitDimCheck(inTensorDescs, extError)) {
        return extError.errorType;
    }
    if (!KSplitDimCheck(inTensorDescs, extError)) {
        return extError.errorType;
    }
    int64_t kvHeadNum = inTensorDescs.at(IN_KEY_SPLIT1_INDEX).shape.dims[QKV_HEAD_NUM_IDX];
    if (inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_NUM_IDX] != kvHeadNum) {
        extError.errorDesc = "headNum of key and value should be the same,";
        extError.errorData = OperationUtil::ConcatInfo("key headNum: ", kvHeadNum, ", value headNum: ",
                                                       inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_NUM_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return extError.errorType;
    }
    if (inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != QK_SPLIT1_HEAD_SIZE) {
        extError.errorDesc = OperationUtil::ConcatInfo("headSize of value should be ", QK_SPLIT1_HEAD_SIZE);
        extError.errorData = OperationUtil::ConcatInfo("But got value headSize: ",
                                                       inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return extError.errorType;
    }
    if (inTensorDescs.at(IN_MASK_INDEX).shape.dimNum == 3) { // 3: mask: [batch, maxSeqLen, maxSeqLen]
        // 1: seqlen shape: [batch]
        int64_t batch = inTensorDescs.at(IN_SEQLEN_INDEX).shape.dims[SEQLEN_BATCH_IDX];
        if (inTensorDescs.at(IN_SEQLEN_INDEX).shape.dimNum == 2) {      // 2: seqlen shape: [2, batch]
            if (inTensorDescs.at(IN_SEQLEN_INDEX).shape.dims[0] != 2) { // 2: seqlen shape: [2, batch]
                extError.errorDesc = "2D seqlen should have the first dim as 2.";
                extError.errorData = OperationUtil::ConcatInfo("But got seqlen dims[0] = ",
                                                               inTensorDescs.at(IN_SEQLEN_INDEX).shape.dims[0]);
                ATB_LOG(ERROR) << GetLogPrefix() << extError;
            }
            batch = inTensorDescs.at(IN_SEQLEN_INDEX).shape.dims[1];
        }
        if (inTensorDescs.at(IN_MASK_INDEX).shape.dims[0] != batch) {
            extError.errorDesc = "batch of seqlen and mask should be the same.";
            extError.errorData = OperationUtil::ConcatInfo(
                "seqlen batch: ", batch, ", mask batch: ", inTensorDescs.at(IN_MASK_INDEX).shape.dims[0]);
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return extError.errorType;
        }
    }
    return NO_ERROR;
}

bool RingMLAOperation::InputLseDimCheck(const SVector<TensorDesc> &inTensorDescs, ExternalError &extError) const
{
    int64_t qNTokens = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_N_TOKENS_IDX];
    if (inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_N_TOKENS_IDX] != qNTokens) {
        extError.errorDesc = "nTokens of query and prevOut should be the same!";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got query nTokens: ", qNTokens,
            ", prevOut nTokens: ", inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_N_TOKENS_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    if (inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dims[LSE_N_TOKENS_IDX] != qNTokens) {
        extError.errorDesc = "nTokens of query and prvLse should be the same!";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got query nTokens: ", qNTokens,
            ", prevLse nTokens: ", inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dims[LSE_N_TOKENS_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t headNum = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_HEAD_NUM_IDX];
    if (inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_HEAD_NUM_IDX] != headNum) {
        extError.errorDesc = "headNum of query and prevOut should be the same!";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got query headNum: ", headNum,
            ", prevOut headNum: ", inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_HEAD_NUM_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    if (inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dims[LSE_HEAD_NUM_IDX] != headNum) {
        extError.errorDesc = "headNum of query and prevLse should be the same!";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got query headNum: ", headNum,
            ", prevLse headNum: ", inTensorDescs.at(IN_PREV_LSE_INDEX).shape.dims[LSE_HEAD_NUM_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    int64_t headSizeV = inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_SIZE_IDX];
    if (inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] != headSizeV) {
        extError.errorDesc = "headSize of value and prevOut should be the same!";
        extError.errorData = OperationUtil::ConcatInfo(
            "But got value headSize: ", headSizeV,
            ", prevOut headSize: ", inTensorDescs.at(IN_PREV_OUT_INDEX).shape.dims[QKV_HEAD_SIZE_IDX]);
        ATB_LOG(ERROR) << GetLogPrefix() << extError;
        return false;
    }
    return true;
}

Status RingMLAOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (isInputSoftmaxLse_) {
        ExternalError extError;
        extError.errorType = ERROR_INVALID_TENSOR_DIM;
        extError.solutionDesc = "Please check the shape of inTensors: query, value, prevOut and prevLse.";
        if (!InputLseDimCheck(inTensorDescs, extError)) {
            return extError.errorType;
        }
    }
    return NO_ERROR;
}

Status RingMLAOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(OUT_PREV_OUT_INDEX) = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX); // 0: output
    outTensorDescs.at(OUT_PREV_OUT_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] =
        inTensorDescs.at(IN_VALUE_INDEX).shape.dims[QKV_HEAD_SIZE_IDX];
    if (isInputSoftmaxLse_) {
        // 输入lse，且输出lse，复用
        outTensorDescs.at(OUT_PREV_LSE_INDEX) = inTensorDescs.at(IN_PREV_LSE_INDEX); // 1: softmaxLse, 6: prevLse
        return NO_ERROR;
    }
    outTensorDescs.at(OUT_PREV_LSE_INDEX) = inTensorDescs.at(IN_QUERY_SPLIT1_INDEX); // 1: softmaxLse, 0: query
    outTensorDescs.at(OUT_PREV_LSE_INDEX).shape.dims[LSE_N_TOKENS_IDX] =
        inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_N_TOKENS_IDX];
    outTensorDescs.at(OUT_PREV_LSE_INDEX).shape.dims[LSE_HEAD_NUM_IDX] =
        inTensorDescs.at(IN_QUERY_SPLIT1_INDEX).shape.dims[QKV_HEAD_NUM_IDX];
    // query: [nTokens, headNum, headSize] 删除headSize
    outTensorDescs.at(OUT_PREV_LSE_INDEX).shape.dims[QKV_HEAD_SIZE_IDX] = 0;
    outTensorDescs.at(OUT_PREV_LSE_INDEX).dtype = ACL_FLOAT;
    outTensorDescs.at(OUT_PREV_LSE_INDEX).shape.dimNum = LSE_DIM_NUM; // 1: softmaxLse 2: [nTokens, headNum]
    return NO_ERROR;
}

Status RingMLAOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status st;
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);
    SVector<TensorDesc> targetOutTensorDescs = {};
    targetOutTensorDescs.reserve(BASE_OUT_TENSOR_NUM);
    targetOutTensorDescs.resize(BASE_OUT_TENSOR_NUM);
    InferShapeImpl(inTensorDescs, targetOutTensorDescs);
    ExternalError extError;
    extError.errorType = ERROR_INVALID_TENSOR_DIM;
    extError.solutionDesc = "Please check the shape of outTensor.";
    std::stringstream ss;
    for (size_t i = 0; i < BASE_OUT_TENSOR_NUM; ++i) {
        if (!TensorUtil::TensorDescEqual(outTensorDescs.at(i), targetOutTensorDescs.at(i))) {
            extError.errorDesc = OperationUtil::ConcatInfo("Invalid outTensor shape at outTensors[", i, "].");
            ss.str("");
            ss.clear();
            ss << "Target outTensor shape: [";
            int32_t dimNum = static_cast<int32_t>(targetOutTensorDescs.at(i).shape.dimNum);
            for (int32_t j = 0; j < dimNum - 1; ++j) {
                ss << targetOutTensorDescs.at(i).shape.dims[j] << ", ";
            }
            ss << targetOutTensorDescs.at(i).shape.dims[dimNum - 1] << "], ";
            ss << "but got outTensor shape: [";
            dimNum = static_cast<int32_t>(outTensorDescs.at(i).shape.dimNum);
            for (int32_t j = 0; j < dimNum - 1; ++j) {
                ss << outTensorDescs.at(i).shape.dims[j] << ", ";
            }
            ss << outTensorDescs.at(i).shape.dims[dimNum - 1] << "]. ";
            ss << "Target outTensor dtype: " << targetOutTensorDescs.at(i).dtype << ", ";
            ss << "outTensor dtype: " << outTensorDescs.at(i).dtype << ", ";
            ss << "Target outTensor format: " << targetOutTensorDescs.at(i).format << ", ";
            ss << "outTensor format: " << outTensorDescs.at(i).format;
            extError.errorData = ss.str();
            ATB_LOG(ERROR) << GetLogPrefix() << extError;
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}


std::shared_ptr<Runner> RingMLAOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RingMLAOpsRunner>(param_);
}

nlohmann::json RingMLAOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
