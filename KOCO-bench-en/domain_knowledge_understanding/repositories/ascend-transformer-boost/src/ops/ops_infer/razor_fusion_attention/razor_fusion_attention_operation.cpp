/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "razor_fusion_attention_operation.h"
#include "atb/utils/param_to_json.h"
#include "razor_fusion_attention_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t DEFAULT_HEAD_NUM = 1;
static const int32_t DEFAULT_KV_HEAD_NUM = 1;
static const int64_t DEFAULT_HEAD_SIZE = 128;
static const int32_t ZERO = 0;
static const int32_t PRE_TOKENS_BLOCK_SIZE = 128;
static const int32_t NEXT_TOKENS_BLOCK_SIZE = 128;
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t IN_TENSOR_0_QUERY = 0;
static const uint32_t IN_TENSOR_1_KEY = 1;
static const uint32_t IN_TENSOR_2_VALUE = 2;
static const uint64_t IN_TENSOR_DIM_NUM_2 = 2;
static const uint64_t IN_TENSOR_DIM_NUM_3 = 3;

bool RazorFusionAttentionParamCheck(const infer::RazorFusionAttentionParam &opParam)
{
    if (opParam.headNum != DEFAULT_HEAD_NUM || opParam.kvHeadNum != DEFAULT_KV_HEAD_NUM) {
        ATB_LOG(ERROR) << "headNum and kvHeadNum must be 1.";
        return false;
    }
    if (opParam.razorLen < ZERO) {
        ATB_LOG(ERROR) << "razorLen must be greater than or equal to 0.";
        return false;
    }
    if (opParam.preTokens < ZERO || ((opParam.preTokens % PRE_TOKENS_BLOCK_SIZE) != ZERO)) {
        ATB_LOG(ERROR) << "preTokens must be greater than or equal to 0. preTokens should be aligned to 128";
        return false;
    }
    if (opParam.nextTokens < ZERO || ((opParam.nextTokens % NEXT_TOKENS_BLOCK_SIZE) != ZERO)) {
        ATB_LOG(ERROR) << "nextTokens must be greater than or equal to 0. nextTokens should be aligned to 128";
        return false;
    }
    if (opParam.tileQ < ZERO || opParam.tileKv < ZERO || opParam.textQLen < ZERO || opParam.textKvLen < ZERO) {
        ATB_LOG(ERROR) << "tileQ, tileKv, textQLen and textKvLen must be greater than or equal to 0.";
        return false;
    }
    if (opParam.razorLen == ZERO && opParam.tileQ == ZERO && opParam.textQLen == ZERO) {
        ATB_LOG(ERROR) << "razorLen, tileQ and textQLen cannot all be 0.";
        return false;
    }
    if (opParam.razorLen == ZERO && opParam.tileKv == ZERO && opParam.textKvLen == ZERO) {
        ATB_LOG(ERROR) << "razorLen, tileKv and textKvLen cannot all be 0.";
        return false;
    }
    return true;
}

template <> Status CreateOperation(const infer::RazorFusionAttentionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "RazorFusionAttention only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if (!RazorFusionAttentionParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) RazorFusionAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RazorFusionAttentionOperation::RazorFusionAttentionOperation(const infer::RazorFusionAttentionParam &param)
    : OperationBase("RazorFusionAttentionOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RazorFusionAttentionOperation");
}

RazorFusionAttentionOperation::~RazorFusionAttentionOperation() {}

uint32_t RazorFusionAttentionOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RazorFusionAttentionOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RazorFusionAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t queryDimNum = inTensorDescs.at(IN_TENSOR_0_QUERY).shape.dimNum;
    uint64_t keyDimNum = inTensorDescs.at(IN_TENSOR_1_KEY).shape.dimNum;
    uint64_t valueDimNum = inTensorDescs.at(IN_TENSOR_2_VALUE).shape.dimNum;
    if (queryDimNum != keyDimNum || queryDimNum != valueDimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor dimNum must be the same.";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (queryDimNum != IN_TENSOR_DIM_NUM_2 && queryDimNum != IN_TENSOR_DIM_NUM_3) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor dimNum must be 2 or 3.";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }

    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status RazorFusionAttentionOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t queryDimNum = inTensorDescs.at(IN_TENSOR_0_QUERY).shape.dimNum;
    // 校验query key value的dim0
    int64_t queryDim0 = inTensorDescs.at(IN_TENSOR_0_QUERY).shape.dims[0];
    int64_t qSeqLen = param_.razorLen * param_.tileQ + param_.textQLen;
    if (queryDim0 % qSeqLen != ZERO) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor0 dim0 should be aligned to (razorLen * tileQ + textQLen)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t keyDim0 = inTensorDescs.at(IN_TENSOR_1_KEY).shape.dims[0];
    int64_t valueDim0 = inTensorDescs.at(IN_TENSOR_2_VALUE).shape.dims[0];
    int64_t kvSeqLen = param_.razorLen * param_.tileKv + param_.textKvLen;
    if (keyDim0 != valueDim0 || keyDim0 % kvSeqLen != ZERO || valueDim0 % kvSeqLen != ZERO) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor1 dim0 must be equal to intensor2 dim0, and"
                       << "intensor1 dim0 and intensor2 dim0 should be aligned to (razorLen * tileKv + textKvLen)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t queryDim0batch = queryDim0 / qSeqLen;
    int64_t keyDim0batch = keyDim0 / kvSeqLen;
    int64_t valueDim0batch = valueDim0 / kvSeqLen;
    if (queryDim0batch != keyDim0batch || queryDim0batch != valueDim0batch) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor0Dim0/(razorLen * tileQ + textQLen) must equal to "
                       << "intensor1Dim0/(razorLen * tileKv + textKvLen) and "
                       << "intensor2Dim0/(razorLen * tileKv + textKvLen).";
        return ERROR_INVALID_TENSOR_DIM;
    }
    // 校验query key value的dim1
    int64_t queryDim1 = inTensorDescs.at(IN_TENSOR_0_QUERY).shape.dims[1];
    int64_t keyDim1 = inTensorDescs.at(IN_TENSOR_1_KEY).shape.dims[1];
    int64_t valueDim1 = inTensorDescs.at(IN_TENSOR_2_VALUE).shape.dims[1];
    if (queryDimNum == IN_TENSOR_DIM_NUM_2) {
        if (queryDim1 != DEFAULT_HEAD_SIZE || keyDim1 != DEFAULT_HEAD_SIZE || valueDim1 != DEFAULT_HEAD_SIZE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "when dimNum is 2, intensor0Dim1, intensor1Dim1 and intensor2Dim1 must equal to 128.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (queryDimNum == IN_TENSOR_DIM_NUM_3 && (queryDim1 != 1 || keyDim1 != 1 || valueDim1 != 1)) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "when dimNum is 3, intensor0Dim1, intensor1Dim1, and intensor2Dim1 must be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    // 校验query key value的dim2
    if (queryDimNum == IN_TENSOR_DIM_NUM_3) {
        int64_t queryDim2 = inTensorDescs.at(IN_TENSOR_0_QUERY).shape.dims[2];
        int64_t keyDim2 = inTensorDescs.at(IN_TENSOR_1_KEY).shape.dims[2];
        int64_t valueDim2 = inTensorDescs.at(IN_TENSOR_2_VALUE).shape.dims[2];
        if (queryDim2 != DEFAULT_HEAD_SIZE || keyDim2 != DEFAULT_HEAD_SIZE || valueDim2 != DEFAULT_HEAD_SIZE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "intensor0Dim2, intensor1Dim2 and intensor2Dim2 must equal to 128.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status RazorFusionAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                     SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status RazorFusionAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                     const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(0), outTensors.at(0).desc)) {
        ATB_LOG(ERROR) << "outTensor dtype/format/shape must be consistent with inTensor0 dtype/format/shape";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> RazorFusionAttentionOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RazorFusionAttentionOpsRunner>(param_);
}

nlohmann::json RazorFusionAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb