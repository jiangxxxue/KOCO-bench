/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_latent_attention_operation.h"
#include "multi_latent_attention_ops_runner.h"
#include "multi_latent_attention_ops_runner_prefill.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 6;
static const uint32_t OUT_TENSOR_NUM_1 = 1;
static const uint32_t OUT_TENSOR_NUM_2 = 2;
static const uint32_t KVCACHE_INDEX = 2;
static const uint32_t KVCACHE_ROPE_INDEX = 3;
static const uint32_t BLOCK_TABLES_INDEX = 4;
static const uint32_t CONTEXTLENS_INDEX = 5;
static const uint32_t INNER_DIM_512 = 512;
static const uint32_t INNER_DIM_64 = 64;
static const uint32_t INNER_DIM_32 = 32;
static const uint32_t INNER_DIM_16 = 16;
static const uint32_t INNER_DIM_4 = 4;
static const uint32_t NZ_ALIGN_32 = 32;
static const uint32_t NZ_ALIGN_16 = 16;
static const uint32_t MAX_BATCH_SIZE_8192 = 8192;
static const uint32_t IN_TENSOR_0 = 0;
static const uint32_t IN_TENSOR_1 = 1;
static const uint32_t IN_TENSOR_2 = 2;
static const uint32_t IN_TENSOR_3 = 3;
static const uint32_t IN_TENSOR_4 = 4;
static const uint32_t IN_TENSOR_5 = 5;
static const uint32_t IN_TENSOR_6 = 6;
static const uint32_t IN_TENSOR_7 = 7;
static const uint32_t DIM_2 = 2;
static const uint32_t EM_BED_DIM_V = 128;
} // namespace

namespace atb {

static bool ParamRangeCheck(const infer::MultiLatentAttentionParam &opParam);
static bool ParamCheck(const infer::MultiLatentAttentionParam &opParam);
static bool ParamPrefillCheck(const infer::MultiLatentAttentionParam &opParam);

template <>
Status CreateOperation(const infer::MultiLatentAttentionParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2/A3 Inference Product";
        return ERROR_INVALID_PARAM;
    }
    if (!ParamRangeCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    if (opParam.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        if (!ParamPrefillCheck(opParam)) {
            return ERROR_INVALID_PARAM;
        }
    } else {
        if (!ParamCheck(opParam)) {
            return ERROR_INVALID_PARAM;
        }
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new (std::nothrow) MultiLatentAttentionOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new MultiLatentAttentionOperation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

static bool ParamCheck(const infer::MultiLatentAttentionParam &opParam)
{
    if (opParam.headNum != 8 && opParam.headNum != 16 && opParam.headNum != 32 &&  // 8, 16, 32: headNum
        opParam.headNum != 64 && opParam.headNum != 128) {                         // 64, 128: headNum
        ATB_LOG(ERROR) << "headNum should be {8,16,32,64,128}";
        return false;
    }
    if (opParam.kvHeadNum != 1) {
        ATB_LOG(ERROR) << "kvHeadNum should be 1, only support MQA";
        return false;
    }
    if (opParam.cacheMode == infer::MultiLatentAttentionParam::CacheMode::KVCACHE) {
        ATB_LOG(ERROR) << "dont support cacheMode KVCACHE yet";
        return false;
    }
    if ((opParam.calcType != infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC &&
         opParam.calcType != infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) &&
        opParam.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        ATB_LOG(ERROR) << "only mtp(CALC_TYPE_SPEC) support mask";
        return false;
    }
    if ((opParam.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) &&
        (opParam.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING ||
         opParam.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) &&
        (opParam.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED)) {
        ATB_LOG(ERROR) << "int8nz lse not support mask";
        return false;
    }
    return true;
}

static bool ParamRangeCheck(const infer::MultiLatentAttentionParam &opParam)
{
    if (opParam.qkScale <= 0 || opParam.qkScale > 1) {
        ATB_LOG(ERROR) << "qkScale should > 0 and <= 1";
        return false;
    }
    if (opParam.maskType < infer::MultiLatentAttentionParam::MaskType::UNDEFINED ||
        opParam.maskType > infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) {
        ATB_LOG(ERROR) << "invalid maskType";
        return false;
    }
    if (opParam.calcType < infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED ||
        opParam.calcType > infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        ATB_LOG(ERROR) << "invalid calcType";
        return false;
    }
    if (opParam.cacheMode > infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        ATB_LOG(ERROR) << "invalid cacheMode";
        return false;
    }
    return true;
}

static bool ParamPrefillCheck(const infer::MultiLatentAttentionParam &opParam)
{
    if (opParam.headNum < 1 || opParam.headNum > 128) { // 128: headNum
        ATB_LOG(ERROR) << "headNum should be >= 1 and <= 128";
        return false;
    }
    if (opParam.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        if (opParam.headNum != opParam.kvHeadNum) {
            ATB_LOG(ERROR) << "Prefill, headNum should be equal to kvHeadNum";
            return false;
        }
    }
    if (opParam.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
        opParam.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE &&
        opParam.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK &&
        opParam.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) {
        ATB_LOG(ERROR) << "Prefill, maskType support UNDEFINED, MASK_TYPE_MASK_FREE and MASK_TYPE_SWA_NORM";
        return false;
    }
    if (opParam.cacheMode != infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV) {
        ATB_LOG(ERROR) << "Prefill, cacheMode should be KROPE_CTKV";
        return false;
    }
    if (opParam.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SWA_NORM) {
        if (opParam.windowSize <= 0) {
            ATB_LOG(ERROR) << "Prefill, windowSize in swa mode should be greater than 0";
            return false;
        }
    }
    return true;
}

MultiLatentAttentionOperation::MultiLatentAttentionOperation(const infer::MultiLatentAttentionParam &param)
    : OperationBase("MultiLatentAttentionOperation"), param_(param)
{
    std::string opIrKeyStr;
    opIrKeyStr += "MultiLatentAttentionOperation";
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
        param_.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        opIrKeyStr += "Mask";
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC || param_.
                      calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) {
        opIrKeyStr += "Qlens";
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING || param_.
                      calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) {
        opIrKeyStr += "IsRing";
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        opIrKeyStr += "Int8Nz";
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        opIrKeyStr += "Nz";
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        opIrKeyStr += "Prefill";
    }
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKeyStr);
}

MultiLatentAttentionOperation::~MultiLatentAttentionOperation() {}

uint32_t MultiLatentAttentionOperation::GetInputNum() const
{
    uint32_t intensorNumBase = IN_TENSOR_NUM;
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
        param_.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        intensorNumBase++;
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        intensorNumBase++;
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC || param_.
                      calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) {
        intensorNumBase++;
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        intensorNumBase += 2; // 2: qDescale kDescale
    }
    return intensorNumBase;
}

uint32_t MultiLatentAttentionOperation::GetOutputNum() const
{
    bool isRing = param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING || param_.
                  calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING;
    return isRing ? OUT_TENSOR_NUM_2 : OUT_TENSOR_NUM_1;
}

Status MultiLatentAttentionOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                     SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = inTensorDescs.at(1).dtype;
    if ((param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING || param_.
                      calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING)) {
        outTensorDescs.at(1) = outTensorDescs.at(0);
        if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
            outTensorDescs.at(1).dtype = ACL_FLOAT;
        }
        outTensorDescs.at(1).shape.dims[2] = 1; // 2: dim2
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        return InTensorDimCheckPrefill(inTensorDescs);
    }
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                     const SVector<Tensor> &outTensors) const
{
    (void)outTensors;
    SVector<TensorDesc> inTensorDescs = {};
    SVector<TensorDesc> outTensorsDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorsDescs);
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        Status st = InTensorDimCheckPrefill(inTensorDescs);
        if (st != NO_ERROR) {
            return st;
        }
        st = TensorCheck::TensorDescsEqual(inTensors.at(0).desc, outTensors.at(0).desc);
        if (st == ERROR_INVALID_TENSOR_SIZE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dimNum and inTensor0 dimNum should be equal";
        } else if (st == ERROR_INVALID_TENSOR_DIM) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dims and inTensor0 dims should be same";
        }
        return st;
    }
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if ((param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING ||
         param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) &&
        param_.maskType == infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
        param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        st = DimCheckInt8NzLse(inTensorDescs, outTensorsDescs);
    }
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[1];
    if (blockSize > 128) { // 128 : maxblocksize
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != blockSize) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 0 and dim 1 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[2] != param_.kvHeadNum ||      // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != param_.kvHeadNum) { // 2: dim 2
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of kvCache(intensor2) and kvCacheRope(intensor3) equal to kvHeadNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 ||             // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != INNER_DIM_512) { // 3: dim 3
        ATB_LOG(ERROR) << GetLogPrefix() << "head_size of query(intensor0) and kvCache(intensor2) should be 512";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 ||                  // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != INNER_DIM_64) { // 3: dim 3
        ATB_LOG(ERROR) << GetLogPrefix() << "head_size of queryRope(intensor1) and kvCacheRope(intensor3) should be 64";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheckNz(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[2]; // 2: dim 2
    if (blockSize > 128) {                                            // 128 : maxblocksize
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != blockSize) { // 2: dim 2
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 0 and dim 2 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != NZ_ALIGN_16 ||      // 3: dim 3
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != NZ_ALIGN_16) { // 3: dim 3
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 3 of kvCache(intensor2) and kvCacheRope(intensor3) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 ||            // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[1] != INNER_DIM_32) { // 1: dim 1
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "head_size of query(intensor0) should be 512, dim 1 of kvCache(intensor2) should be 32";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 ||                 // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != INNER_DIM_4) { // 1: dim 1
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "head_size of queryRope(intensor1) should be 64, dim 1 of kvCacheRope(intensor3) should be 4";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheckInt8Nz(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t numBlocks = inTensorDesc.at(KVCACHE_INDEX).shape.dims[0];
    int64_t blockSize = inTensorDesc.at(KVCACHE_INDEX).shape.dims[2]; // 2: dim 2
    if (blockSize > 128) {                                            // 128 : maxblocksize
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSize shoule <= 128";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[0] != numBlocks ||
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[2] != blockSize) { // 2: dim 2
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 0 and dim 2 of kvCache(intensor2) and kvCacheRope(intensor3) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(KVCACHE_INDEX).shape.dims[3] != NZ_ALIGN_32 ||      // 3: dim 3
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[3] != NZ_ALIGN_16) { // 3: dim 3
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 3 of kvCache(intensor2) should be 32 and kvCacheRope(intensor3) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[1] != param_.headNum || inTensorDesc.at(1).shape.dims[1] != param_.headNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of query(intensor0) and queryRope(intensor1) equal to headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (inTensorDesc.at(0).shape.dims[2] != INNER_DIM_512 || // 2: dim 2
        inTensorDesc.at(KVCACHE_INDEX).shape.dims[1] != INNER_DIM_16) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "head_size of query(intensor0) should be 512, dim 1 of kvCache(intensor2) should be 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(1).shape.dims[2] != INNER_DIM_64 || // 2: dim 2
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dims[1] != INNER_DIM_4) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "head_size of queryRope(intensor1) should be 64, dim 1 of kvCacheRope(intensor3) should be 4";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::DimCheckSpec(const SVector<TensorDesc> &inTensorDesc, size_t idx) const
{
    if (inTensorDesc.at(idx).shape.dimNum != 1) { // 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDesc.at(idx).shape.dims[0] != inTensorDesc.at(CONTEXTLENS_INDEX).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of qSeqlen(intensor" << idx
                       << ") should be equal to dim0 of contextLens(intensor5)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::DimCheckInt8Nz(const SVector<TensorDesc> &inTensorDesc, size_t idx) const
{
    if (inTensorDesc.at(idx).shape.dimNum != 1 || inTensorDesc.at(idx + 1).shape.dimNum != 1) { // 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDesc.at(idx).shape.dims[0] != param_.headNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of of qkDescale(intensor" << idx
                       << ") should be equal to dim0 of headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(idx + 1).shape.dims[0] != param_.headNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of of pvDescale(intensor" << (idx + 1)
                       << ") should be equal to dim0 of headNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::DimCheckInt8NzLse(const SVector<TensorDesc> &inTensorDesc,
                                                        const SVector<TensorDesc> &outTensorDesc) const
{
    if (outTensorDesc.at(0).shape.dimNum != inTensorDesc.at(0).shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid outtensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if ((outTensorDesc.at(0).shape.dims[0] != inTensorDesc.at(0).shape.dims[0]) ||
        (outTensorDesc.at(0).shape.dims[1] != inTensorDesc.at(0).shape.dims[1]) ||
        (outTensorDesc.at(0).shape.dims[DIM_2] != inTensorDesc.at(0).shape.dims[DIM_2])) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid outtensor dims";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensorDesc.at(1).shape.dimNum != inTensorDesc.at(0).shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid outtensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if ((outTensorDesc.at(1).shape.dims[0] != inTensorDesc.at(0).shape.dims[0]) ||
        (outTensorDesc.at(1).shape.dims[1] != inTensorDesc.at(0).shape.dims[1]) ||
        (outTensorDesc.at(1).shape.dims[DIM_2] != 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid outtensor dims";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::DimCheck(const SVector<TensorDesc> &inTensorDesc) const
{
    if (inTensorDesc.at(0).shape.dimNum != 3 ||                  // 0: query 3: 3 dims
        inTensorDesc.at(1).shape.dimNum != 3 ||                  // 1: queryRope 3: 3 dims
        inTensorDesc.at(KVCACHE_INDEX).shape.dimNum != 4 ||      // 4: 4 dims
        inTensorDesc.at(KVCACHE_ROPE_INDEX).shape.dimNum != 4 || // 4: 4 dims
        inTensorDesc.at(BLOCK_TABLES_INDEX).shape.dimNum != 2 || // 2: 2 dims
        inTensorDesc.at(CONTEXTLENS_INDEX).shape.dimNum != 1) {  // 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t numTokens = inTensorDesc.at(0).shape.dims[0];
    int64_t batch = inTensorDesc.at(BLOCK_TABLES_INDEX).shape.dims[0];
    if (batch > MAX_BATCH_SIZE_8192) {
        ATB_LOG(ERROR) << GetLogPrefix() << "batch should be <= " << MAX_BATCH_SIZE_8192;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_UNDEFINED && numTokens != batch) {
        ATB_LOG(ERROR) << GetLogPrefix() << "numTokens and batch should be same in decoder stage";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(CONTEXTLENS_INDEX).shape.dims[0] != batch) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 0 of block_tables(intensor4) and contextLens(intensor5) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    Status st = NO_ERROR;
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::KROPE_CTKV) {
        st = QKVDimCheck(inTensorDesc);
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        st = QKVDimCheckInt8Nz(inTensorDesc);
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::NZCACHE) {
        st = QKVDimCheckNz(inTensorDesc);
    }
    if (st != NO_ERROR) {
        return st;
    }
    if ((param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING ||
         param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) &&
        param_.maskType == infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
        param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        size_t idx = IN_TENSOR_6;
        if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING) {
            st = DimCheckSpec(inTensorDesc, idx);
            if (st != NO_ERROR) {
                return st;
            }
            ++idx;
        }
        st = DimCheckInt8Nz(inTensorDesc, idx);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::InTensorDimCheckPrefill(const SVector<TensorDesc> &inTensorDesc) const
{
    if ((inTensorDesc.at(IN_TENSOR_0).shape.dimNum != 2 &&  // 0: query 2:   2 dims
         inTensorDesc.at(IN_TENSOR_0).shape.dimNum != 3) || // 0: query 3:   3 dims
        (inTensorDesc.at(IN_TENSOR_1).shape.dimNum !=       // 1: queryRope
         inTensorDesc.at(IN_TENSOR_0).shape.dimNum) ||      // 0: query
        inTensorDesc.at(IN_TENSOR_2).shape.dimNum != 3 ||   // 2: key        3: 3 dims
        inTensorDesc.at(IN_TENSOR_3).shape.dimNum != 3 ||   // 3: keyRope    3: 3 dims
        inTensorDesc.at(IN_TENSOR_4).shape.dimNum != 3 ||   // 4: value      3: 3 dims
        inTensorDesc.at(IN_TENSOR_5).shape.dimNum != 1 ||   // 5: qSeqLen    1: 1 dims
        inTensorDesc.at(IN_TENSOR_6).shape.dimNum != 1 ||   // 6: kvSeqLen   1: 1 dims
        (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE &&
         inTensorDesc.at(IN_TENSOR_7).shape.dimNum != 2)) { // 7: mask       2: 2 dims
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDesc.at(IN_TENSOR_5).shape.dims[0] > MAX_BATCH_SIZE_8192) {
        ATB_LOG(ERROR) << GetLogPrefix() << "batch should be <= " << MAX_BATCH_SIZE_8192
                       << ", batch = " << inTensorDesc.at(IN_TENSOR_5).shape.dims[0];
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_5).shape.dims[0] != inTensorDesc.at(IN_TENSOR_6).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 0 of qSeqLen(intensor5) and dim 0 of kvSeqLen(intensor6) shouble be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    Status st = QKVDimCheckPrefill(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        if (inTensorDesc.at(IN_TENSOR_7).shape.dims[0] != 512) { // 512: 压缩mask的规格
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of mask(intensor7) should be equal to 512";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (inTensorDesc.at(IN_TENSOR_7).shape.dims[1] != 512) { // 512: 压缩mask的规格
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of mask(intensor7) should be equal to 512";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QDimCheckPrefill(const SVector<TensorDesc> &inTensorDesc) const
{
    int64_t numTokens = inTensorDesc.at(IN_TENSOR_0).shape.dims[0];
    if (inTensorDesc.at(IN_TENSOR_0).shape.dimNum == 2) { // 2: 2 dim
        if (inTensorDesc.at(IN_TENSOR_0).shape.dims[1] != param_.headNum * EM_BED_DIM_V) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of query(intensor0) should be equal to headNum * embeddimV";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensorDesc.at(IN_TENSOR_0).shape.dims[1] != param_.headNum) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of query(intensor0) should be equal to headNum ";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (inTensorDesc.at(IN_TENSOR_0).shape.dims[DIM_2] != EM_BED_DIM_V) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 2 of query(intensor0) should be equal to embeddimV ";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensorDesc.at(IN_TENSOR_1).shape.dims[0] != numTokens) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of query(intensor0) and queryRope(intensor1) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_1).shape.dimNum == 2) {                        // 2: 2 dim
        if (inTensorDesc.at(IN_TENSOR_1).shape.dims[1] != param_.headNum * 64) { // 64: embeddim - embeddimV
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of queryRope(intensor1)should be equal to headNum * 64";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensorDesc.at(IN_TENSOR_1).shape.dims[1] != param_.headNum) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of queryRope(intensor1)should be equal to headNum ";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (inTensorDesc.at(IN_TENSOR_1).shape.dims[DIM_2] != 64) { // 64: embeddim - embeddimV
            ATB_LOG(ERROR) << GetLogPrefix() << "dim 2 of queryRope(intensor1)should be equal to 64 ";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::KVDimCheckPrefill(const SVector<TensorDesc> &inTensorDesc) const
{
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        if ((inTensorDesc.at(IN_TENSOR_2).shape.dims[1] != param_.kvHeadNum ||
             inTensorDesc.at(IN_TENSOR_2).shape.dims[DIM_2] != EM_BED_DIM_V) &&
            (inTensorDesc.at(IN_TENSOR_2).shape.dims[0] != inTensorDesc.at(IN_TENSOR_5).shape.dims[0] ||
             inTensorDesc.at(IN_TENSOR_2).shape.dims[DIM_2] != param_.kvHeadNum * EM_BED_DIM_V)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "key(intensor2) is invalid, only support [B*S,N,D] / [B,S,N*D]";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (inTensorDesc.at(IN_TENSOR_2).shape.dims[0] != inTensorDesc.at(IN_TENSOR_5).shape.dims[0] ||
            inTensorDesc.at(IN_TENSOR_2).shape.dims[DIM_2] != param_.kvHeadNum * EM_BED_DIM_V) {
            ATB_LOG(ERROR) << GetLogPrefix() << "key(intensor2) is invalid, only support [B,S,N*D]";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (inTensorDesc.at(IN_TENSOR_3).shape.dims[0] != inTensorDesc.at(IN_TENSOR_2).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of keyRope(intensor3) and key(intensor2) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_3).shape.dims[1] != inTensorDesc.at(IN_TENSOR_2).shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of keyRope(intensor3) and key(intensor2) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_3).shape.dims[DIM_2] !=
        inTensorDesc.at(IN_TENSOR_2).shape.dims[DIM_2] / 2) { // 2: embiddim-embeddimV=64,embeddimV=128,128/64 = 2
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "dim 2 of keyRope(intensor3) should be equal to dim 2 of key(intensor2) / 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_4).shape.dims[0] != inTensorDesc.at(IN_TENSOR_2).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 0 of value(intensor4) and key(intensor2) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_4).shape.dims[1] != inTensorDesc.at(IN_TENSOR_2).shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 1 of value(intensor4) and key(intensor2) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDesc.at(IN_TENSOR_4).shape.dims[DIM_2] != inTensorDesc.at(IN_TENSOR_2).shape.dims[DIM_2]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim 2 of value(intensor4) and key(intensor2) should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOperation::QKVDimCheckPrefill(const SVector<TensorDesc> &inTensorDesc) const
{
    Status st = NO_ERROR;
    st = QDimCheckPrefill(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    st = KVDimCheckPrefill(inTensorDesc);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> MultiLatentAttentionOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_PREFILL) {
        return std::make_shared<MultiLatentAttentionOpsRunnerPrefill>(param_);
    }
    return std::make_shared<MultiLatentAttentionOpsRunner>(param_);
}

nlohmann::json MultiLatentAttentionOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb