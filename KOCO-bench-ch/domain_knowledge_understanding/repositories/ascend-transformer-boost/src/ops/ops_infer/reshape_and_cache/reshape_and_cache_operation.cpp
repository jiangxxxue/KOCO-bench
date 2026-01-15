/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reshape_and_cache_operation.h"
#include "reshape_and_cache_ops_runner.h"
#include "reshape_and_cache_ops_runner_310p.h"
#include "reshape_and_cache_ops_runner_SISO.h"
#include "reshape_and_cache_ops_runner_A2_NZ.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {

static const int64_t NZBLOCKSIZE = 16;
static const uint32_t DIM_ALIGN_16_NZ = 16;
static const uint32_t HEADSIZE_ALIGN = 32;
static const uint32_t IN_TENSOR_0_KEY = 0;
static const uint32_t IN_TENSOR_1_VALUE = 1;
static const uint32_t IN_TENSOR_2_KEYCACHE = 2;
static const uint32_t IN_TENSOR_3_VALUECACHE = 3;
static const uint32_t IN_TENSOR_4_SLOTMAPPING = 4;
static const uint32_t IN_TENSOR_0_KEY_SISO = 0;
static const uint32_t IN_TENSOR_1_KEYCACHE_SISO = 1;
static const uint32_t IN_TENSOR_2_SLOTMAPPING_SISO = 2;
static const uint32_t OUT_TENSOR_NUM_NORM = 2;
static const uint32_t OUT_TENSOR_NUM_SISO = 1;
static const int64_t BLOCK_SIZE_16_NZ = 16;
static const int64_t BLOCK_SIZE_32_NZ = 32;

template <> Status CreateOperation(const infer::ReshapeAndCacheParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        if (opParam.compressType != infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
            ATB_LOG(ERROR) << "head compress only support Atlas 800I A2 inference product";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
            ATB_LOG(ERROR) << "key_cache SISO only support Atlas 800I A2 inference product";
            return ERROR_INVALID_PARAM;
        }
    }
    if (opParam.compressType < infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED ||
        opParam.compressType > infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        ATB_LOG(ERROR) << "wrong compress head";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvCacheCfg < infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE ||
        opParam.kvCacheCfg > infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        ATB_LOG(ERROR) << "wrong kvcache config";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS &&
        opParam.compressType != infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << "key_cache SISO do not support compress func";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ &&
        opParam.compressType != infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_UNDEFINED) {
        ATB_LOG(ERROR) << "Format NZ only supports compressType:COMPRESS_TYPE_UNDEFINED! ";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) ReshapeAndCacheOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

ReshapeAndCacheOperation::ReshapeAndCacheOperation(const infer::ReshapeAndCacheParam &param)
    : OperationBase("ReshapeAndCacheOperation"), param_(param)
{
    is910b_ = GetSingleton<Config>().Is910B();
    if (!is910b_) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperationAtlas300");
    } else if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperationSISO");
    } else if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperationAtlas800A2NZ");
    } else if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperationKvHeadCompressAlibi");
    } else if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD_ROPE) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperationKvHeadCompressRope");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReshapeAndCacheOperation");
    }
}

ReshapeAndCacheOperation::~ReshapeAndCacheOperation() {}

uint32_t ReshapeAndCacheOperation::GetInputNum() const
{
    uint32_t inTensorNumBase = 5;
    if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
        inTensorNumBase -= 2; // 2: value, value_cache
    } else if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD) {
        inTensorNumBase += 2; // 2: wins, seqLen
    } else if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD_ROPE) {
        inTensorNumBase += 3; // 3: wins, seqLen, offsetIndex
    }
    return inTensorNumBase;
}

uint32_t ReshapeAndCacheOperation::GetOutputNum() const
{
    if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE ||
        param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        return OUT_TENSOR_NUM_NORM;
    } else {
        return OUT_TENSOR_NUM_SISO;
    }
}

Status ReshapeAndCacheOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE ||
        param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_TENSOR_2_KEYCACHE);
        outTensorDescs.at(1) = inTensorDescs.at(IN_TENSOR_3_VALUECACHE);
    } else {
        outTensorDescs.at(0) = inTensorDescs.at(IN_TENSOR_1_KEYCACHE_SISO);
    }
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = NO_ERROR;
    if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
        st = DimCheckSISO(inTensorDescs);
    } else if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        st = KVCacheDimCheck910BNZ(inTensorDescs);
    } else {
        st = DimCheck(inTensorDescs);
    }
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }

    if (GetSingleton<Config>().Is310B()) {
        if (inTensorDescs.at(IN_TENSOR_2_KEYCACHE).format != aclFormat::ACL_FORMAT_FRACTAL_NZ ||
            inTensorDescs.at(IN_TENSOR_3_VALUECACHE).format != aclFormat::ACL_FORMAT_FRACTAL_NZ) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Cache format only support NZ";
            return ERROR_INVALID_PARAM;
        }
    }

    Status st = NO_ERROR;
    if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
        st = DimCheckSISO(inTensorDescs);
    } else if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        st = KVCacheDimCheck910BNZ(inTensorDescs);
    } else {
        st = DimCheck(inTensorDescs);
    }
    if (st != NO_ERROR) {
        return st;
    }
    uint32_t keyCacheIndex = (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE ||
                                param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) ?
                                IN_TENSOR_2_KEYCACHE : IN_TENSOR_1_KEYCACHE_SISO;
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(keyCacheIndex), outTensors.at(0).desc) ||
        ((param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE ||
        param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) &&
        !TensorUtil::TensorDescEqual(inTensorDescs.at(IN_TENSOR_3_VALUECACHE), outTensors.at(1).desc))) {
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor & outtensor descs should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    ATB_LOG(DEBUG) << "outTensors Size:" << outTensors.size();
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEY).shape.dimNum != 3 ||         // 0: key 3: 3 dims
        inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dimNum != 3 ||       // 1: value 3: 3 dims
        inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dimNum != 4 ||    // 2: keyCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dimNum != 4 ||  // 3: valueCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_4_SLOTMAPPING).shape.dimNum != 1) { // 4: slotMapping 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t numTokens = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[0];      // 0: key
    int64_t numHead = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1];        // 1: key num_head
    int64_t numBlocks = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[0]; // 2: keyCache
    int64_t blockSizeIndex = is910b_ ? 1 : 2;
    int64_t blockSize = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[blockSizeIndex]; // 2: keyCache
    if (numTokens != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[0]) {                  // 1: value
        ATB_LOG(ERROR) << GetLogPrefix() << "numTokens should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numHead != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1]) { // 1: value numHead
        ATB_LOG(ERROR) << GetLogPrefix() << "numHead should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (numBlocks != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[0]) { // 3: valueCache
        ATB_LOG(ERROR) << GetLogPrefix() << "numBlocks should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[blockSizeIndex]) { // 3: valueCache
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return is910b_ ? KVCacheDimCheck910B(inTensorDescs) : KVCacheDimCheck310P(inTensorDescs);
}

Status ReshapeAndCacheOperation::DimCheckSISO(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEY_SISO).shape.dimNum != 3 ||         // 0: key 3: 3 dims
        inTensorDescs.at(IN_TENSOR_1_KEYCACHE_SISO).shape.dimNum != 4 ||    // 1: keyCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_2_SLOTMAPPING_SISO).shape.dimNum != 1) { // 2: slotMapping 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t numTokens = inTensorDescs.at(IN_TENSOR_0_KEY_SISO).shape.dims[0];        // 0: numTokens dim
    int64_t kNumHead = inTensorDescs.at(IN_TENSOR_0_KEY_SISO).shape.dims[1];         // 1: numHead dim
    int64_t kheadSize = inTensorDescs.at(IN_TENSOR_0_KEY_SISO).shape.dims[2];        // 2: headSize dim
    if (numTokens != inTensorDescs.at(IN_TENSOR_2_SLOTMAPPING_SISO).shape.dims[0]) { // 0 : numTokens dim
        ATB_LOG(ERROR) << GetLogPrefix() << "The dim 0 of intensor0 and dim 0 of intensor2 should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (kNumHead != inTensorDescs.at(IN_TENSOR_1_KEYCACHE_SISO).shape.dims[2]) { // 2 : kNumHead dim
        ATB_LOG(ERROR) << GetLogPrefix() << "The dim 1 of intensor0 and dim 2 of intensor1 should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (kheadSize != inTensorDescs.at(IN_TENSOR_1_KEYCACHE_SISO).shape.dims[3]) { // 3 : kheadSize dim
        ATB_LOG(ERROR) << GetLogPrefix() << "The dim 2 of intensor0 and dim 3 of intensor1 should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::KVCacheDimCheck910B(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t kNumHead = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1];    // 1: value
    int64_t vNumHead = inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1];  // 1: value
    int64_t kheadSize = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[2];   // 2: headSize dim
    int64_t vheadSize = inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[2]; // 2: headSize dim
    if (param_.compressType != infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED) {
        if (kheadSize != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[2]) { // 1: value 2: 2nd dim
            ATB_LOG(ERROR) << GetLogPrefix() << "head compress key and value headSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if ((kheadSize != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] ||    // 2: keyCache 3: headSize dim
             kheadSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[3])) { // 3: valueCache 3: headSize dim
            ATB_LOG(ERROR) << GetLogPrefix() << "head compress keyCache and valueCache headSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[2] != 1 ||   // 2: keyCache 2: dim 2
            inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[2] != 1) { // 3: valueCache 2: dim 2
            ATB_LOG(ERROR) << GetLogPrefix() << "The dim 2 of head compress keyCache and valueCache should be 1";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (kheadSize % HEADSIZE_ALIGN != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "head compress key and value headSizes must be a multiple of 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED &&
        (kNumHead != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[2] ||    // 2: keyCache
         vNumHead != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[2])) { // 3: valueCache
        ATB_LOG(ERROR) << GetLogPrefix() << "numHeads should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (kheadSize != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] ||   // 2: keyCache 3: 3rd dim
        vheadSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[3]) { // 3: valueCache 3: 3rd dim
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "key and keycache headSize should be same, value and value cache headSize should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::KVCacheDimCheck910BNZ(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEY).shape.dimNum != 3 ||         // 0: key 3: 3 dims
        inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dimNum != 3 ||       // 1: value 3: 3 dims
        inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dimNum != 4 ||    // 2: keyCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dimNum != 4 ||  // 3: valueCache 4: 4 dims
        inTensorDescs.at(IN_TENSOR_4_SLOTMAPPING).shape.dimNum != 1) { // 4: slotMapping 1: 1 dim
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[0] !=
        inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[0]) {     // 1:key value numTokens
        ATB_LOG(ERROR) << GetLogPrefix() << "numTokens should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1] !=
        inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1]) { // 1: key value numHead
        ATB_LOG(ERROR) << GetLogPrefix() << "numHead should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t kNumHead = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1];   // 1: value
    int64_t kHeadSize = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[2];   // 2: kheadSize dim
    int64_t VHeadSize = inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[2]; // 2: vheadSize dim
    int64_t vNumHead = inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1]; // 1: value
    int64_t blockSize = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[2]; // 2: keyCache; 2: blocksize
    if (blockSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[2]) { // 3: valueCache
        ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize % DIM_ALIGN_16_NZ != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The blockSize of keyCache and valueCache should be 16-aligned";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED) {
        if (inTensorDescs.at(IN_TENSOR_0_KEY).dtype == ACL_INT8) { // key: int8   keyCache last_dim=32
            if (kNumHead * kHeadSize != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[1] * BLOCK_SIZE_32_NZ ||
                inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] != BLOCK_SIZE_32_NZ) { // 3: last dim
                ATB_LOG(ERROR) << GetLogPrefix() << "int8 NZ format tensor dim should be\
                    aligned to 32 and keyCache last dim should be 32!";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else { // key: fp16/bf16   keyCache last_dim=16
            if (kNumHead * kHeadSize != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[1] * BLOCK_SIZE_16_NZ ||
                inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] != BLOCK_SIZE_16_NZ) { // 3: last dim
                ATB_LOG(ERROR) << GetLogPrefix() << "fp16/bf16 NZ format tensor dim should be\
                aligned to 16 and keyCache last dim should be 16!";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } // value valueCache
        if (vNumHead * VHeadSize != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[1] * BLOCK_SIZE_16_NZ ||
            inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[3] != BLOCK_SIZE_16_NZ) { // 3: last dim
            ATB_LOG(ERROR) << GetLogPrefix() << "NZ format tensor dim should be aligned to 16!";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status ReshapeAndCacheOperation::KVCacheDimCheck310P(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t kNumHead = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[1];   // 1: value
    int64_t headSize = inTensorDescs.at(IN_TENSOR_0_KEY).shape.dims[2];   // 2: headSize dim
    int64_t vNumHead = inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[1]; // 1: value
    int64_t blockSize = inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[2]; // 2: keyCache; 2: blocksize
    int64_t hiddenSizeK = kNumHead * headSize;
    int64_t hiddenSizeV = vNumHead * headSize;

    if (headSize != inTensorDescs.at(IN_TENSOR_1_VALUE).shape.dims[2]) { // 1: value 2: 2nd dim
        ATB_LOG(ERROR) << GetLogPrefix() << "key and value headSizes should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (blockSize % DIM_ALIGN_16_NZ != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The blockSize of keyCache and valueCache should be 16-aligned";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED) {
        if (hiddenSizeK != inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[1] * NZBLOCKSIZE ||   // 2: keyCache
            hiddenSizeV != inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[1] * NZBLOCKSIZE || // 3: valueCache
            inTensorDescs.at(IN_TENSOR_2_KEYCACHE).shape.dims[3] != NZBLOCKSIZE ||   // 2: keyCache 3: 3rd dim
            inTensorDescs.at(IN_TENSOR_3_VALUECACHE).shape.dims[3] != NZBLOCKSIZE) { // 3: valueCache 3: 3rd dim
            ATB_LOG(ERROR) << GetLogPrefix() << "NZ format tensor dim should be aligned to 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> ReshapeAndCacheOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (GetSingleton<Config>().Is910B()) {
        if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS) {
            return std::make_shared<ReshapeAndCacheOpsRunnerSISO>(param_);
        } else if (param_.kvCacheCfg == infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
            return std::make_shared<ReshapeAndCacheOpsRunnerA2NZ>(param_);
        } else {
            return std::make_shared<ReshapeAndCacheOpsRunner>(param_);
        }
    } else {
        /* NZ的场景 */
        return std::make_shared<ReshapeAndCacheOpsRunner310P>(param_);
    }
}
} // namespace atb
