/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "paged_cache_load_operation.h"
#include "paged_cache_load_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_0_KEYCACHE = 0;
static const uint32_t IN_TENSOR_1_VALUECACHE = 1;
static const uint32_t IN_TENSOR_2_BLOCKTABLE = 2;
static const uint32_t IN_TENSOR_3_CONTEXTLENS = 3;
static const uint32_t IN_TENSOR_4_KEY = 4;
static const uint32_t IN_TENSOR_5_VALUE = 5;
static const uint32_t IN_TENSOR_6_SEQ_STARTS = 6;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t IN_TENSOR_NUM = 6;
static const uint32_t IN_TENSOR_NUM_SEQ_STARTS = 7;
static const uint32_t INPUTKEY_DIM = 4;
static const uint32_t INPUTVALUE_DIM = 4;
static const uint32_t INPUTCONTEXTLENS_DIM = 1;
static const uint32_t INPUTBLOCK_DIM = 2;
static const uint32_t OUT_DIM = 3;
static const uint32_t SIXTEEN = 16;
static const uint32_t THIRTYTWO = 32;
static const uint32_t MAX_SEQ_LEN = 2048;
static const uint32_t MAX_k = 147456;
static const uint32_t MAX_v = 147456;
static const uint32_t BLOCKSIZEINDEX = 2;
static const uint32_t BLOCKSIZEINDEX_ND = 1;
static const uint32_t NUM_HEADS_INDEX = 2;
static const uint32_t HEAD_SIZE_INDEX = 3;
static const uint32_t ALIGN_INT8 = 32;
static const uint32_t ALIGN_FP16_BF16 = 16;
static const uint32_t ZERO = 0;
static const uint32_t ONE = 1;

template <> Status CreateOperation(const infer::PagedCacheLoadParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "paged cache load only support Atlas 800I A2 inference product!";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvCacheCfg < infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ ||
        opParam.kvCacheCfg > infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_ND) {
        ATB_LOG(ERROR) << "wrong kvcache config";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ &&
        (opParam.hasSeqStarts || opParam.isSeqLensCumsumMode)) {
        ATB_LOG(ERROR) << "foramt NZ donot support hasSeqStarts=true or isSeqLensCumsumMode=true!";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) PagedCacheLoadOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

PagedCacheLoadOperation::PagedCacheLoadOperation(const infer::PagedCacheLoadParam &param)
    : OperationBase("PagedCacheLoadOperation"), param_(param)
{
    if (GetSingleton<Config>().Is910B()) {
        if (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("PagedCacheLoadOperation");
        } else {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("PagedCacheLoadOperationND");
        }
    }
}

PagedCacheLoadOperation::~PagedCacheLoadOperation() {}

uint32_t PagedCacheLoadOperation::GetInputNum() const
{
    if (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) {
        return IN_TENSOR_NUM;
    } else {
        return IN_TENSOR_NUM_SEQ_STARTS;
    }
}

uint32_t PagedCacheLoadOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status PagedCacheLoadOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
    SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(IN_TENSOR_4_KEY);
    outTensorDescs.at(1) = inTensorDescs.at(IN_TENSOR_5_VALUE);
    return NO_ERROR;
}

Status PagedCacheLoadOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return DimCheck(inTensorDescs);
}

Status PagedCacheLoadOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
    const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    if (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) { // NZ
        if (outTensors.at(0).desc.shape.dims[1] !=
            inTensorDescs.at(0).shape.dims[1] * inTensorDescs.at(0).shape.dims[OUT_DIM] ||
            outTensors.at(1).desc.shape.dims[1] !=
            inTensorDescs.at(1).shape.dims[1] * inTensorDescs.at(1).shape.dims[OUT_DIM]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The last dimension of outTensors needs to remain aligned";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (outTensors.at(0).desc.shape.dims[1] != inTensorDescs.at(0).shape.dims[2] ||
                outTensors.at(0).desc.shape.dims[2] != inTensorDescs.at(0).shape.dims[3] ||
                outTensors.at(1).desc.shape.dims[1] != inTensorDescs.at(1).shape.dims[2] ||
                outTensors.at(1).desc.shape.dims[2] != inTensorDescs.at(1).shape.dims[3]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "The last dimension of outTensors needs to remain aligned";
                return ERROR_INVALID_TENSOR_DIM;
    }
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status PagedCacheLoadOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t numBlocks = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[0]; // 0: keyCache
    int64_t lencontext = inTensorDescs.at(IN_TENSOR_2_BLOCKTABLE).shape.dims[0]; // 2: blocktable
    int64_t sumcontext = inTensorDescs.at(IN_TENSOR_4_KEY).shape.dims[0]; // 4: key
    if (numBlocks != inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "numBlocks should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (sumcontext != inTensorDescs.at(IN_TENSOR_5_VALUE).shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "sumcontextlens should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_ND) {
        int64_t blockSize = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[BLOCKSIZEINDEX_ND]; // 1: keyCache
        int64_t num_heads = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[BLOCKSIZEINDEX]; // 1: keyCache
        if (blockSize != inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[BLOCKSIZEINDEX_ND]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (num_heads != inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[BLOCKSIZEINDEX]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "numHeads should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (blockSize == ZERO) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSize cannot be zero";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (param_.isSeqLensCumsumMode) {
            if (lencontext != inTensorDescs.at(IN_TENSOR_3_CONTEXTLENS).shape.dims[0] - ONE) { // 1:-1
                ATB_LOG(ERROR) << GetLogPrefix() <<
                    "the lencontext of blocktable should match the lencontext of SeqLens when isSeqLensCumsumMode is true.";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else if (lencontext != inTensorDescs.at(IN_TENSOR_3_CONTEXTLENS).shape.dims[0]) {
                    ATB_LOG(ERROR) << GetLogPrefix() <<
                        "the lencontext of blocktable should match the lencontext of SeqLens when isSeqLensCumsumMode is true.";
                    return ERROR_INVALID_TENSOR_DIM;
        }
    } else { // NZ
        int64_t blockSize = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[BLOCKSIZEINDEX]; // 1: keyCache
        if (blockSize != inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[BLOCKSIZEINDEX]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSizes should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (blockSize == ZERO) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSize cannot be zero";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (blockSize % 16 != ZERO) {
            ATB_LOG(ERROR) << GetLogPrefix() << "blockSize must be aligned to 16";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (lencontext != inTensorDescs.at(IN_TENSOR_3_CONTEXTLENS).shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "lenscontextlens should be same";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return (param_.kvCacheCfg == infer::PagedCacheLoadParam::KvCacheCfg::K_CACHE_V_CACHE_NZ) ?
            KVCacheDimCheck910BNZ(inTensorDescs) : KVCacheDimCheck910BND(inTensorDescs);
}

Status PagedCacheLoadOperation::KVCacheDimCheck910BNZ(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dimNum != INPUTKEY_DIM ||     // 0: keyCache
        inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dimNum != INPUTVALUE_DIM || // 1: value Cache
        inTensorDescs.at(IN_TENSOR_2_BLOCKTABLE).shape.dimNum != INPUTBLOCK_DIM || // 2: dim=2
        inTensorDescs.at(IN_TENSOR_3_CONTEXTLENS).shape.dimNum != INPUTCONTEXTLENS_DIM || // 1: dim=1
        inTensorDescs.at(IN_TENSOR_4_KEY).shape.dimNum != INPUTBLOCK_DIM ||        // 2: dim=2
        inTensorDescs.at(IN_TENSOR_5_VALUE).shape.dimNum != INPUTBLOCK_DIM) {      // 2: dim=2
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).dtype == ACL_INT8) {
        if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[OUT_DIM] != THIRTYTWO ||
                inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[OUT_DIM] != THIRTYTWO) { // 1: valueCache
            ATB_LOG(ERROR) << GetLogPrefix() << "The last dimension of keycache and valuecache must be 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[1] * THIRTYTWO > MAX_k ||
                inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[1] * THIRTYTWO > MAX_v) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The scend dimension of blocktables must be less than 147456";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else  if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[OUT_DIM] != SIXTEEN ||
                inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[OUT_DIM] != SIXTEEN) { // 1: valueCache
            ATB_LOG(ERROR) << GetLogPrefix() << "The last dimension of keycache and valuecache must be 16";
            return ERROR_INVALID_TENSOR_DIM;
    } else if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[1] * SIXTEEN > MAX_k ||
                inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[1] * SIXTEEN > MAX_v) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The scend dimension of blocktables must be less than 147456";
            return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status PagedCacheLoadOperation::KVCacheDimCheck910BND(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dimNum != INPUTKEY_DIM || // 0: keyCache
        inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dimNum != INPUTVALUE_DIM || // 1: valueCache
        inTensorDescs.at(IN_TENSOR_2_BLOCKTABLE).shape.dimNum != INPUTBLOCK_DIM || // 2: blockTable
        inTensorDescs.at(IN_TENSOR_3_CONTEXTLENS).shape.dimNum != INPUTCONTEXTLENS_DIM || // 3: SeqLens
        inTensorDescs.at(IN_TENSOR_4_KEY).shape.dimNum != OUT_DIM || // 4:key
        inTensorDescs.at(IN_TENSOR_5_VALUE).shape.dimNum != OUT_DIM || // 5: value
        inTensorDescs.at(IN_TENSOR_6_SEQ_STARTS).shape.dimNum != INPUTCONTEXTLENS_DIM) { // 6: seq start
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid intensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t num_heads = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[NUM_HEADS_INDEX]; // 2: num heads
    int64_t head_size_k = inTensorDescs.at(IN_TENSOR_0_KEYCACHE).shape.dims[HEAD_SIZE_INDEX]; // 3: head_size_k
    int64_t head_size_v = inTensorDescs.at(IN_TENSOR_1_VALUECACHE).shape.dims[HEAD_SIZE_INDEX]; // 3: head_size_v
    if (inTensorDescs.at(IN_TENSOR_0_KEYCACHE).dtype == ACL_INT8) { // keyCache: int8
        if (num_heads * head_size_k % ALIGN_INT8 != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "int8 ND format num_heads*head_size_k should be aligned to 32!";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (num_heads * head_size_k % ALIGN_FP16_BF16 != 0) {  // keyCache: fp16/bf16
            ATB_LOG(ERROR) << GetLogPrefix() << "fp16/bf16 ND format num_heads*head_size_k should be aligned to 16!";
            return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_1_VALUECACHE).dtype == ACL_INT8) { // valueCache: int8
        if (num_heads * head_size_v % ALIGN_INT8 != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "int8 ND format num_heads*head_size_v should be aligned to 32!";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else if (num_heads * head_size_v % ALIGN_FP16_BF16 != 0) { // valueCache: fp16/bf16
            ATB_LOG(ERROR) << GetLogPrefix() << "fp16/bf16 ND format num_heads*head_size_v should be aligned to 16!";
            return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t lencontext = inTensorDescs.at(IN_TENSOR_2_BLOCKTABLE).shape.dims[0]; // 2: blockTable-batch(len(contextLens))
    int64_t seqstart = inTensorDescs.at(IN_TENSOR_6_SEQ_STARTS).shape.dims[0]; // 6: seq_start-batch(len(contextLens))
    if (param_.hasSeqStarts) {
        if (seqstart != lencontext) {
            ATB_LOG(ERROR) << GetLogPrefix() <<
                "the length of seq_startus should match lencontext when hasSeqStarts is true.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> PagedCacheLoadOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<PagedCacheLoadOpsRunner>(param_);
}
}