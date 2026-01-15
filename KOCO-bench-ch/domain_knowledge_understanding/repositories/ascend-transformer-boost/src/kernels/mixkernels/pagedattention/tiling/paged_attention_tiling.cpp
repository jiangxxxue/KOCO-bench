/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <numeric>
#include <algorithm>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/platform/platform_info.h"
#include "paged_attention_tiling.h"

namespace AtbOps {

const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM16 = 16;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const float SPLITKV_RATION = 0.8;
Status GetPagedAttentionNdInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                               OpParam::PagedAttention &param);
Status GetPagedAttentionNzInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo);
Status GetPagedAttentionInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                             OpParam::PagedAttention &param);
Status GetPagedAttentionMaskInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                                 OpParam::PagedAttention &param, bool isMLA);

void GetMaskStride(PagedAttentionInfo &mmInfo, OpParam::PagedAttention &param, Tensor &maskTensor)
{
    auto maskType = param.maskType;
    auto maskShape = maskTensor.desc.dims;
    auto maskDim = maskShape.size();
    auto cs = param.compressHead;
    if (maskType == OpParam::PagedAttention::MASK_TYPE_ALIBI) {
        if (maskDim == DIM_3) {
            mmInfo.maxPromptLen = static_cast<int32_t>(maskShape.at(DIM_2));  // 3 position (h, 1, max_s)
            mmInfo.batchStride = cs ? mmInfo.maxPromptLen * maskShape.at(DIM_0) / param.kvHead : 0;
            mmInfo.headStride = mmInfo.maxPromptLen;
            mmInfo.modCoef = cs ? param.kvHead : -1;
        } else if (maskDim == DIM_4) {
            mmInfo.maxPromptLen = static_cast<int32_t>(maskShape.at(DIM_3));  // 4 position (b, 1/h, 1, max_s)
            auto maskHead = maskShape.at(DIM_1);
            if (maskHead == 1) {
                mmInfo.batchStride = mmInfo.maxPromptLen;
                mmInfo.divCoef = cs ? param.kvHead : 1;
                mmInfo.headStride = 0;
            } else {
                mmInfo.batchStride = mmInfo.maxPromptLen * maskShape.at(DIM_1) / (cs ? param.kvHead : 1);
                mmInfo.headStride = mmInfo.maxPromptLen;
            }
        }
    } else if (maskType == OpParam::PagedAttention::MASK_TYPE_NORM) {
        if (maskDim == DIM_2) {
            mmInfo.maxPromptLen = static_cast<int32_t>(maskShape.at(DIM_1));  // 2 position, (1, max_s)
            if (maskShape.at(DIM_1) == maskShape.at(DIM_0)) {
                mmInfo.isMaskSquare = 1;
            }
            mmInfo.batchStride = 0;
            mmInfo.headStride = 0;
        } else if (maskDim == DIM_3) {
            mmInfo.maxPromptLen = static_cast<int32_t>(maskShape.at(DIM_2));  // 3 position, (b, 1, max_s)
            mmInfo.batchStride = mmInfo.maxPromptLen;
            mmInfo.headStride = 0;
            mmInfo.divCoef = cs ? param.kvHead : 1;
        }
    } else if (maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD  ||
                    maskType == OpParam::PagedAttention::MASK_TYPE_MASK_FREE) {
        mmInfo.maxPromptLen = static_cast<int32_t>(maskShape.at(DIM_1));
        mmInfo.batchStride = 0;
        mmInfo.headStride = 0;
    }
}

Status GetTilingKeyTypeBase(PagedAttentionInfo &mmInfo, const Tensor &qTensor)
{
    if (qTensor.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.type = TilingKeyType::TILING_BF16_DATA;
    } else {
        mmInfo.type = TilingKeyType::TILING_HALF_DATA;
    }
    return Status::OkStatus();
}

Status GetTilingKeyTypeDequant(PagedAttentionInfo &mmInfo,
                               const Tensor &qTensor, const Tensor &deqScale1, bool isMLA)
{
    if ((qTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) && (deqScale1.desc.dtype == TENSOR_DTYPE_FLOAT)) {
        mmInfo.type = TilingKeyType::TILING_INT8_VEC_QUANT;
    } else if ((qTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) && (deqScale1.desc.dtype != TENSOR_DTYPE_FLOAT)) {
        mmInfo.type = TilingKeyType::TILING_INT8_CUBE_QUANT;
    } else if ((qTensor.desc.dtype == TENSOR_DTYPE_BF16) && (deqScale1.desc.dtype == TENSOR_DTYPE_FLOAT)) {
        mmInfo.type = TilingKeyType::TILING_INT8_VEC_QUANTBF16;
    }
    MKI_CHECK(!isMLA, "MLA does not support antiquant for now",
            return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status GetTilingKeyTypeQuant(PagedAttentionInfo &mmInfo, const Tensor &outTensor)
{
    if (outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.type = TilingKeyType::TILING_QUANT_FP16OUT;
        mmInfo.tBlockAlign = BLOCK_SIZE_32;
    } else if (outTensor.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.type = TilingKeyType::TILING_QUANT_BF16OUT;
        mmInfo.tBlockAlign = BLOCK_SIZE_32;
    }
    return Status::OkStatus();
}

Status GetTilingKeyTypeOmni(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo, OpParam::PagedAttention &param,
                            const Tensor &qTensor, const Tensor &kTensor, const Tensor &outTensor, bool isMLA)
{
    auto deqScale1 = param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND ?
                     launchParam.GetInTensor(DIM_4) : launchParam.GetInTensor(DIM_5);
    if (qTensor.desc.dtype == TENSOR_DTYPE_INT8) {
        OP_TILING_CHECK_STATUS_RETURN(GetTilingKeyTypeQuant(mmInfo, outTensor));
    } else if (kTensor.desc.dtype == TENSOR_DTYPE_INT8) {
        OP_TILING_CHECK_STATUS_RETURN(GetTilingKeyTypeDequant(mmInfo, qTensor, deqScale1, isMLA));
    } else {
        OP_TILING_CHECK_STATUS_RETURN(GetTilingKeyTypeBase(mmInfo, qTensor));
    }
    return Status::OkStatus();
}

Status GetPagedAttentionMaskInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                                 OpParam::PagedAttention &param, bool isMLA)
{
    Tensor maskTensor = param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ?
                        launchParam.GetInTensor(DIM_4) : launchParam.GetInTensor(DIM_3);
    if (!CheckEmptyTensor(maskTensor)) {
        GetMaskStride(mmInfo, param, maskTensor);
    }
    auto kTensor = launchParam.GetInTensor(DIM_1);
    auto qTensor = launchParam.GetInTensor(DIM_0);
    auto outTensor = launchParam.GetOutTensor(0);
    if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        OP_TILING_CHECK_STATUS_RETURN(GetTilingKeyTypeBase(mmInfo, qTensor));
    } else {
        OP_TILING_CHECK_STATUS_RETURN(GetTilingKeyTypeOmni(launchParam, mmInfo, param,
                                                           qTensor, kTensor, outTensor, isMLA));
    }
    MKI_LOG(INFO) << "maxPromptLen is: " << mmInfo.maxPromptLen << " batchStride is: " << mmInfo.batchStride
                  << " headStride is: " << mmInfo.headStride << " type is: " << static_cast<int32_t>(mmInfo.type);
    return Status::OkStatus();
}

Status CheckPagedAttentionNdEnhance(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                                    OpParam::PagedAttention &param)
{
    auto qTensor = launchParam.GetInTensor(DIM_0);
    auto kTensor = launchParam.GetInTensor(DIM_1);
    if ((qTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || qTensor.desc.dtype == TENSOR_DTYPE_BF16) &&
        (kTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || kTensor.desc.dtype == TENSOR_DTYPE_BF16)) {
        if (mmInfo.blockSize == NUM256 && (mmInfo.numHeads == NUM32 || mmInfo.numHeads == NUM16) &&
            mmInfo.kvHeads == NUM1 && mmInfo.embeddingSize == NUM576 &&
            mmInfo.embeddingSizeV == NUM512) {
            return Status::OkStatus();
        }
    }

    MKI_CHECK(mmInfo.embeddingSize > 0 && mmInfo.embeddingSize <= MAX_EMBEDDING,
              "embedding size is invalid, embedding size QK should not exceed 576",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.embeddingSizeV > 0 && mmInfo.embeddingSizeV <= MAX_EMBEDDING,
              "embedding size is invalid, embedding size V should not exceed 576",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    if (mmInfo.embeddingSize > MLA_THRESHOLD || mmInfo.embeddingSizeV > MLA_THRESHOLD) {
        MKI_CHECK(mmInfo.blockSize <= MLA_BLOCK_SIZE_LIMIT,
                "block size should not exceed 128 when embedDims exceed 256",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    } else if ((mmInfo.embeddingSize == MLA_THRESHOLD && mmInfo.embeddingSizeV == MLA_THRESHOLD)
        && mmInfo.blockSize >= MLA_BLOCK_SIZE_LIMIT) {
        MKI_CHECK(mmInfo.blockSize % MLA_BLOCK_SIZE_LIMIT == 0,
        "when embedding size equal to 256, blocksize should be 128-aligned",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    } else if (mmInfo.embeddingSize == mmInfo.embeddingSizeV) {
        MKI_CHECK(mmInfo.embeddingSize * mmInfo.blockSize <= BLOCK_LIMIT &&
                      mmInfo.embeddingSizeV * mmInfo.blockSize <= BLOCK_LIMIT,
                  "embeddingSize * blockSize should not exceed 128 * 128",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    return Status::OkStatus();
}

Status GetPagedAttentionNdInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                               OpParam::PagedAttention &param)
{
    auto qShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto kcacheShape = launchParam.GetInTensor(DIM_1).desc.dims;
    auto tableShape = param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ?
                      launchParam.GetInTensor(DIM_3).desc.dims : launchParam.GetInTensor(DIM_2).desc.dims;
    mmInfo.embeddingSize = static_cast<int32_t>(kcacheShape.at(DIM_3));
    mmInfo.embeddingSizeV = param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ?
                            launchParam.GetInTensor(DIM_2).desc.dims.at(DIM_3) : param.headDimV;
    mmInfo.numBlocks = static_cast<int32_t>(kcacheShape.at(DIM_0));
    mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_1));
    if (param.dataShapeType == 1) {
        mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_2));
    }

    mmInfo.maxNumBlocksPerQuery = static_cast<int32_t>(tableShape.at(DIM_1));
    mmInfo.tor = param.tor;
    mmInfo.kvSeqLen = param.kvSeqLen.data();
    mmInfo.qSeqLen = param.qSeqLen.data();
    param.kvHead = param.kvHead <= 0 ? param.headSize : param.kvHead;
    mmInfo.batch = static_cast<int32_t>(param.kvSeqLen.size());
    mmInfo.qHeadOriginal = param.headSize;
    if (param.compressHead) {
        mmInfo.kvHeads = 1;
        mmInfo.numTokens = static_cast<int32_t>(qShape.at(DIM_0) * param.kvHead);
        mmInfo.numHeads = static_cast<int32_t>(param.headSize / param.kvHead);
        mmInfo.compressHead = 1;
    } else {
        mmInfo.kvHeads = param.kvHead;
        mmInfo.numTokens = static_cast<int32_t>(qShape.at(DIM_0));
        mmInfo.numHeads = static_cast<int32_t>(param.headSize);
    }
    bool isMLA = (mmInfo.embeddingSize > 256 || mmInfo.embeddingSizeV > 256 ||
                      mmInfo.embeddingSize != mmInfo.embeddingSizeV);
    mmInfo.dataShapeType = param.dataShapeType;
    OP_TILING_CHECK_STATUS_RETURN(CheckPagedAttentionNdEnhance(launchParam, mmInfo, param));
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        OP_TILING_CHECK_STATUS_RETURN(GetPagedAttentionMaskInfo(launchParam, mmInfo, param, isMLA));
    }

    // Check tiling data
    MKI_LOG(INFO) << "numTokens is: " << mmInfo.numTokens << " numHeads is: " << mmInfo.numHeads
                  << "batch is: " << mmInfo.batch << " embeddingSize is: " << mmInfo.embeddingSize
                  << " numBlocks is: " << mmInfo.numBlocks << " blockSize is: " << mmInfo.blockSize
                  << " maxNumBlocksPerQuery: " << mmInfo.maxNumBlocksPerQuery << " kv_heads: " << mmInfo.kvHeads
                  << " compressHead: " << mmInfo.compressHead  << " dataShapeType: "<< mmInfo.dataShapeType;
    return Status::OkStatus();
}

Status GetPagedAttentionNzMaskInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo,
                                   OpParam::PagedAttention::MaskType maskType)
{
    if (maskType != OpParam::PagedAttention::MASK_TYPE_NONE) {
        auto maskShape = launchParam.GetInTensor(DIM_5).desc.dims;
        auto maxPromptLen = maskShape.at(DIM_1) * BLOCK_SIZE;
        mmInfo.maxPromptLen = maxPromptLen;
        auto maskDimZero = maskShape.at(DIM_0);
        switch (maskType) {
            case OpParam::PagedAttention::MASK_TYPE_ALIBI:
                mmInfo.batchStride = (maskDimZero == mmInfo.numHeads) ? 0 : mmInfo.numHeads * BLOCK_SIZE * maxPromptLen;
                mmInfo.headStride = BLOCK_SIZE * maxPromptLen;
                break;
            case OpParam::PagedAttention::MASK_TYPE_NORM:
                mmInfo.headStride = 0;
                mmInfo.batchStride = (maskDimZero == static_cast<int64_t>(mmInfo.batch)) ?
                                          maskShape.at(DIM_2) * maxPromptLen : 0;
                break;
            case OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD:
                break;
            case OpParam::PagedAttention::MASK_TYPE_MASK_FREE:
                break;
            default:
                return Status::FailStatus(ERROR_INVALID_VALUE, "Invalid MaskType");
        }
    }
    return Status::OkStatus();
}

Status CheckQSeqlenValid(const std::vector<int32_t>& qSeqlen)
{
    int64_t numTokens = 0;
    for (size_t idx = 0; idx < qSeqlen.size(); ++idx) {
        int64_t curQSeqlen = static_cast<int64_t>(qSeqlen.at(idx));
        MKI_CHECK(curQSeqlen > 0, "seqlen is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        numTokens += curQSeqlen;
        MKI_CHECK(numTokens <= INT32_MAX, "seqlen is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    return Status::OkStatus();
}

Status GetPagedAttentionNzInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo)
{
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::PagedAttention), "OpParam is invalid",
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    auto &param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
    auto qShape = launchParam.GetInTensor(0).desc.dims;
    auto kcacheShape = launchParam.GetInTensor(1).desc.dims;
    auto blockTableShape = launchParam.GetInTensor(3).desc.dims;
    auto blockLens = launchParam.GetInTensor(4).desc.dims;
    auto head =  param.headSize;
    auto maskType = param.maskType;
    Status ret = Status::OkStatus();
    if (param.qSeqLen.size() > 0) {
        ret = CheckQSeqlenValid(param.qSeqLen);
    }
    if (!ret.Ok()) {
        return ret;
    }
    mmInfo.numTokens = (param.qSeqLen.size() > 0) ? std::accumulate(param.qSeqLen.begin(), param.qSeqLen.end(), 0) :
                                                    static_cast<int32_t>(blockLens.at(0));
    mmInfo.batch = (param.qSeqLen.size() > 0) ? static_cast<int32_t>(param.qSeqLen.size()) : mmInfo.numTokens;
    if (param.qSeqLen.size() > 0) {
        mmInfo.qSeqLen = const_cast<int32_t*>(param.qSeqLen.data());
    }
    mmInfo.numHeads = head;
    mmInfo.embeddingSize = qShape.at(1) * NZ_BLOCK_SIZE / param.headSize;
    mmInfo.numBlocks = static_cast<int32_t>(kcacheShape.at(0));
    mmInfo.blockSize = static_cast<int32_t>(kcacheShape.at(DIM_2));
    MKI_CHECK(mmInfo.embeddingSize * mmInfo.blockSize <= BLOCK_LIMIT,
                 "embeddingSize * blockSize should no greater than 128 * 128",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    mmInfo.maxNumBlocksPerQuery = blockTableShape.at(1);
    mmInfo.kvHeads = param.kvHead;
    mmInfo.tor = param.tor;
    mmInfo.maxPromptLen = 0;
    ret = GetPagedAttentionNzMaskInfo(launchParam, mmInfo, maskType);
    if (!ret.Ok()) {
        return ret;
    }
    MKI_CHECK(mmInfo.embeddingSize <= MAX_EMBEDDING && mmInfo.embeddingSize > 0, "headdim is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_LOG(INFO) << "headStride is: " << mmInfo.headStride << " batchStride is: " << mmInfo.batchStride;
    return Status::OkStatus();
}

Status GetPagedAttentionInfo(const LaunchParam &launchParam, PagedAttentionInfo &mmInfo, OpParam::PagedAttention &param)
{
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) {
        OP_TILING_CHECK_STATUS_RETURN(GetPagedAttentionNzInfo(launchParam, mmInfo));
    } else if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ||
               param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND ||
               param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        OP_TILING_CHECK_STATUS_RETURN(GetPagedAttentionNdInfo(launchParam, mmInfo, param));
    } else {
        OP_TILING_CHECK_STATUS_RETURN(Status::FailStatus(ERROR_INVALID_VALUE, "param.type is invalid"));
    }
    MKI_CHECK(mmInfo.blockSize > 0 && mmInfo.blockSize % BLOCK_SIZE == 0, "blockSize is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status GetPaParallelCheck(const uint32_t *tilingParam, const OpParam::PagedAttention param,
                          const PagedAttentionInfo &mmInfo)
{
    bool isParallel = (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) ?
                      (tilingParam[PREFILL_BATCH] > 0) : false;
    isParallel = (param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) ?
                 (mmInfo.numTokens != mmInfo.batch) : isParallel;
    bool isLookahead = param.maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD;
    bool isPrefix = param.maskType == OpParam::PagedAttention::MASK_TYPE_MASK_FREE;

    // parallel support check
    if (isParallel || isLookahead || isPrefix) {
        MKI_CHECK(mmInfo.batch > 0 && mmInfo.batch <= PARALLEL_MAX_BATCH,
                "parallel batch size invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.numHeads > 0 && mmInfo.numHeads <= PARALLEL_MAX_HEAD,
                "parallel headnum invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.blockSize > 0 && mmInfo.blockSize <= PARALLEL_MAX_BLK_SIZE,
                "parallel block size invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        // embedding * blksize <= 128*128
        if (param.type != OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
            MKI_CHECK(mmInfo.embeddingSize * mmInfo.numHeads <= BLOCK_LIMIT,
            "parallel block and embedding invalid",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        MKI_CHECK(mmInfo.dataShapeType == 0,
                "parallel decoding not support BNSD input",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.type == TilingKeyType::TILING_HALF_DATA
                || mmInfo.type == TilingKeyType::TILING_BF16_DATA,
                "parallel decoding only support fp16 or bf16 data type",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) {
            MKI_CHECK(param.maskType == OpParam::PagedAttention::MASK_TYPE_NORM ||
                    param.maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD,
                    "prefill pa only support triu mask and lookahead mask",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(mmInfo.embeddingSize == mmInfo.embeddingSizeV,
                "parallel decoding not support MLA",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        } else if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) {
            MKI_CHECK(param.maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD ||
                param.maskType == OpParam::PagedAttention::MASK_TYPE_NONE ||
                param.maskType == OpParam::PagedAttention::MASK_TYPE_MASK_FREE,
                "prefill pa only support no mask and lookahead mask",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(mmInfo.embeddingSize > 0 && mmInfo.embeddingSizeV == 0,
                "parallel decoding not support MLA",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
        }
    }
    return Status::OkStatus();
}

Status PagedAttentionTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
    PagedAttentionInfo mmInfo = {0};
    Status ret1  = GetPagedAttentionInfo(launchParam, mmInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret1);
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(tilingHost);
    Status ret = GetPagedAttentionTilingParam(launchParam, mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    ret1 = GetPaParallelCheck(tilingParam, param, mmInfo);
    OP_TILING_CHECK_STATUS_RETURN(ret1);
    uint32_t dataLenHalf = sizeof(uint16_t);
    uint32_t dataLenFloat = sizeof(float);
    uint32_t oCoreTempSize = 0;
    uint32_t lCoreTempSize = 0;
    uint32_t k16TempSize = 0;
    uint32_t v16TempSize = 0;
    uint64_t basicWorkSpaceHalf = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenHalf;
    uint64_t basicWorkSpaceFloat = blockDim * WORKSPACE_BLOCK_SIZE_DB * dataLenFloat;
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) {
        oCoreTempSize = blockDim * SPLITKV_RATION * static_cast<uint32_t>(mmInfo.numHeads) *
            blockDim * static_cast<uint32_t>(mmInfo.embeddingSize * sizeof(float));
        // (num_tokens, num_heads, kvsplit)
        lCoreTempSize = blockDim * SPLITKV_RATION * static_cast<uint32_t>(mmInfo.numHeads) *
            blockDim * sizeof(float);
        k16TempSize = NUM2 * blockDim * NUM256 * static_cast<uint32_t>(mmInfo.numHeads) *
                               static_cast<uint32_t>(mmInfo.embeddingSize) * sizeof(uint16_t);
        v16TempSize = NUM2 * blockDim * NUM256 * static_cast<uint32_t>(mmInfo.numHeads) *
                               static_cast<uint32_t>(mmInfo.embeddingSize) * sizeof(uint16_t);
        kernelInfo.GetScratchSizes() = {basicWorkSpaceFloat, basicWorkSpaceHalf, basicWorkSpaceFloat * 2,
                                        basicWorkSpaceFloat, oCoreTempSize, lCoreTempSize,
                                        k16TempSize, v16TempSize, };
    } else if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND) {
        kernelInfo.GetScratchSizes() = {basicWorkSpaceFloat, basicWorkSpaceHalf, basicWorkSpaceFloat * 2,
                                        basicWorkSpaceFloat, oCoreTempSize, lCoreTempSize,
                                        k16TempSize, v16TempSize, };
    } else if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        kernelInfo.GetScratchSizes() = {basicWorkSpaceFloat * 4, basicWorkSpaceHalf * 4,
                                        basicWorkSpaceFloat * 24, basicWorkSpaceFloat * 4, };
    }
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        kernelInfo.SetTilingId(tilingParam[TILING_KEY_ID]);
    }
    kernelInfo.SetBlockDim(blockDim);
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << " block dim = " << blockDim;
    return Status::OkStatus();
}

} // namespace AtbOps
