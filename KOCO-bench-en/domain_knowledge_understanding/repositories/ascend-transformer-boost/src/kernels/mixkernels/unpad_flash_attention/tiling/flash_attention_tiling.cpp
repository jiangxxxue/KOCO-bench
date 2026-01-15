/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include "flash_attention_tiling.h"

#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/platform/platform_info.h>

#define UNUSED_VALUE(x) (void)(x)

namespace AtbOps {
using namespace Mki;
Status GetFlashAttentionNoCacheMaskInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param,
                                        uint32_t batch);

Status InitInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param);

void InitMaxKVSeqlenBNSD(Mki::SVector<int64_t> &kcacheShape, AtbOps::UnpadFlashAttentionInfo &mmInfo, size_t &batch);

void InitEmbed(int32_t &embed, const UnpadFlashAttentionInfo &mmInfo,
               const AtbOps::OpParam::UnpadFlashAttention &param);

Status InitMaxKVSeqlen(size_t &batch, Mki::SVector<int64_t> &kcacheShape,
                       UnpadFlashAttentionInfo &mmInfo, AtbOps::OpParam::UnpadFlashAttention &param);

inline void InitGetInputTensors(UnpadFlashAttentionInfo &mmInfo, const LaunchParam &launchParam,
                                const AtbOps::OpParam::UnpadFlashAttention &param)
{
    uint32_t index = 0;
    switch (param.type) {
        case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE:
        case OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE:
            mmInfo.tensors.query = launchParam.GetInTensor(index++);
            mmInfo.tensors.kCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.vCache = mmInfo.tensors.kCache;
            mmInfo.tensors.layerId = launchParam.GetInTensor(index++);
            mmInfo.tensors.mask = launchParam.GetInTensor(index++);
            break;
        case OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND:
            mmInfo.tensors.query = launchParam.GetInTensor(index++);
            mmInfo.tensors.mask = launchParam.GetInTensor(index++);
            break;
        case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND:
            mmInfo.tensors.query = launchParam.GetInTensor(index++);
            mmInfo.tensors.kCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.vCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.blockTable = launchParam.GetInTensor(index++);
            mmInfo.tensors.mask = launchParam.GetInTensor(index++);
            mmInfo.tensors.alibiCoeff = launchParam.GetInTensor(index++);
            break;
        case OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION:
            mmInfo.tensors.query = launchParam.GetInTensor(index++);
            mmInfo.tensors.kCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.vCache = launchParam.GetInTensor(index++);
            break;
        default:
            mmInfo.tensors.query = launchParam.GetInTensor(index++);
            mmInfo.tensors.kCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.vCache = launchParam.GetInTensor(index++);
            mmInfo.tensors.layerId = launchParam.GetInTensor(index++);
            mmInfo.tensors.mask = launchParam.GetInTensor(index++);
            mmInfo.tensors.alibiCoeff = launchParam.GetInTensor(index++);
            break;
    }
}


void GetSwaTilingKey(UnpadFlashAttentionInfo &mmInfo)
{
    if (mmInfo.windowSize > 0 &&
        mmInfo.type != OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND
        && mmInfo.type != OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER) {
        mmInfo.tilingKey = static_cast<uint32_t>(TilingKeyType::TILING_SWA);
        if (mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS) {
            mmInfo.tilingKey = static_cast<uint32_t>(TilingKeyType::TILING_SWA_CMP);
        }
    }
}

void GetKernelTilingId(const UnpadFlashAttentionInfo &mmInfo, KernelInfo &kernelInfo, uint32_t *tilingParam)
{
    if (mmInfo.type != OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND
        && mmInfo.type != OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER
        && mmInfo.type != OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        kernelInfo.SetTilingId(tilingParam[TILING_KEY_INDEX]);
    }
}


Status FlashAttentionTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    UnpadFlashAttentionInfo mmInfo;
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttention), "OpParam is invalid",
              return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    OpParam::UnpadFlashAttention param = AnyCast<OpParam::UnpadFlashAttention>(launchParam.GetParam());
    InitGetInputTensors(mmInfo, launchParam, param);
    Status ret = InitInfo(mmInfo, param);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    std::vector<int32_t> batchRunStatusDefault;
    if (param.batchRunStatus.size() != static_cast<size_t>(mmInfo.batchSize)) {
        MKI_LOG(INFO) << "size of batchRunStatus is not equal batch, will be resize to (batch, 1) ";
        batchRunStatusDefault.resize(mmInfo.batchSize, 1);
        mmInfo.batchRunStatus = batchRunStatusDefault.data();
    }
    MKI_CHECK(mmInfo.batchRunStatus != nullptr, "batchRunStatus cannot be nullptr",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    MKI_CHECK(blockDim > 0, "blockDim cannot <= 0", return Status::FailStatus(ERROR_INVALID_VALUE));
    mmInfo.blockDim = blockDim;
    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(tilingHost);
    ret = GetUnpadFlashAttentionTilingParam(mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);

    uint64_t dataLenFloat = sizeof(float);
    uint64_t workSize = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(DOUBLE_PING_PONG_SIZE) * dataLenFloat;
    if (mmInfo.splitm) {
        workSize = static_cast<uint64_t>(blockDim) * static_cast<uint64_t>(SPLITM_DOUBLE_PING_PONG_SIZE) * dataLenFloat;
    }

    if (param.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        MKI_CHECK(mmInfo.batchSize <= MAX_BATCH, "batch should not exceed 60", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.kvHead <= RELAY_BLOCK_TILING, "kvhead should not exceed 8", return Status::FailStatus(ERROR_INVALID_VALUE));
        uint64_t glWorkSize = static_cast<uint64_t>(mmInfo.batchSize) *
                              static_cast<uint64_t>(mmInfo.innerBatchSize) * 4 * 2;
        uint64_t goWorkSize = static_cast<uint64_t>(mmInfo.batchSize) * static_cast<uint64_t>(mmInfo.innerBatchSize) *
                              mmInfo.embeddingSize * 4 * 2;
        kernelInfo.GetScratchSizes() = {workSize, workSize, workSize, goWorkSize, glWorkSize};
        blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    } else {
        kernelInfo.GetScratchSizes() = {workSize, workSize, workSize * 6, workSize}; // oTmp/S/P
    }
    if (param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE ||
        param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE) {
        kernelInfo.SetTilingId(tilingParam[TILING_KEY_INDEX]);
    }
    kernelInfo.SetBlockDim(blockDim);
    GetKernelTilingId(mmInfo, kernelInfo, tilingParam);
    return Status::OkStatus();
}

Status GetAlibiMaskInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param, const Tensor &tensorMask,
                        int32_t maxSeq)
{
    auto maskShape = tensorMask.desc.dims;
    auto maskDim = maskShape.size();
    MKI_CHECK(maskDim <= DIM_4 && maskDim >= DIM_2, "maskdim invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    if (maskDim == DIM_3) { // [h, ms, ms]
        mmInfo.maskStride = 0;
        mmInfo.headStride = static_cast<uint32_t>(maxSeq);
    } else if (maskDim == DIM_4) { // [bs,1,ms,ms]  [bs,headnum,ms,ms]
        MKI_CHECK(static_cast<uint64_t>(maskShape.at(DIM_2)) * maskShape.at(1) <= UINT32_MAX, "maxSeq * headnum can not large than UINT32_MAX",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.maskStride = maskShape.at(1) * maskShape.at(DIM_2);
        mmInfo.headStride = static_cast<uint32_t>(maxSeq);
    } else if (maskDim == DIM_2) {
        MKI_CHECK(maxSeq == LONG_SEQ_ALIBI_LEN, "alibi mask shape must be [256, 256] for long seq opt",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.isAlibiMaskSqrt = param.isAlibiMaskSqrt;
    }
    MKI_CHECK(static_cast<uint64_t>(mmInfo.headStride) * static_cast<uint32_t>(maxSeq) <= UINT32_MAX,
              "maxSeq can not exceed UINT32_MAX", return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status GetSwaMaskInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param, const Tensor &tensorMask)
{
    auto maxQSeq = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
    if (*maxQSeq != 1) {
        auto maskShape = tensorMask.desc.dims;
        auto maskDim = maskShape.size();
        auto maskType = param.maskType;
        int32_t maxSeq = maskShape.at(maskDim - 1);
        mmInfo.maxSeqLen = maxSeq;
        MKI_CHECK(maskDim == DIM_2, "maskdim invalid",
                return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(maskShape.at(0) <= maskShape.at(1),
                    "mask shape invalid, maxkv should be larger than maxq",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        if (maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS) {
            MKI_CHECK(maskShape.at(1) == SWA_COMPRESS_MASK_SIZE,
                    "mask shape invalid, swa compress mask shape should be 512,512",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(maskShape.at(0) == SWA_COMPRESS_MASK_SIZE,
                    "mask shape invalid, swa compress mask shape should be 512,512",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        }
    }
    return Status::OkStatus();
}

Status GetFlashAttentionNoCacheMaskInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param,
                                        const Tensor &tensorMask)
{
    auto &tensorAlibiCoeff = mmInfo.tensors.alibiCoeff;
    auto maskShape = tensorMask.desc.dims;
    auto maskType = param.maskType;
    auto maxQSeq = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
    auto maxKvSeq = std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    MKI_LOG(INFO) << "max kv seq" << *maxKvSeq;
    if (maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS ||
        maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_NORM) {
        auto ret = GetSwaMaskInfo(mmInfo, param, tensorMask);
        OP_TILING_CHECK_STATUS_RETURN(ret);
    } else if (maskType != OpParam::UnpadFlashAttention::MASK_TYPE_NONE
        && maskType != OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK) {
        auto maskDim = maskShape.size();
        int32_t maxSeq = maskShape.at(maskDim - 1);
        mmInfo.maxSeqLen = maxSeq;
        if (maxSeq == LONG_SEQ_LEN && maskShape.at(maskDim - DIM_2) != maxSeq) {
            mmInfo.alibiCompressOffset = static_cast<uint32_t>(maskShape.at(maskDim - DIM_2));
        }
        if (!(maxSeq == LONG_SEQ_LEN && param.isTriuMask != 0) && CheckEmptyTensor(tensorAlibiCoeff)) {
            MKI_CHECK(*maxQSeq <= maxSeq, "qSeqLen larger than maxSeqLen",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(*maxKvSeq <= maxSeq, "kvSeqLen larger than maxSeqLen",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
        }
        if (maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI
            || maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS
            || maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_SQRT
            || maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN
            || maskType == OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI_COMPRESS_128) {
            auto ret = GetAlibiMaskInfo(mmInfo, param, tensorMask, maxSeq);
            OP_TILING_CHECK_STATUS_RETURN(ret);
        } else {
            MKI_CHECK(maskDim <= DIM_3 && maskDim >= DIM_2, "maskdim invalid",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            MKI_CHECK(maskShape.at(1) <= UINT32_MAX, "maxShape can not exceed UINT32_MAX",
                      return Status::FailStatus(ERROR_INVALID_VALUE));
            if (maskDim == DIM_2) { // [ms,ms]
                mmInfo.maskStride = 0;
                mmInfo.headStride = 0;
            } else if (maskDim == DIM_3) { // [bs,ms,ms]
                mmInfo.maskStride = maskShape.at(1);
                mmInfo.headStride = 0;
            }
            mmInfo.isLongSeq = (maxSeq == LONG_SEQ_LEN && param.isTriuMask != 0) ? 1 : 0;
        }
    }
    return Status::OkStatus();
}

inline void FlashAttentionLog(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    MKI_LOG(INFO) << "batch is: " << mmInfo.batchSize << " kvMaxSeq is: " << mmInfo.maxKvSeqLen
                  << " maskMaxSeq is: " << mmInfo.maxSeqLen << " head is: " << mmInfo.innerBatchSize
                  << " qSeq  is: " << *(mmInfo.qSeq) << " kvSeq is: " << *(mmInfo.kvSeq) << " tor is : " << mmInfo.tor
                  << " kv_head is: " << mmInfo.kvHead << " embed is: " << mmInfo.embeddingSize
                  << " maskStrid is: " << mmInfo.maskStride << " headStride is: " << mmInfo.headStride << " isTriuMask "
                  << mmInfo.isTriuMask << " mix type: " << mmInfo.type << " is clamp is:" << mmInfo.isClamp
                  << " clamp min is " << mmInfo.clampMin << " clamp max is " << mmInfo.clampMax << " maskType is "
                  << param.maskType << " longseq is " << mmInfo.isLongSeq << " Alibi sqrt is " << mmInfo.isAlibiMaskSqrt
                  << " mask type is " << mmInfo.maskType << " Alibi compress offset is " << mmInfo.alibiCompressOffset
                  << " dataShapeType is " << mmInfo.dataShapeType << " alibiLeftAlign is " << mmInfo.alibiLeftAlign
                  << " blockSize is " << mmInfo.blockSize << " maxBlockSize is " << mmInfo.maxNumBlocks
                  << " sliding window size is " << mmInfo.windowSize << " CacheType is" << mmInfo.cacheType;
}

inline Status FlashAttentionPreCheck(OpParam::UnpadFlashAttention &param)
{
    MKI_CHECK(param.headSize > 0, "headSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((param.kvHead > 0 && param.kvHead <= param.headSize && param.headSize % param.kvHead == 0) ||
                  (param.kvHead == 0),
              "kvHead is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.size() == param.kvSeqLen.size(), "qSeqLen size Not equal to kvSeqLen",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.data() != nullptr, "qSeq cannot be nullptr",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.kvSeqLen.data() != nullptr, "kvSeq cannot be nullptr",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.kTensorList.size() == param.vTensorList.size(), "k and v cache have different batch number",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline Status FlashAttentionPostCheck(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    MKI_CHECK(mmInfo.batchSize > 0 && mmInfo.batchSize <= ND_BATCH_LIMIT, "batch is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(static_cast<int32_t>(param.qSeqLen.size()) == mmInfo.batchSize, "SeqLen size is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.isTriuMask == 0 || mmInfo.isTriuMask == 1, "param isTriuMask is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.isLongSeq == 0 || mmInfo.isLongSeq == 1, "param isLongSeq is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.isAlibiMaskSqrt == 0 || mmInfo.isAlibiMaskSqrt == 1, "param isAlibiMaskSqrt is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.cacheType <= OpParam::UnpadFlashAttention::CACHE_TYPE_CYCLE, "param cacheType is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    if (mmInfo.isMLA) {
        MKI_CHECK(mmInfo.embeddingSize <= MAX_EMBEDDING && mmInfo.embeddingSize > 0, "headdimQ is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.embeddingSizeV <= MAX_EMBEDDING && mmInfo.embeddingSizeV > 0, "headdimV is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        MKI_CHECK(mmInfo.embeddingSize <= MLA_THRESHOLD && mmInfo.embeddingSize > 0, "headdim is invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    MKI_CHECK((mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_NORM ||
            mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS) == (mmInfo.windowSize > 0),
            "swa window size should be greater than 0", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.batchContinuous != (mmInfo.kTensorList.size() != 0),
              "invalid kv-batchwise setting, is kv batchContinuous or not?",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline Status FlashAttentionSwaCheck(const UnpadFlashAttentionInfo mmInfo)
{
    MKI_CHECK(mmInfo.dataShapeType == 0, "swa not support BNSD input",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.quantType == 0, "swa not support quant",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.isClamp == 0, "swa not support clamp",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.scaleType == OpParam::UnpadFlashAttention::SCALE_TOR,
                "swa not support logN", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.tensors.query.desc.dtype != TENSOR_DTYPE_INT8,
                "swa not support qkv int8 input",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND ||
                mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND ||
                mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND,
                "swa only support encoder nd and decoder nd type",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline void GetMLATilingKey(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    uint32_t embeddingKey = (mmInfo.embeddingSize <= MLA_THRESHOLD) ? 1 : 0;
    uint32_t int8Key = (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_INT8) ? 1 : 0;
    uint32_t typeKey = (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16) ? 1 : 0;
    if (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_INT8) {
        typeKey = (param.outDataType == TENSOR_DTYPE_BF16) ? 1 : 0;
    }
    uint32_t highPKey = (param.type ==
        OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE) ? 1 : 0;
    highPKey = (typeKey == 1) ? 1 : highPKey;
    mmInfo.tilingKey = (highPKey << DIM_3) + (typeKey << DIM_2) + (int8Key << DIM_1) + embeddingKey;
}

void GetKvCacheTilingKey(UnpadFlashAttentionInfo &mmInfo)
{
    uint32_t int8Key = (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_INT8) ? 1 : 0;
    uint32_t typeKey = (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16) ? 1 : 0;
    uint32_t maskTypeKey = mmInfo.maskType;
    mmInfo.tilingKey = (maskTypeKey << DIM_2) + (typeKey << DIM_1) + int8Key;
}

void GetTilingKey(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    if (param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE ||
        param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE) {
        GetMLATilingKey(mmInfo, param);
    } else if (param.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        mmInfo.tilingKey = mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16 ? 1 : 0;
    } else if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
        GetKvCacheTilingKey(mmInfo);
    } else {
        if (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16) {
            mmInfo.tilingKey = static_cast<uint32_t>(TilingKeyType::TILING_BF16_DATA);
        } else if (mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_INT8) {
            if (param.outDataType == TENSOR_DTYPE_BF16) {
                mmInfo.tilingKey = static_cast<uint32_t>(TilingKeyType::TILING_INT8_IN_BF16_OUT);
            } else if (param.outDataType == TENSOR_DTYPE_FLOAT16) {
                mmInfo.tilingKey = static_cast<uint32_t>(TilingKeyType::TILING_INT8_IN_HALF_OUT);
            }
        }
        GetSwaTilingKey(mmInfo);
    }
}

void GetSpecTilingKey(UnpadFlashAttentionInfo &mmInfo, const OpParam::UnpadFlashAttention &param)
{
    // Currently, data type << 20, masktype << 0
    if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND &&
        mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.tilingKey = SPEC_TILING_KEY + 1;
    } else if ((param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND ||
        param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND) &&
        mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.tilingKey = SPEC_TILING_KEY + (static_cast<uint32_t>(mmInfo.splitm) << DIM_4);
    } else if (mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK &&
        param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND &&
        mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_FLOAT16) {
        mmInfo.tilingKey = SPEC_TILING_KEY + 1;
    } else if (mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK &&
        param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND &&
        mmInfo.tensors.query.desc.dtype == TENSOR_DTYPE_BF16) {
        mmInfo.tilingKey = SPEC_TILING_KEY;
    }
}
void FlashAttentionEncoderBNSD(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param, int32_t embed)
{
    bool optEnable = mmInfo.maskType == 0 && embed == 128 &&
                     mmInfo.scaleType == OpParam::UnpadFlashAttention::SCALE_TOR && param.dataShapeType == 1;
    if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND && optEnable) {
        mmInfo.splitm = true;
        for (size_t seqIdx = 0; seqIdx < param.qSeqLen.size(); seqIdx++) {
            int32_t qSeqlen = *(param.qSeqLen.data() + seqIdx);
            if (qSeqlen <= SPLIT_M_THRESHOLD) {
                mmInfo.splitm = false;
                break;
            }
        }
    }
}

void FlashAttentionFillRazorInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    mmInfo.tileQ = param.tileQ;
    mmInfo.tileKv = param.tileKv;
    mmInfo.textQLen = param.textQLen;
    mmInfo.textKvLen = param.textKvLen;
    mmInfo.razorLen = param.razorLen;
    mmInfo.preTokens = param.preTokens;
    mmInfo.nextTokens = param.nextTokens;
}

void FlashAttentionFillInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param,
                            size_t batch, int32_t embed)
{
    mmInfo.type = param.type;
    mmInfo.batchSize = static_cast<int32_t>(batch);
    mmInfo.innerBatchSize = param.headSize;
    mmInfo.embeddingSize = embed;
    mmInfo.embeddingSizeV = param.headDimV == 0 ? embed : param.headDimV;
    mmInfo.qSeq = param.qSeqLen.data();
    mmInfo.kvSeq = param.kvSeqLen.data();
    FlashAttentionFillRazorInfo(mmInfo, param);
    mmInfo.scaleType = param.scaleType;
    mmInfo.batchRunStatus = param.batchRunStatus.data();
    mmInfo.tor = param.tor;
    mmInfo.kvHead = param.kvHead == 0 ? param.headSize : param.kvHead;
    mmInfo.isClamp = param.isClamp;
    mmInfo.clampMin = param.clampMin;
    mmInfo.clampMax = param.clampMax;
    mmInfo.kvShareMap = param.kvShareMap;
    mmInfo.kvShareLen = param.kvShareLen;
    mmInfo.isTriuMask = param.isTriuMask;
    mmInfo.alibiLeftAlign = param.alibiLeftAlign;
    mmInfo.kTensorList = param.kTensorList;
    mmInfo.vTensorList = param.vTensorList;
    mmInfo.kShareTensorList = param.kShareTensorList;
    mmInfo.vShareTensorList = param.vShareTensorList;
    mmInfo.maskType = static_cast<uint32_t>(param.maskType);
    mmInfo.quantType = static_cast<uint32_t>(param.quantType);
    mmInfo.windowSize = param.windowSize;
    mmInfo.cacheType = static_cast<uint32_t>(param.cacheType);
    GetTilingKey(mmInfo, param);
    if (mmInfo.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        for (int32_t seqIdx = 0; seqIdx < mmInfo.batchSize; seqIdx++) {
            mmInfo.batchShareMap[mmInfo.kvShareMap[seqIdx]].push_back(seqIdx);
        }
    }
    mmInfo.isMLA = ((param.headDimV != 0 &&
        (mmInfo.embeddingSize != mmInfo.embeddingSizeV || mmInfo.embeddingSize > MLA_THRESHOLD))
        || (param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE ||
        param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE))
            ? true : false;
    if (param.dataShapeType == 1) {
        auto &kcacheShape = mmInfo.tensors.kCache.desc.dims;
        mmInfo.maxSeqLen = kcacheShape[kcacheShape.size() - DIM_2];
        auto &qShape = mmInfo.tensors.query.desc.dims;
        mmInfo.maxQSeqLen = qShape[qShape.size() - DIM_2];
    }
    mmInfo.dataShapeType = param.dataShapeType;
    MKI_LOG(INFO) << "window size is " << mmInfo.windowSize << " cache type is " << mmInfo.cacheType;
}

Status FlashAttentionVCacheCheck(const UnpadFlashAttentionInfo &mmInfo, const OpParam::UnpadFlashAttention &param)
{
    SVector<int64_t> vcacheShape;
    if (CheckEmptyTensor(mmInfo.tensors.vCache)) {
        vcacheShape = param.vTensorList[0].desc.dims; // [max_seq, hiddenSize]
    } else {
        vcacheShape = mmInfo.tensors.vCache.desc.dims;
    }
    auto vcacheDim = vcacheShape.size();
    MKI_CHECK(vcacheDim > 0, "vcache shape is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_LOG(INFO) << "[InitInfo] embeddingSizeV is:" << vcacheShape[vcacheDim - 1] << " " << mmInfo.kvHead << " "
                  << mmInfo.embeddingSizeV;
    MKI_CHECK((vcacheShape[vcacheDim - 1] == mmInfo.embeddingSizeV) ||
                  (vcacheShape[vcacheDim - 1] == mmInfo.kvHead * mmInfo.embeddingSizeV),
              "headdimV is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status InitInfo(UnpadFlashAttentionInfo &mmInfo, OpParam::UnpadFlashAttention &param)
{
    OP_TILING_CHECK_STATUS_RETURN(FlashAttentionPreCheck(param));
    size_t batch = 0;
    int32_t embed = 0;
    // alibi kernel改写type，上层不用感知变化
    if (param.type == OpParam::UnpadFlashAttention::UNPAD_ALIBI_FLASH_ATTENTION_ND) {
        param.maskType = OpParam::UnpadFlashAttention::MASK_TYPE_ALIBI;
    }
    SVector<int64_t> kcacheShape;
    if (CheckEmptyTensor(mmInfo.tensors.kCache)) {
        mmInfo.batchContinuous = false;
        batch = param.kTensorList.size();
        kcacheShape = param.kTensorList[0].desc.dims; // [max_seq, hiddenSize]
        MKI_CHECK(kcacheShape.size() == 2, "invalid kvCache single batch shape",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        mmInfo.batchContinuous = true;
        kcacheShape = mmInfo.tensors.kCache.desc.dims;
    }
    InitEmbed(embed, mmInfo, param);
    FlashAttentionEncoderBNSD(mmInfo, param, embed);
    batch = param.qSeqLen.size();
    OP_TILING_CHECK_STATUS_RETURN(InitMaxKVSeqlen(batch, kcacheShape, mmInfo, param));
    FlashAttentionFillInfo(mmInfo, param, batch, embed);
    // mask_none/mask_free opt模板方法tilingkey
    bool optEnable = (mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_NONE ||
                        mmInfo.maskType == OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK) &&
                        mmInfo.embeddingSize <= 128 && !mmInfo.isMLA &&
                        mmInfo.scaleType == OpParam::UnpadFlashAttention::SCALE_TOR;
    if (optEnable) {
        GetSpecTilingKey(mmInfo, param);
    }
    MKI_CHECK(optEnable || mmInfo.maskType != OpParam::UnpadFlashAttention::MASK_TYPE_CAUSAL_MASK,
            "maskType is mask free, not support opt unable, isMLA or embeddingSize > 128 or scaleType not tor.",
            return Status::FailStatus(ERROR_INVALID_VALUE));
    if (param.type != OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE &&
        param.type != OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE) {
        OP_TILING_CHECK_STATUS_RETURN(FlashAttentionVCacheCheck(mmInfo, param));
    }
    OP_TILING_CHECK_STATUS_RETURN(FlashAttentionPostCheck(mmInfo, param));
    if (mmInfo.windowSize > 0) {
        OP_TILING_CHECK_STATUS_RETURN(FlashAttentionSwaCheck(mmInfo));
    }
    if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
        mmInfo.blockSize = (mmInfo.tensors.kCache.desc.dims).at(1);
        MKI_CHECK(mmInfo.blockSize == 128,
            "blockSize should be 128", return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.maxNumBlocks = (mmInfo.tensors.blockTable.desc.dims).at(1);
    }
    FlashAttentionLog(mmInfo, param);
    return Status::OkStatus();
}
void InitMaxKVSeqlenBNSD(Mki::SVector<int64_t> &kcacheShape, AtbOps::UnpadFlashAttentionInfo &mmInfo, size_t &batch)
{
    if (kcacheShape.size() == DIM_5) {                      // 5 is shape : [layer, batch, headNum, maxSeq，headDim]
        mmInfo.maxKvSeqLen = kcacheShape.at(DIM_3);         // MaxSeqLen is the 4st dimension of K
        batch = static_cast<size_t>(kcacheShape.at(DIM_1)); // Batch is the 1th dimension of Q
        MKI_LOG(INFO) << "batch is:" << batch;
    } else {
        MKI_LOG(INFO) << "kcacheShape.size() is:" << kcacheShape.size();
        mmInfo.maxKvSeqLen = kcacheShape.at(0);
    }
}
void InitEmbed(int32_t &embed, const UnpadFlashAttentionInfo &mmInfo,
               const AtbOps::OpParam::UnpadFlashAttention &param)
{
    embed = (mmInfo.tensors.query.desc.dims).at(1) / param.headSize; // headdim
    if (mmInfo.tensors.query.desc.dims.size() == DIM_3) {
        embed = (mmInfo.tensors.query.desc.dims).at(2); // 2 is head_size dim
    }

    if (param.dataShapeType == 1 && (param.type == OpParam::UnpadFlashAttention::UNPAD_ALIBI_FLASH_ATTENTION_ND ||
                                     param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND ||
                                     param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND ||
                                     param.type == OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION ||
                                     param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND)) {
        // encoder bnsd q shape 为B,N,S,D
        embed = (mmInfo.tensors.query.desc.dims).at(DIM_3);
        MKI_LOG(INFO) << "[InitInfo] embed is:" << embed;
    } else if (param.dataShapeType == 1 &&
               (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND ||
                param.type == OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER)) {
        // decoder shape 为 B，ND (num_tokens,hiddenSize)
        embed = (mmInfo.tensors.query.desc.dims).at(1) / param.headSize;
        MKI_LOG(INFO) << "[InitInfo] embed is:" << embed;
    }

    MKI_LOG(INFO) << "[InitInfo] embed is:" << embed;
}

void InitMaxKVSeqlenMLACombineCache(Mki::SVector<int64_t> &kcacheShape, UnpadFlashAttentionInfo &mmInfo)
{
    if (CheckEmptyTensor(mmInfo.tensors.layerId)) {
        mmInfo.maxKvSeqLen = mmInfo.maxSeqLen;
        if (kcacheShape.size() == DIM_4) {  // [batch, maxseq, heads, headdim]
            mmInfo.maxKvSeqLen = kcacheShape.at(DIM_1);
        } else {
            mmInfo.maxKvSeqLen = kcacheShape.at(DIM_0);
            mmInfo.isNoCache = true;
        }
    } else {
        if (kcacheShape.size() == DIM_4) {                  // [layer, batch, maxseq, hiddensize]
            mmInfo.maxKvSeqLen = kcacheShape.at(DIM_2);
        }
    }
}

Status InitMaxKVSeqlen(size_t &batch, Mki::SVector<int64_t> &kcacheShape, UnpadFlashAttentionInfo &mmInfo,
    AtbOps::OpParam::UnpadFlashAttention &param)
{
    if (param.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        SVector<int64_t> &kcacheshareShape = param.kShareTensorList[0].desc.dims;
        mmInfo.maxKvSeqLen = kcacheShape.at(DIM_0);
        mmInfo.maxKvShareSeqLen = kcacheshareShape.at(DIM_0);
    } else if (param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_COMBINE_CACHE ||
        param.type == OpParam::UnpadFlashAttention::MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE) {
        OP_TILING_CHECK_STATUS_RETURN(GetFlashAttentionNoCacheMaskInfo(mmInfo, param, mmInfo.tensors.mask));
        InitMaxKVSeqlenMLACombineCache(kcacheShape, mmInfo);
    } else if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND ||
        param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION ||
        (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND && kcacheShape.size() < DIM_4)) {
        OP_TILING_CHECK_STATUS_RETURN(GetFlashAttentionNoCacheMaskInfo(mmInfo, param, mmInfo.tensors.mask));
        mmInfo.maxKvSeqLen = mmInfo.dataShapeType == 0 ? kcacheShape[kcacheShape.size() - DIM_2] : mmInfo.maxSeqLen;
        mmInfo.isNoCache = true;
    } else if (param.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
        mmInfo.isNoCache = true;
        mmInfo.maxKvSeqLen = static_cast<std::int64_t>(kcacheShape.at(0)) * kcacheShape.at(0) / batch;
        OP_TILING_CHECK_STATUS_RETURN(GetFlashAttentionNoCacheMaskInfo(mmInfo, param, mmInfo.tensors.mask));
    } else {
        if (param.dataShapeType == 1) {
            InitMaxKVSeqlenBNSD(kcacheShape, mmInfo, batch);
        } else {
            if (kcacheShape.size() == DIM_4) {                  // 4 is shape : [layer, batch, maxseq, hiddensize]
                mmInfo.maxKvSeqLen = kcacheShape.at(2);         // MaxSeqLen is the 2st dimension of K
                batch = static_cast<size_t>(kcacheShape.at(1)); // Batch is the 1th dimension of Q
            } else if (kcacheShape.size() == DIM_3) {
                mmInfo.maxKvSeqLen = kcacheShape.at(1); // [batch, maxseq, hiddensize]
                batch = static_cast<size_t>(kcacheShape.at(0));
            } else {
                mmInfo.maxKvSeqLen = kcacheShape.at(0);
            }
        }
        OP_TILING_CHECK_STATUS_RETURN(GetFlashAttentionNoCacheMaskInfo(mmInfo, param, mmInfo.tensors.mask));
    }
    return Status::OkStatus();
}
} // namespace AtbOps
