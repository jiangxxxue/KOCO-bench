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
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/math/math.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "flash_attention_nz_tiling_dependency.h"

namespace AtbOps {
using namespace Mki;
inline void FlashAttentionNzLog(UnpadFlashAttentionNzInfo &mmInfo, const uint32_t *tilingParam, uint64_t tilingSize,
                                uint32_t blockDim)
{
    MKI_LOG(INFO) << "batch is: " << mmInfo.batchSize << " maxSeq is: " << mmInfo.maxSeqLen
                  << " head is: " << mmInfo.innerBatchSize << " embed is: " << mmInfo.embeddingSize
                  << " qSeq is: " << mmInfo.qSeq << " kvSeq is: " << mmInfo.kvSeq << " kvHeads is " << mmInfo.kvHeads
                  << " qTokens is " << mmInfo.qTokens << " maskStride is " << mmInfo.maskStride << " isCache is "
                  << mmInfo.isCache << " headMaskStride is " << mmInfo.headMaskStride << " batchMaskStride is "
                  << mmInfo.batchMaskStride << " qTight is " << mmInfo.qTight << " longSeq is " << mmInfo.isLongSeq
                  << " windowLen is " << mmInfo.windowLen << " cacheType is " <<mmInfo.cacheType;
    MKI_LOG(INFO) << "tiling is";
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(GetNzRealCoreTilingOffset() + mmInfo.batchSize * NZ_REAL_CORE_TILING_SIZE); i++) {
        MKI_LOG(INFO) << i << " index: " << tilingParam[i] << ",";
    }
    MKI_LOG(INFO) << "launchBufferSize = " << tilingSize << ", block dim = " << blockDim;
}

inline Status FlashAttentionNzCheck(const LaunchParam &launchParam, OpParam::UnpadFlashAttentionNz &param)
{
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::UnpadFlashAttentionNz), "OpParam is invalid",
                 return Status::FailStatus(ERROR_INFERSHAPE_ERROR));
    MKI_CHECK(param.headSize > 0, "headSize is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((param.kvHead > 0 && param.kvHead <= param.headSize && param.headSize % param.kvHead == 0) ||
                     (param.kvHead == 0),
                 "kvHead is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.data() != nullptr, "qSeq cannot be nullptr",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.kvSeqLen.data() != nullptr, "kvSeq cannot be nullptr",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.qSeqLen.size() == param.kvSeqLen.size(), "qSeqLen size Not equal to kvSeqLen",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    auto batch = param.qSeqLen.size();
    MKI_CHECK(batch > 0 && batch <= NZ_BATCH_LIMIT, "qSeqLen size is too big",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    auto qShape = launchParam.GetInTensor(DIM_0).desc.dims;
    int32_t embeddingSize = static_cast<int32_t>(qShape.at(1) * NZ_BLOCK_SIZE / param.headSize);
    MKI_CHECK(embeddingSize <= MAX_EMBEDDING && embeddingSize > 0, "headdim is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(param.cacheType == 0 || param.cacheType == 1, "cacheType is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

inline Status FlashAttentionSwaCheck(const UnpadFlashAttentionNzInfo mmInfo)
{
    MKI_CHECK(mmInfo.dataDimOrder == 0, "swa not support BNSD input",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.scaleType == OpParam::UnpadFlashAttentionNz::SCALE_TOR,
                "swa not support logN", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK((mmInfo.maskType == OpParam::UnpadFlashAttentionNz::MaskType::MASK_TYPE_SWA_NORM
            || mmInfo.maskType == OpParam::UnpadFlashAttentionNz::MaskType::MASK_TYPE_SWA_COMPRESS),
            "Swa only support swa mask.",
            return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}


Status CheckSeqLen(const OpParam::UnpadFlashAttentionNz &param, int32_t maxSeq)
{
    auto maxQSeq = std::max_element(param.qSeqLen.begin(), param.qSeqLen.end());
    auto maxKvSeq = std::max_element(param.kvSeqLen.begin(), param.kvSeqLen.end());
    MKI_CHECK(*maxQSeq <= maxSeq, "qSeqLen larger than maxSeqLen",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(*maxKvSeq <= maxSeq, "kvSeqLen larger than maxSeqLen",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
    return Status::OkStatus();
}

Status SetAlibiMaskInfo(UnpadFlashAttentionNzInfo &mmInfo, const LaunchParam &launchParam,
                        OpParam::UnpadFlashAttentionNz &param)
{
    auto maskShape = launchParam.GetInTensor(DIM_4).desc.dims;
    auto maskDimZero = maskShape.at(DIM_0);
    auto &tensorAlibiCoeff = launchParam.GetInTensor(DIM_5);
    if (!CheckEmptyTensor(tensorAlibiCoeff)) { // [1, 256//16, 256, 16]
        MKI_CHECK(mmInfo.maskStride == LONG_SEQ_ALIBI_LEN &&
                            maskShape.at(DIM_1) * maskShape.at(DIM_3) == LONG_SEQ_ALIBI_LEN,
                        "alibi mask shape must be [256, 256] for long seq opt",
                        return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(mmInfo.isTriu != 0, "alibi mask must be triu",
                        return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.batchMaskStride = 0;
        mmInfo.headMaskStride = 0;
        mmInfo.isAlibiMaskSqrt = param.isAlibiMaskSqrt;
    } else if (maskShape.at(DIM_1) * maskShape.at(DIM_3) == LONG_SEQ_LEN &&
                maskShape.at(DIM_2) > LONG_SEQ_LEN) { // [heads, 128//16, maxseq, 16]
        mmInfo.alibiCompressOffset = static_cast<uint32_t>(maskShape.at(DIM_2));
        mmInfo.batchMaskStride = 0;
        mmInfo.headMaskStride = LONG_SEQ_LEN;
    } else if (maskDimZero == param.headSize) {
        OP_TILING_CHECK_STATUS_RETURN(CheckSeqLen(param, maskShape.at(DIM_1) * maskShape.at(DIM_3)));
        mmInfo.batchMaskStride = 0;
        mmInfo.headMaskStride = static_cast<uint32_t>(mmInfo.maskStride);
    } else {
        OP_TILING_CHECK_STATUS_RETURN(CheckSeqLen(param, maskShape.at(DIM_1) * maskShape.at(DIM_3)));
        MKI_CHECK(static_cast<uint64_t>(param.headSize) * static_cast<uint64_t>(mmInfo.maskStride) <=
                            UINT32_MAX,
                        "maskStride * headSize can not large than UINT32_MAX",
                        return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.batchMaskStride = static_cast<uint32_t>(param.headSize * mmInfo.maskStride);
        mmInfo.headMaskStride = static_cast<uint32_t>(mmInfo.maskStride);
    }
    return Status::OkStatus();
}

Status SetSwaMaskInfo(UnpadFlashAttentionNzInfo &mmInfo, const LaunchParam &launchParam,
                      OpParam::UnpadFlashAttentionNz &param)
{
    auto maskShape = launchParam.GetInTensor(DIM_4).desc.dims;
    mmInfo.maskStride = static_cast<int32_t>(maskShape.at(DIM_2));
    if (mmInfo.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_NORM) {
        MKI_CHECK(maskShape.size() == 4,
                    "SWA mask only support [1, seqlen/16, seqlen, 16] shape currently.",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        OP_TILING_CHECK_STATUS_RETURN(CheckSeqLen(param, maskShape.at(DIM_1) * maskShape.at(DIM_3)));
        mmInfo.headMaskStride = 0;
        mmInfo.batchMaskStride = 0;
        mmInfo.isLongSeq = 0;
        mmInfo.isTriu = 0;
    } else if (mmInfo.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_COMPRESS) {
        MKI_CHECK(maskShape.size() == 4,
                    "SWA mask only support [1, 512/16, 512, 16] shape currently.",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK((maskShape.at(DIM_1) * maskShape.at(DIM_3) == SWA_MASK_COMPRESS_SIZE)
                && (maskShape.at(DIM_2) == SWA_MASK_COMPRESS_SIZE),
                    "SWA mask only support [1, 512/16, 512, 16] shape currently.",
                    return Status::FailStatus(ERROR_INVALID_VALUE));
        mmInfo.headMaskStride = 0;
        mmInfo.batchMaskStride = 0;
        mmInfo.isLongSeq = 0;
        mmInfo.isTriu = 0;
    } else {
        return Status::FailStatus(ERROR_INVALID_VALUE, "Invalid MaskType");
    }
    return Status::OkStatus();
}

Status FlashAttentionNzMaskInfo(UnpadFlashAttentionNzInfo &mmInfo, const LaunchParam &launchParam,
                                OpParam::UnpadFlashAttentionNz &param)
{
    auto maskShape = launchParam.GetInTensor(DIM_4).desc.dims;
    mmInfo.maskStride = static_cast<int32_t>(maskShape.at(DIM_2));
    auto maskDimZero = maskShape.at(DIM_0);
    switch (mmInfo.maskType) {
        case OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_NORM:
        case OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_COMPRESS:
            OP_TILING_CHECK_STATUS_RETURN(SetSwaMaskInfo(mmInfo, launchParam, param));
            break;
        case OpParam::UnpadFlashAttentionNz::MASK_TYPE_ALIBI:
            OP_TILING_CHECK_STATUS_RETURN(SetAlibiMaskInfo(mmInfo, launchParam, param));
            break;
        case OpParam::UnpadFlashAttentionNz::MASK_TYPE_NORM:
            mmInfo.isLongSeq = ((mmInfo.isTriu == 1) && (mmInfo.maskStride == LONG_SEQ_LEN)) ? 1 : 0;
            mmInfo.headMaskStride = 0;
            mmInfo.batchMaskStride = static_cast<uint32_t>(
                ((maskDimZero == static_cast<int64_t>(param.qSeqLen.size())) && (mmInfo.isLongSeq == 0))
                    ? mmInfo.maskStride
                    : 0);
            if (mmInfo.isLongSeq == 0) {
                OP_TILING_CHECK_STATUS_RETURN(CheckSeqLen(param, maskShape.at(DIM_1) * maskShape.at(DIM_3)));
            }
            break;
        default: return Status::FailStatus(ERROR_INVALID_VALUE, "Invalid MaskType");
    }
    return Status::OkStatus();
}

Status FlashAttentionNzPrecInfoCheck(const UnpadFlashAttentionNzInfo &mmInfo)
{
    uint32_t blockDim = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (mmInfo.precType == OpParam::UnpadFlashAttentionNz::PrecType::BMM1_FP32_EXP_FP32) {
        MKI_CHECK(blockDim <= 8, "FP32 can not support", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    if (mmInfo.precType == OpParam::UnpadFlashAttentionNz::PrecType::BMM2_ONLINE_SOFTMAX_FP16) {
        MKI_CHECK(blockDim <= 8, "Cube online softmax can not support", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    return Status::OkStatus();
}

Status FillFlashAttentionNzInfo(UnpadFlashAttentionNzInfo &mmInfo, const LaunchParam &launchParam,
    OpParam::UnpadFlashAttentionNz &param)
{
    SVector<int64_t> kCacheShape;
    if (!mmInfo.batchContinuous) {
        kCacheShape = param.kTensorList[0].desc.dims;
    } else {
        kCacheShape = launchParam.GetInTensor(DIM_1).desc.dims;
    }
    auto qShape = launchParam.GetInTensor(DIM_0).desc.dims;
    auto head = param.headSize; //  headsize
    mmInfo.isCache = kCacheShape.size() == DIM_5;
    mmInfo.type = param.type;
    mmInfo.batchSize = static_cast<int32_t>(param.qSeqLen.size());
    mmInfo.innerBatchSize = static_cast<int32_t>(head);
    mmInfo.embeddingSize = param.dataDimOrder == OpParam::UnpadFlashAttentionNz::TYPE_BNSD ?
    static_cast<int32_t>(qShape.at(1) * NZ_BLOCK_SIZE) : static_cast<int32_t>(qShape.at(1) * NZ_BLOCK_SIZE / head);
    mmInfo.qSeq = param.qSeqLen.data();
    mmInfo.kvSeq = param.kvSeqLen.data();
    mmInfo.kvHeads = param.kvHead > 0 ? param.kvHead : head;
    mmInfo.qTokens = qShape.at(DIM_2);
    mmInfo.tor = param.tor;
    mmInfo.qTight = (mmInfo.isCache || !mmInfo.batchContinuous) ? 0 : 1;
    mmInfo.isTriu = param.isTriuMask;
    mmInfo.maskType = param.maskType;
    mmInfo.kTensorList = param.kTensorList;
    mmInfo.vTensorList = param.vTensorList;
    mmInfo.dataDimOrder = param.dataDimOrder;
    mmInfo.precType = param.precType;
    mmInfo.scaleType = param.scaleType;
    OP_TILING_CHECK_STATUS_RETURN(FlashAttentionNzPrecInfoCheck(mmInfo));
    mmInfo.alibiLeftAlign = param.alibiLeftAlign;
    MKI_CHECK(param.windowSize >= 0, "invalid param.windowSize < 0, which is: " << param.windowSize,
              return Status::FailStatus(Mki::ERROR_INVALID_VALUE));
    mmInfo.windowLen = param.windowSize;
    mmInfo.cacheType = param.cacheType;
    if (mmInfo.maskType != OpParam::UnpadFlashAttentionNz::MASK_TYPE_NONE
        && !(param.type == OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER
            && param.maskType == OpParam::UnpadFlashAttentionNz::MASK_TYPE_SWA_NORM)) {
        auto maskShape = launchParam.GetInTensor(DIM_4).desc.dims;
        auto maxSeq = mmInfo.isCache ? kCacheShape.at(DIM_3) : maskShape.at(DIM_2);
        mmInfo.maxSeqLen = static_cast<int32_t>(maxSeq);
        auto maxKVSeq = mmInfo.batchContinuous ? kCacheShape.at(DIM_3) : kCacheShape.at(DIM_1);
        mmInfo.maxKVSeqLen = static_cast<int32_t>(maxKVSeq);
        OP_TILING_CHECK_STATUS_RETURN(FlashAttentionNzMaskInfo(mmInfo, launchParam, param));
    } else {
        auto maxSeq = mmInfo.isCache ? kCacheShape.at(DIM_3) : 0;
        mmInfo.maxSeqLen = static_cast<int32_t>(maxSeq);
        auto maxKVSeq = mmInfo.batchContinuous ? kCacheShape.at(DIM_3) : kCacheShape.at(DIM_1);
        mmInfo.maxKVSeqLen = static_cast<int32_t>(maxKVSeq);
    }
    return Status::OkStatus();
}

Status FlashAttentionNzTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto param = AnyCast<OpParam::UnpadFlashAttentionNz>(launchParam.GetParam());
    OP_TILING_CHECK_STATUS_RETURN(FlashAttentionNzCheck(launchParam, param));

    UnpadFlashAttentionNzInfo mmInfo;
    auto &kCache = launchParam.GetInTensor(DIM_1);
    if (CheckEmptyTensor(kCache)) {
        mmInfo.batchContinuous = false;
    }
    OP_TILING_CHECK_STATUS_RETURN(FillFlashAttentionNzInfo(mmInfo, launchParam, param));
    // Check params
    uint32_t blockDim = 0;

    uint8_t *tilingHost = kernelInfo.GetTilingHostAddr();
    uint64_t tilingSize = kernelInfo.GetTilingSize();
    uint32_t *tilingParam = reinterpret_cast<uint32_t *>(tilingHost);
    auto ret = GetUnpadFlashAttentionTilingParam(mmInfo, blockDim, tilingParam, tilingSize);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    kernelInfo.SetBlockDim(blockDim);
    if (mmInfo.windowLen > 0) {
        OP_TILING_CHECK_STATUS_RETURN(FlashAttentionSwaCheck(mmInfo));
    }
    FlashAttentionNzLog(mmInfo, tilingParam, tilingSize, blockDim);
    kernelInfo.SetTilingId(tilingParam[GetTilingKeyIndex()]);
    return Status::OkStatus();
}

} // namespace AtbOps
