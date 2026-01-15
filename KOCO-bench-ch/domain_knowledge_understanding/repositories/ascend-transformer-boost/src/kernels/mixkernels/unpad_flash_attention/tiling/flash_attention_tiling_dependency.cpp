/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "flash_attention_tiling_dependency.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <securec.h>
#include <string>

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>

#define UNUSED_VALUE(x) (void)(x)

namespace AtbOps {
const int32_t PP_MM_NUM = 8;
const int32_t PP_INDEX = 16;
const int32_t INDEX2 = 2;
const int32_t INDEX3 = 3;
const int32_t INDEX4 = 4;
const int32_t INDEX5 = 5;
const int32_t INDEX6 = 6;
const int32_t INDEX7 = 7;
const int32_t INDEX8 = 8;
const int32_t INDEX9 = 9;
const int32_t INDEX10 = 10;
const int32_t INDEX11 = 11;
const int32_t INDEX12 = 12;
const int32_t INDEX13 = 13;
const int32_t INDEX14 = 14;
const int32_t INDEX15 = 15;
const int32_t INDEX16 = 16;
const int32_t INDEX17 = 17;
const int32_t INDEX18 = 18;
const int32_t INDEX19 = 19;
const int32_t INDEX20 = 20;
const int32_t INDEX21 = 21;
const int32_t INDEX22 = 22;
const int32_t INDEX23 = 23;
const int32_t INDEX24 = 24;
const int32_t INDEX25 = 25;
const int32_t INDEX26 = 26;
const int32_t INDEX27 = 27;
const int32_t INDEX28 = 28;
const int32_t INDEX29 = 29;
const int32_t INDEX30 = 30;
const int32_t INDEX31 = 31;
const int32_t INDEX32 = 32;
const int32_t INDEX33 = 33;
const int32_t INDEX34 = 34;
const int32_t INDEX35 = 35;
const int32_t INDEX36 = 36;
const int32_t SPECIALNUMBATCH = 16;
const int32_t SPECIALNUMHEADS = 8;
constexpr std::array<int32_t, PP_MM_NUM> PP_MM = {16, 32, 48, 64, 80, 96, 112, 128};
inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}
const int32_t PP_NN_NUM = 16;
constexpr std::array<int32_t, PP_NN_NUM> PP_NN = {16,  32,  48,  64,  80,  96,  112, 128,
                                                  144, 160, 176, 192, 208, 224, 240, 256};
inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound)
{
    return (val > PP_NN[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

const int32_t PP_BLOCK_BUFFER_SIZE_FP32 = 128 * 64;
const int32_t PP_NN_FP32_NUM = 8;
constexpr std::array<int32_t, PP_NN_FP32_NUM> PP_NN_FP32 = {16, 32, 48, 64, 80, 96, 112, 128};
inline int32_t ConvertValueToIndexNNFP32(int32_t val, int32_t idxBound)
{
    return (val > PP_NN_FP32[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;
const int32_t DECODER_PP_BLOCK_BUFFER_SIZE = 128 * 128;
const int32_t HIGH_32BIT = 32;
const int32_t EMBEDINF_LIMIT = 128;
const int32_t NUM_TWO = 2;
inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> HIGH_32BIT); }
inline uint32_t GetLoww32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

Status CheckSeqlen(const UnpadFlashAttentionInfo &mmInfo, const AddrOffsets &addrOffsets, int32_t qSeqlen,
                   int32_t kvSeqlen, int32_t seqIdx)
{
    UNUSED_VALUE(seqIdx);
    if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION ||
        mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_ND) {
        MKI_CHECK(qSeqlen >= 0, "qSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqlen >= 0, "kvSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        if (mmInfo.dataShapeType == 0) {
            MKI_CHECK(kvSeqlen == qSeqlen || qSeqlen == 1, "kvSeqlen should equal qSeqlen or qSeqlen == 1",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        }
    } else if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER) {
        MKI_CHECK(qSeqlen > 0, "qSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqlen >= 0, "kvSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    } else if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
        MKI_CHECK(qSeqlen >= 0, "qSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqlen >= 0, "kvSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK((kvSeqlen - qSeqlen) % 128 == 0, "prefix should be 128-aligned",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
    } else if (mmInfo.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        MKI_CHECK(qSeqlen == 1, "qSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqlen > 0, "kvSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        MKI_CHECK(qSeqlen > 0, "qSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqlen > 0, "kvSeqlen is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    MKI_CHECK(addrOffsets.totalQBlkNum >= 0, "totalQBlkNum overflow",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    if (mmInfo.maskType != OpParam::UnpadFlashAttention::MASK_TYPE_SWA_COMPRESS) {
        MKI_CHECK((mmInfo.maxSeqLen == 0) || (mmInfo.maxSeqLen == LONG_SEQ_ALIBI_LEN) ||
            (mmInfo.maxSeqLen == LONG_SEQ_LEN) ||
            (mmInfo.dataShapeType == 0 && kvSeqlen <= mmInfo.maxSeqLen && qSeqlen <= mmInfo.maxSeqLen) ||
            (mmInfo.dataShapeType == 1 && kvSeqlen <= mmInfo.maxSeqLen && qSeqlen <= mmInfo.maxQSeqLen),
            "mask invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    return Status::OkStatus();
}

uint32_t SetTilingKey(const UnpadFlashAttentionInfo &mmInfo)
{
    uint32_t tilingKey = 0;
    tilingKey = static_cast<uint32_t>(mmInfo.tilingKey);
    MKI_LOG(INFO) << "tilingKey :" << tilingKey;
    return tilingKey;
}

void FillTilingHead(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                    uint32_t *tilingParam, int32_t kvRealHeads)
{
    float clampMin = mmInfo.clampMin;
    uint32_t *clampMinUptr = static_cast<uint32_t *>(static_cast<void *>(&clampMin));
    float clampMax = mmInfo.clampMax;
    uint32_t *clampMaxUptr = static_cast<uint32_t *>(static_cast<void *>(&clampMax));
    tilingParam[0] = static_cast<uint32_t>(mmInfo.batchSize);
    tilingParam[1] = static_cast<uint32_t>(mmInfo.maxSeqLen);
    tilingParam[INDEX2] = static_cast<uint32_t>(mmInfo.innerBatchSize);
    tilingParam[INDEX3] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[INDEX4] = static_cast<uint32_t>(kvRealHeads);
    tilingParam[INDEX5] = torUptr;
    tilingParam[INDEX6] = static_cast<uint32_t>(mmInfo.headStride);
    tilingParam[INDEX7] = static_cast<uint32_t>(mmInfo.maskStride);
    tilingParam[INDEX8] = mmInfo.isTriuMask;
    tilingParam[INDEX9] = static_cast<uint32_t>(addrOffsets.totalQBlkNum);
    tilingParam[INDEX10] = static_cast<uint32_t>(mmInfo.isClamp);
    tilingParam[INDEX11] = *clampMinUptr;
    tilingParam[INDEX12] = *clampMaxUptr;
    tilingParam[INDEX13] = mmInfo.headStride;
    tilingParam[INDEX14] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    tilingParam[INDEX15] = static_cast<uint32_t>(TILING_PARA_SIZE);
    tilingParam[INDEX16] = SetTilingKey(mmInfo);
    tilingParam[INDEX17] = mmInfo.isLongSeq;
    tilingParam[INDEX18] = mmInfo.maxKvSeqLen;
    tilingParam[INDEX19] = mmInfo.isAlibiMaskSqrt;
    tilingParam[INDEX20] = mmInfo.maskType;
    tilingParam[INDEX21] = mmInfo.alibiCompressOffset;
    tilingParam[INDEX22] = mmInfo.alibiLeftAlign;
    tilingParam[INDEX23] = static_cast<uint32_t>(mmInfo.embeddingSizeV);
    tilingParam[INDEX24] = static_cast<uint32_t>(mmInfo.quantType);
    tilingParam[INDEX25] = mmInfo.dataShapeType;
    tilingParam[INDEX26] = mmInfo.scaleType;
    tilingParam[INDEX27] = mmInfo.windowSize;
    if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
        tilingParam[INDEX28] = static_cast<uint32_t>(mmInfo.maxNumBlocks);
    }
    tilingParam[INDEX29] = mmInfo.maxQSeqLen;
    tilingParam[INDEX30] = static_cast<uint32_t>(mmInfo.preTokens);
    tilingParam[INDEX31] = static_cast<uint32_t>(mmInfo.nextTokens);
    tilingParam[INDEX32] = static_cast<uint32_t>(mmInfo.razorLen);
    tilingParam[INDEX33] = static_cast<uint32_t>(mmInfo.tileQ);
    tilingParam[INDEX34] = static_cast<uint32_t>(mmInfo.tileKv);
    tilingParam[INDEX35] = static_cast<uint32_t>(mmInfo.textQLen);
    tilingParam[INDEX36] = static_cast<uint32_t>(mmInfo.textKvLen);
}

void FillTilingOffsetParam(int32_t seqIdx, AddrOffsets &addrOffsets, uint32_t *tilingParam)
{
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX5] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX6] = GetHigh32Bit(addrOffsets.addrKSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX7] = GetLoww32Bit(addrOffsets.addrKSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX8] = GetHigh32Bit(addrOffsets.addrVSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX9] = GetLoww32Bit(addrOffsets.addrVSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX10] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX11] = GetLoww32Bit(addrOffsets.addrOSeqOffset);
}

void InitTilingKWithN(const UnpadFlashAttentionInfo &mmInfo, int32_t &embeddingSizeAligned, int32_t &nIbd,
    int32_t seqIdx, uint32_t *tilingParam)
{
    int32_t tilingK = embeddingSizeAligned < LONG_SEQ_LEN ? LONG_SEQ_LEN : embeddingSizeAligned;
    int32_t kvSeqlen = *(mmInfo.kvSeq + seqIdx);
    int32_t kvSeqlenAligned = (kvSeqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    if (mmInfo.quantType == static_cast<uint32_t>(OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_OFFLINE)
        || mmInfo.quantType == static_cast<uint32_t>(OpParam::UnpadFlashAttention::QuantType::TYPE_QUANT_QKV_ONLINE)) {
            embeddingSizeAligned = static_cast<int32_t>(Mki::Utils::RoundUp(embeddingSizeAligned, BLOCK_SIZE_INT8));
    }
    int32_t nUbd = mmInfo.isMLA
                ? std::min(LONG_SEQ_LEN, kvSeqlenAligned)
                : std::min((PP_BLOCK_BUFFER_SIZE / tilingK / BLOCK_SIZE) * BLOCK_SIZE, kvSeqlenAligned);
    nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX3] = static_cast<uint32_t>(PP_NN[nIbd]);
}

Status EncoderFillTilingParam(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                              int32_t kvRealHeads, uint32_t *tilingParam)
{
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batchSize; seqIdx++) {
        int32_t qSeqlen = *(mmInfo.qSeq + seqIdx);
        int32_t qSeqlenAligned = (qSeqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvSeqlen = *(mmInfo.kvSeq + seqIdx);
        int32_t batchState = *(mmInfo.batchRunStatus + seqIdx);
        int32_t embeddingSizeAligned = (mmInfo.embeddingSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t nIbd = 0;
        InitTilingKWithN(mmInfo, embeddingSizeAligned, nIbd, seqIdx, tilingParam);
        int32_t mUbd =
            mmInfo.isMLA ? std::min(LONG_SEQ_LEN, qSeqlenAligned)
                         : std::min((PP_BLOCK_BUFFER_SIZE / std::max(embeddingSizeAligned, PP_NN[nIbd]) / BLOCK_SIZE) *
                                        BLOCK_SIZE,
                                    qSeqlenAligned);
        if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND) {
            mUbd = mmInfo.isMLA ? std::min(LONG_SEQ_LEN, qSeqlenAligned)
                : std::min((PP_BLOCK_BUFFER_SIZE / std::max(embeddingSizeAligned, mmInfo.blockSize) / BLOCK_SIZE) *
                    BLOCK_SIZE, qSeqlenAligned);
            uint32_t blockSize = static_cast<uint32_t>(mmInfo.blockSize);
            tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX3] = blockSize;
        }
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);

        mUbd = mmInfo.splitm ? PP_NN[INDEX15] : PP_MM[mIbd];
        if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_RAZOR_FUSION) {
            uint32_t razorBlkNum =  (qSeqlen != 0 && kvSeqlen != 0) ? ((mmInfo.razorLen + PP_MM[mIbd] - 1) /
                PP_MM[mIbd]) * mmInfo.tileQ : 0;
            uint32_t textBlKNum =  (qSeqlen != 0 && kvSeqlen != 0) ? ((mmInfo.textQLen + PP_MM[mIbd] - 1) /
                PP_MM[mIbd]) : 0;
            addrOffsets.totalQBlkNum += razorBlkNum + textBlKNum;
            tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX3] = static_cast<uint32_t>(PP_NN[nIbd]);
        } else {
            addrOffsets.totalQBlkNum += (qSeqlen != 0 && kvSeqlen != 0) ? ((qSeqlen + mUbd - 1) / mUbd) : 0;
        }
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX2] = static_cast<uint32_t>(mUbd);
        Status ret1 = CheckSeqlen(mmInfo, addrOffsets, qSeqlen, kvSeqlen, seqIdx);
        OP_TILING_CHECK_STATUS_RETURN(ret1);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE] = static_cast<uint32_t>(qSeqlen);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + 1] = static_cast<uint32_t>(kvSeqlen);
        
        FillTilingOffsetParam(seqIdx, addrOffsets, tilingParam);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX13] =
            static_cast<uint32_t>(addrOffsets.totalQBlkNum);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX14] = static_cast<uint32_t>(batchState);
        auto kvFactor = (mmInfo.isNoCache) ? kvSeqlen : mmInfo.maxKvSeqLen;
        FillAddrOffsets(mmInfo, addrOffsets, kvRealHeads, qSeqlen, kvFactor);
    }
    FillTilingHead(mmInfo, torUptr, addrOffsets, tilingParam, kvRealHeads);
    addrOffsets.block = static_cast<int64_t>(mmInfo.innerBatchSize) * addrOffsets.totalQBlkNum;
    return Status::OkStatus();
}

void FillAddrOffsets(const AtbOps::UnpadFlashAttentionInfo &mmInfo, AtbOps::AddrOffsets &addrOffsets,
                     int32_t kvRealHeads, int32_t qSeqlen, int32_t kvFactor)
{
    if (mmInfo.dataShapeType == 1) {
        MKI_LOG(INFO) << "addrOffsets mmInfo.maxSeqLen" << mmInfo.maxSeqLen << "mmInfo.innerBatchSize"
                      << mmInfo.innerBatchSize << "mmInfo.embeddingSize" << mmInfo.embeddingSize;
        MKI_LOG(INFO) << "addrOffsets mmInfo.maxKvSeqLen" << mmInfo.maxKvSeqLen << "kvRealHeads"
                      << mmInfo.innerBatchSize << "mmInfo.embeddingSize" << mmInfo.embeddingSize;
        addrOffsets.addrQSeqOffset +=
            static_cast<uint64_t>(mmInfo.maxQSeqLen) * mmInfo.innerBatchSize * mmInfo.embeddingSize;
        addrOffsets.addrKSeqOffset += static_cast<uint64_t>(mmInfo.maxKvSeqLen) * kvRealHeads * mmInfo.embeddingSize;
        addrOffsets.addrVSeqOffset += static_cast<uint64_t>(mmInfo.maxKvSeqLen) * kvRealHeads * mmInfo.embeddingSizeV;
        addrOffsets.addrOSeqOffset +=
            static_cast<uint64_t>(mmInfo.maxQSeqLen) * mmInfo.innerBatchSize * mmInfo.embeddingSizeV;
    } else {
        addrOffsets.addrQSeqOffset += static_cast<uint64_t>(qSeqlen) * mmInfo.innerBatchSize * mmInfo.embeddingSize;
        addrOffsets.addrKSeqOffset += static_cast<uint64_t>(kvFactor) * kvRealHeads * mmInfo.embeddingSize;
        addrOffsets.addrVSeqOffset += static_cast<uint64_t>(kvFactor) * kvRealHeads * mmInfo.embeddingSizeV;
        addrOffsets.addrOSeqOffset += static_cast<uint64_t>(qSeqlen) * mmInfo.innerBatchSize * mmInfo.embeddingSizeV;
    }
}

void FillSplitBatchPtr(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, int32_t seqIdx)
{
    uint64_t kCachePtr = reinterpret_cast<uint64_t>(mmInfo.kTensorList[seqIdx].data);
    uint64_t vCachePtr = reinterpret_cast<uint64_t>(mmInfo.vTensorList[seqIdx].data);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX13] = GetHigh32Bit(kCachePtr);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX14] = GetLoww32Bit(kCachePtr);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX15] = GetHigh32Bit(vCachePtr);
    tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX16] = GetLoww32Bit(vCachePtr);
}

void FillSplitBatchPtr(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, int32_t nowBatchTiling,
                       int32_t seqIdx, int32_t shareSeqIdx)
{
    shareSeqIdx = shareSeqIdx == -1 ? 0 : shareSeqIdx;
    uint64_t kCachePtr = reinterpret_cast<uint64_t>(mmInfo.kTensorList[seqIdx].data);
    uint64_t vCachePtr = reinterpret_cast<uint64_t>(mmInfo.vTensorList[seqIdx].data);
    uint64_t kShareCachePtr = reinterpret_cast<uint64_t>(mmInfo.kShareTensorList[shareSeqIdx].data);
    uint64_t vShareCachePtr = reinterpret_cast<uint64_t>(mmInfo.vShareTensorList[shareSeqIdx].data);
    tilingParam[nowBatchTiling + INDEX6] = GetHigh32Bit(kCachePtr);
    tilingParam[nowBatchTiling + INDEX7] = GetLoww32Bit(kCachePtr);
    tilingParam[nowBatchTiling + INDEX8] = GetHigh32Bit(vCachePtr);
    tilingParam[nowBatchTiling + INDEX9] = GetLoww32Bit(vCachePtr);
    tilingParam[nowBatchTiling + INDEX14] = GetHigh32Bit(kShareCachePtr);
    tilingParam[nowBatchTiling + INDEX15] = GetLoww32Bit(kShareCachePtr);
    tilingParam[nowBatchTiling + INDEX16] = GetHigh32Bit(vShareCachePtr);
    tilingParam[nowBatchTiling + INDEX17] = GetLoww32Bit(vShareCachePtr);
}

void SplitTaskRelay(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, int32_t groupNum,
                    uint32_t blockIdx, uint32_t shareBlockTiling)
{
    int32_t taskHeadStart = tilingParam[TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling];
    int32_t taskHeadEnd = tilingParam[TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling + 1];
    int32_t curBatch = taskHeadStart % (mmInfo.batchSize * groupNum) / groupNum;
    int32_t curHeadId = taskHeadStart / (mmInfo.batchSize * groupNum) * groupNum;
    uint32_t shareIndex = TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling + INDEX3;
    uint32_t unshareIndex = TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling + shareBlockTiling / 2 + INDEX3;
    int totalShareNum = 0;
    for (int32_t headId = curHeadId; headId < mmInfo.innerBatchSize; headId += groupNum) {
        int32_t nowBatch = headId == curHeadId ? curBatch : 0;
        if (mmInfo.batchSize * groupNum * (headId / groupNum) + nowBatch * groupNum >= taskHeadEnd) {
            break;
        }
        uint64_t offsetTiling = TILING_RELAY_HEAD_SIZE +
                                shareBlockTiling * mmInfo.blockDim + TILING_PARA_SIZE * nowBatch;
        uint32_t nowShare = tilingParam[INDEX18 + offsetTiling];
        uint32_t qLen = 0;
        while (mmInfo.batchSize * groupNum * (headId / groupNum) +
               nowBatch * groupNum < taskHeadEnd && nowBatch < mmInfo.batchSize) {
            uint32_t shareIdx = tilingParam[INDEX18 + offsetTiling];
            tilingParam[unshareIndex] = nowBatch;
            tilingParam[unshareIndex + 1] = headId;
            unshareIndex += INDEX4;
            if (shareIdx == static_cast<uint32_t>(-1)) {
                nowBatch++;
                offsetTiling += TILING_PARA_SIZE;
                continue;
            }
            if (shareIdx != nowShare) {
                nowShare = shareIdx;
                qLen = 0;
            }
            tilingParam[shareIndex] = nowBatch;
            tilingParam[shareIndex + 1] = headId;
            tilingParam[shareIndex + INDEX2] = shareIdx;
            tilingParam[shareIndex + INDEX3] = 0;
            if (qLen == 0 && (mmInfo.batchSize * groupNum * (headId / groupNum) +
                nowBatch * groupNum + groupNum >= taskHeadEnd || nowBatch + 1 == mmInfo.batchSize ||
                tilingParam[offsetTiling + TILING_PARA_SIZE + INDEX18] != nowShare) && groupNum == 1) {
                tilingParam[shareIndex + INDEX3] = 1;
            }
            shareIndex += INDEX4;
            totalShareNum++;
            qLen = qLen + groupNum;
            nowBatch++;
            offsetTiling += TILING_PARA_SIZE;
        }
    }
    tilingParam[TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling + INDEX2] = totalShareNum;
}

void SplitCoreRelay(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, uint32_t shareBlockTiling)
{
    uint32_t taskNum = mmInfo.batchSize * mmInfo.kvHead;
    uint32_t taskNumPerCore = taskNum / mmInfo.blockDim;
    uint32_t tailTaskNum = taskNum % mmInfo.blockDim;
    int32_t groupNum = mmInfo.innerBatchSize / mmInfo.kvHead;
    int32_t taskStart = 0;
    int32_t taskEnd = 0;
    for (uint32_t blockIdx = 0; blockIdx < mmInfo.blockDim; blockIdx++) {
        taskStart = taskEnd;
        taskEnd = taskEnd + (blockIdx < tailTaskNum ? taskNumPerCore + 1 : taskNumPerCore);
        uint32_t nowBlockTiling = TILING_RELAY_HEAD_SIZE + blockIdx * shareBlockTiling;
        tilingParam[nowBlockTiling] = taskStart * groupNum;
        tilingParam[nowBlockTiling + 1] = taskEnd * groupNum;
        if (taskStart >= taskEnd) {
            continue;
        }
        SplitTaskRelay(mmInfo, tilingParam, groupNum, blockIdx, shareBlockTiling);
    }
}

void DecoderSplitHeadNum(const UnpadFlashAttentionInfo &mmInfo, int32_t kvRealHeads, uint32_t *tilingParam)
{
    if (mmInfo.embeddingSize % INDEX16 == 0 && mmInfo.embeddingSize <= EMBEDINF_LIMIT &&
        kvRealHeads == mmInfo.innerBatchSize) {
        tilingParam[INDEX12] = 2; // 2 improve bandwidth utilization
    } else {
        tilingParam[INDEX12] = 1;
    }
    if (mmInfo.isMLA) {
        tilingParam[INDEX12] = 1;
    }
    MKI_LOG(INFO) << "headnum_split :" << tilingParam[INDEX12];
}

void DecoderSplitCore(const UnpadFlashAttentionInfo &mmInfo, AddrOffsets &addrOffsets, int32_t kvRealHeads,
                      uint32_t *tilingParam)
{
    int32_t coreNumPerBatch = (mmInfo.blockDim + mmInfo.batchSize - 1) / mmInfo.batchSize;
    int32_t headSplit = (mmInfo.innerBatchSize + coreNumPerBatch - 1) / coreNumPerBatch;
    if (mmInfo.batchSize >= SPECIALNUMBATCH && mmInfo.innerBatchSize >= SPECIALNUMHEADS) {
        headSplit = 8; // 8 performs better
    }
    if (headSplit > 1) {
        headSplit = (headSplit + 1) / NUM_TWO * NUM_TWO;
    }
    int32_t loopLen = (mmInfo.innerBatchSize + headSplit - 1) / headSplit;
    addrOffsets.block = static_cast<int64_t>(loopLen) * mmInfo.batchSize;
    uint32_t formerBatch = mmInfo.batchSize;
    uint32_t tailBatch = 0;
    uint32_t formerHeadSplit = static_cast<uint32_t>(headSplit);
    uint32_t tailHeadSplit = 0;
    // 调整最后一轮的负载
    if (addrOffsets.block > mmInfo.blockDim) {
        uint32_t processLoop = static_cast<uint32_t>(addrOffsets.block / static_cast<int64_t>(mmInfo.blockDim));
        formerBatch = processLoop * static_cast<uint32_t>(mmInfo.blockDim / loopLen);
        tailBatch = mmInfo.batchSize - formerBatch;
        uint32_t processRemain = tailBatch * static_cast<uint32_t>(loopLen);
        // 当最后一轮任务数小于16时调整
        if (processRemain < SPECIALNUMBATCH && tailBatch > 0) {
            coreNumPerBatch = (mmInfo.blockDim + static_cast<int32_t>(tailBatch) - 1) / static_cast<int32_t>(tailBatch);
            tailHeadSplit = static_cast<uint32_t>((mmInfo.innerBatchSize + coreNumPerBatch - 1) / coreNumPerBatch);
            if (tailHeadSplit > 1) {
                tailHeadSplit = (tailHeadSplit + 1) / NUM_TWO * NUM_TWO;
            }
        } else {
            formerBatch = mmInfo.batchSize;
            tailBatch = 0;
        }
    }
    MKI_LOG(INFO) << "formerBatch :" << formerBatch << " ,formerHeadSplit :" << formerHeadSplit
                  << " ,tailBatch :" << tailBatch << " ,tailHeadSplit :" << tailHeadSplit;
    tilingParam[INDEX7] = static_cast<uint32_t>(formerBatch);
    tilingParam[INDEX8] = static_cast<uint32_t>(formerHeadSplit);
    tilingParam[INDEX9] = static_cast<uint32_t>(tailBatch);
    tilingParam[INDEX10] = static_cast<uint32_t>(tailHeadSplit);
    tilingParam[INDEX11] = static_cast<uint32_t>(mmInfo.maskStride);
    DecoderSplitHeadNum(mmInfo, kvRealHeads, tilingParam);
}

void FillTilingHeadDecoder(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                           int32_t kvRealHeads, uint32_t *tilingParam)
{
    tilingParam[0] = static_cast<uint32_t>(mmInfo.batchSize);
    tilingParam[1] = static_cast<uint32_t>(mmInfo.maxSeqLen);
    tilingParam[INDEX2] = static_cast<uint32_t>(mmInfo.innerBatchSize);
    tilingParam[INDEX3] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[INDEX4] = static_cast<uint32_t>(kvRealHeads);
    tilingParam[INDEX5] = torUptr;
    tilingParam[INDEX6] = static_cast<uint32_t>(mmInfo.headStride);
    DecoderSplitCore(mmInfo, addrOffsets, kvRealHeads, tilingParam);
    tilingParam[INDEX13] = SetTilingKey(mmInfo);
    tilingParam[INDEX14] = static_cast<uint32_t>(TILING_HEAD_SIZE);
    tilingParam[INDEX15] = static_cast<uint32_t>(TILING_PARA_SIZE);
    tilingParam[INDEX16] = static_cast<uint32_t>(mmInfo.isClamp);
    float clampMin = mmInfo.clampMin;
    float clampMax = mmInfo.clampMax;
    tilingParam[INDEX17] = *static_cast<uint32_t *>(static_cast<void *>(&clampMin));
    tilingParam[INDEX18] = *static_cast<uint32_t *>(static_cast<void *>(&clampMax));
    tilingParam[INDEX19] = static_cast<uint32_t>(mmInfo.maxKvSeqLen);
    tilingParam[INDEX20] = static_cast<uint32_t>(mmInfo.batchContinuous);
    tilingParam[INDEX21] = static_cast<uint32_t>(mmInfo.embeddingSizeV);
    tilingParam[INDEX22] =  mmInfo.scaleType;
    tilingParam[INDEX23] = static_cast<uint32_t>(mmInfo.maskType);
    tilingParam[INDEX24] = static_cast<uint32_t>(mmInfo.dataShapeType);
    MKI_LOG(INFO) << "dataShapeType : " << tilingParam[INDEX24];
    tilingParam[INDEX25] = static_cast<uint32_t>(mmInfo.windowSize);
    tilingParam[INDEX26] = static_cast<uint32_t>(mmInfo.cacheType);
}

Status DecoderFillTilingParamRelay(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr,
                                   AddrOffsets &addrOffsets, int32_t kvRealHeads, uint32_t *tilingParam)
{
    FillTilingHeadDecoder(mmInfo, torUptr, addrOffsets, kvRealHeads, tilingParam);
    uint32_t nowBatch = 0;
    uint32_t shareBlockTiling = ((mmInfo.batchSize * mmInfo.kvHead + mmInfo.blockDim - 1) /
                                  mmInfo.blockDim + 1) * RELAY_BLOCK_TILING;
    tilingParam[INDEX28] = static_cast<uint32_t>(mmInfo.innerBatchSize / mmInfo.kvHead);
    tilingParam[INDEX29] = shareBlockTiling;
    tilingParam[INDEX30] = TILING_RELAY_HEAD_SIZE;
    tilingParam[INDEX14] = static_cast<uint32_t>(mmInfo.blockDim * shareBlockTiling + TILING_RELAY_HEAD_SIZE);
    for (auto &it: mmInfo.batchShareMap) {
        uint32_t shareLen = it.first != -1 ? mmInfo.kvShareLen[it.first] : 0;
        for (auto seqIdx: it.second) {
            int32_t qSeqlen = *(mmInfo.qSeq + seqIdx);
            int32_t kvSeqlen = *(mmInfo.kvSeq + seqIdx);
            int32_t kvSeqlenAligned = (kvSeqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            int32_t embeddingSizeAligned = (mmInfo.embeddingSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            int32_t nUbd = std::min((DECODER_PP_BLOCK_BUFFER_SIZE * INDEX2 / embeddingSizeAligned / BLOCK_SIZE) *
                                    BLOCK_SIZE, kvSeqlenAligned);
            int32_t nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
            OP_TILING_CHECK_STATUS_RETURN(CheckSeqlen(mmInfo, addrOffsets, qSeqlen, kvSeqlen, seqIdx));
            addrOffsets.addrQSeqOffset = static_cast<uint64_t>(seqIdx * qSeqlen *
                                         mmInfo.innerBatchSize * mmInfo.embeddingSize);
            addrOffsets.addrOSeqOffset = static_cast<uint64_t>(seqIdx * qSeqlen *
                                         mmInfo.innerBatchSize * mmInfo.embeddingSize);
            addrOffsets.addrLSeqOffset = seqIdx * mmInfo.innerBatchSize * INDEX2;
            addrOffsets.addrOFdSeqOffset = seqIdx * static_cast<uint64_t>(mmInfo.innerBatchSize *
                                            mmInfo.embeddingSize * qSeqlen) * INDEX2;
            uint32_t nowBatchTiling = TILING_RELAY_HEAD_SIZE + shareBlockTiling * mmInfo.blockDim +
                                      nowBatch * TILING_PARA_SIZE;
            tilingParam[nowBatchTiling] = static_cast<uint32_t>(qSeqlen);
            tilingParam[nowBatchTiling + 1] = static_cast<uint32_t>(kvSeqlen);
            tilingParam[nowBatchTiling + INDEX3] = static_cast<uint32_t>(PP_NN[nIbd]);
            tilingParam[nowBatchTiling + INDEX4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
            tilingParam[nowBatchTiling + INDEX5] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
            tilingParam[nowBatchTiling + INDEX10] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
            tilingParam[nowBatchTiling + INDEX11] = GetLoww32Bit(addrOffsets.addrOSeqOffset);
            tilingParam[nowBatchTiling + INDEX13] = static_cast<uint32_t>(shareLen);
            tilingParam[nowBatchTiling + INDEX18] = static_cast<uint32_t>(it.first);
            tilingParam[nowBatchTiling + INDEX19] = seqIdx;
            tilingParam[nowBatchTiling + INDEX20] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
            tilingParam[nowBatchTiling + INDEX21] = GetLoww32Bit(addrOffsets.addrLSeqOffset);
            tilingParam[nowBatchTiling + INDEX22] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
            tilingParam[nowBatchTiling + INDEX23] = GetLoww32Bit(addrOffsets.addrOFdSeqOffset);
            FillSplitBatchPtr(mmInfo, tilingParam, nowBatchTiling, seqIdx, it.first);
            nowBatch++;
        }
    }
    SplitCoreRelay(mmInfo, tilingParam, shareBlockTiling);
    return Status::OkStatus();
}

Status DecoderFillTilingParam(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                              int32_t kvRealHeads, uint32_t *tilingParam)
{
    FillTilingHeadDecoder(mmInfo, torUptr, addrOffsets, kvRealHeads, tilingParam);
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batchSize; seqIdx++) {
        int32_t batchState = *(mmInfo.batchRunStatus + seqIdx);
        int32_t qSeqlen = *(mmInfo.qSeq + seqIdx);
        int32_t qSeqlenAligned = (qSeqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvSeqlen = *(mmInfo.kvSeq + seqIdx);
        int32_t kvSeqlenAligned = (kvSeqlen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t embeddingSizeAligned = (mmInfo.embeddingSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t nUbd = mmInfo.isMLA
                           ? std::min(LONG_SEQ_LEN, kvSeqlenAligned)
                           : std::min((DECODER_PP_BLOCK_BUFFER_SIZE / embeddingSizeAligned / BLOCK_SIZE) * BLOCK_SIZE,
                                      kvSeqlenAligned);
        int32_t nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
        int32_t mUbd = mmInfo.isMLA ? std::min(LONG_SEQ_LEN, qSeqlenAligned) :
            std::min((DECODER_PP_BLOCK_BUFFER_SIZE  / std::max(embeddingSizeAligned,
            PP_NN[nIbd]) / BLOCK_SIZE) * BLOCK_SIZE, qSeqlenAligned);
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
        int32_t curQBlockNum = ((qSeqlen + PP_MM[mIbd] - 1) / PP_MM[mIbd]);
        addrOffsets.totalQBlkNum += curQBlockNum;
        Status ret1 = CheckSeqlen(mmInfo, addrOffsets, qSeqlen, kvSeqlen, seqIdx);
        OP_TILING_CHECK_STATUS_RETURN(ret1);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE] = static_cast<uint32_t>(qSeqlen);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + 1] = static_cast<uint32_t>(kvSeqlen);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX2] = static_cast<uint32_t>(PP_MM[mIbd]);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX3] = static_cast<uint32_t>(PP_NN[nIbd]);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX5] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX6] =
            mmInfo.batchContinuous ? GetHigh32Bit(addrOffsets.addrKSeqOffset) : 0;
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX7] =
            mmInfo.batchContinuous ? GetLoww32Bit(addrOffsets.addrKSeqOffset) : 0;
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX8] =
            mmInfo.batchContinuous ? GetHigh32Bit(addrOffsets.addrVSeqOffset) : 0;
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX9] =
            mmInfo.batchContinuous ? GetLoww32Bit(addrOffsets.addrVSeqOffset) : 0;
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX10] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX11] = GetLoww32Bit(addrOffsets.addrOSeqOffset);
        tilingParam[TILING_HEAD_SIZE + seqIdx * TILING_PARA_SIZE + INDEX12] = static_cast<uint32_t>(batchState);
        if (!mmInfo.batchContinuous) {
            FillSplitBatchPtr(mmInfo, tilingParam, seqIdx);
        }
        MKI_LOG(INFO) << "dataShapeType :" << mmInfo.dataShapeType << " ,maxSeqLen :" << mmInfo.maxSeqLen
                      << " ,innerBatchSize :" << mmInfo.innerBatchSize << " ,embeddingSize :" << mmInfo.embeddingSize
                      << " ,embeddingSizeV :" << mmInfo.embeddingSizeV << ",maxKvSeqLen" << mmInfo.maxKvSeqLen
                      << ",kvRealHeads" << kvRealHeads;
 
        FillAddr(addrOffsets, qSeqlen, mmInfo, kvRealHeads);
    }
    return Status::OkStatus();
}

void FillAddr(AtbOps::AddrOffsets &addrOffsets, int32_t qSeqlen, const AtbOps::UnpadFlashAttentionInfo &mmInfo,
              int32_t kvRealHeads)
{
    addrOffsets.addrQSeqOffset += static_cast<uint64_t>(qSeqlen * mmInfo.innerBatchSize * mmInfo.embeddingSize);
    addrOffsets.addrKSeqOffset += static_cast<uint64_t>(mmInfo.maxKvSeqLen * kvRealHeads * mmInfo.embeddingSize);
    addrOffsets.addrVSeqOffset += static_cast<uint64_t>(mmInfo.maxKvSeqLen * kvRealHeads * mmInfo.embeddingSizeV);
    addrOffsets.addrOSeqOffset += static_cast<uint64_t>(qSeqlen * mmInfo.innerBatchSize * mmInfo.embeddingSizeV);
}

Status GetFillTilingParam(const UnpadFlashAttentionInfo &mmInfo,
                          uint32_t *tilingParam, AddrOffsets &addrOffsets)
{
    float tor = mmInfo.tor;
    uint32_t *torUptr = reinterpret_cast<uint32_t *>(&tor);
    int32_t kvRealHeads = mmInfo.kvHead > 0 ? mmInfo.kvHead : mmInfo.innerBatchSize;
    if (mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND ||
        mmInfo.type == OpParam::UnpadFlashAttention::UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER) {
        Status ret1 = DecoderFillTilingParam(mmInfo, *torUptr, addrOffsets, kvRealHeads, tilingParam);
        OP_TILING_CHECK_STATUS_RETURN(ret1);
    } else if (mmInfo.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        Status ret1 = DecoderFillTilingParamRelay(mmInfo, *torUptr, addrOffsets, kvRealHeads, tilingParam);
        OP_TILING_CHECK_STATUS_RETURN(ret1);
    } else {
        Status ret1 = EncoderFillTilingParam(mmInfo, *torUptr, addrOffsets, kvRealHeads, tilingParam);
        OP_TILING_CHECK_STATUS_RETURN(ret1);
    }
    return Status::OkStatus();
}
Status GetUnpadFlashAttentionTilingParam(const UnpadFlashAttentionInfo &mmInfo, uint32_t &blockDim,
                                         uint32_t *tilingParam, uint32_t tilingParamSize)
{
    if (tilingParam == nullptr) {
        MKI_LOG(ERROR) << "pointer tilingParam or seq is nullptr.";
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingParam is nullptr");
    }
    uint32_t tilingHeadSize = mmInfo.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND ?
                        TILING_RELAY_HEAD_SIZE : TILING_HEAD_SIZE;
    uint64_t initSize =
        static_cast<uint64_t>((mmInfo.batchSize * TILING_PARA_SIZE + tilingHeadSize) * sizeof(uint32_t));
    uint64_t totalSize = static_cast<uint64_t>(tilingHeadSize) +
                     static_cast<uint64_t>(mmInfo.batchSize) *
                     static_cast<uint64_t>(TILING_PARA_SIZE);
    if (mmInfo.type == OpParam::UnpadFlashAttention::RELAY_ATTENTION_DECODER_ND) {
        uint32_t shareBlockTiling = ((mmInfo.batchSize * mmInfo.kvHead + mmInfo.blockDim - 1) /
                                      mmInfo.blockDim + 1) * RELAY_BLOCK_TILING;
        initSize = static_cast<uint64_t>((mmInfo.batchSize * TILING_PARA_SIZE +
                                          shareBlockTiling * mmInfo.blockDim + tilingHeadSize) * sizeof(uint32_t));
        totalSize = static_cast<uint64_t>(tilingHeadSize) +
                     static_cast<uint64_t>(mmInfo.batchSize * TILING_PARA_SIZE) +
                     static_cast<uint64_t>(shareBlockTiling * mmInfo.blockDim);
    }
    auto ret = memset_s(tilingParam, tilingParamSize, 0, initSize);
    MKI_CHECK(ret == EOK, "Failed to clear the array", return Status::FailStatus(-1));
    AddrOffsets addrOffsets;
    Status ret1 = GetFillTilingParam(mmInfo, tilingParam, addrOffsets);
    OP_TILING_CHECK_STATUS_RETURN(ret1);
    MKI_LOG(INFO) << "tiling is";
    MKI_CHECK(static_cast<int64_t>(totalSize) <= std::numeric_limits<int32_t>::max(), "batch is too large",
        return Status::FailStatus(ERROR_INVALID_VALUE));

    for (uint32_t i = 0; i < static_cast<uint32_t>(totalSize); i++) {
        if (i == (static_cast<uint32_t>(totalSize) - 1)) {
            MKI_LOG(INFO) << tilingParam[i] << ")";
            break;
        }
        MKI_LOG(INFO) << i << " index: " << tilingParam[i] << ",";
    }
    MKI_CHECK(addrOffsets.block > 0 && addrOffsets.block <= UINT32_MAX, "blockDim overflow",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t logicalblock = static_cast<uint32_t>(addrOffsets.block);
    MKI_LOG(INFO) << "totalQBlkNum is " << addrOffsets.totalQBlkNum;
    MKI_LOG(INFO) << "block is " << logicalblock;
    if (blockDim > logicalblock) {
        blockDim = logicalblock;
    }
    MKI_LOG(INFO) << "blockDim is " << blockDim;
    return Status::OkStatus();
}
} // namespace AtbOps
