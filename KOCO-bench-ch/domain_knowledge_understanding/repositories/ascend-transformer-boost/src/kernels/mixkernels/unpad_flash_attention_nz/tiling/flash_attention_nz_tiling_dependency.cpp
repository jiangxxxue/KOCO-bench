/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "flash_attention_nz_tiling_dependency.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <securec.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/types.h>
#include "mki/utils/platform/platform_info.h"

namespace AtbOps {
using namespace Mki;
const int32_t PP_MM_NUM = 8;
const uint32_t CORE_NUM_LIMIT = 10;
const int32_t BIT_SHIFT = 32;
const int32_t NZ_INDEX_0 = 0;
const int32_t NZ_INDEX_1 = 1;
const int32_t NZ_INDEX_2 = 2;
const int32_t NZ_INDEX_3 = 3;
const int32_t NZ_INDEX_4 = 4;
const int32_t NZ_INDEX_5 = 5;
const int32_t NZ_INDEX_6 = 6;
const int32_t NZ_INDEX_7 = 7;
const int32_t NZ_INDEX_8 = 8;
const int32_t NZ_INDEX_9 = 9;
const int32_t NZ_INDEX_10 = 10;
const int32_t NZ_INDEX_11 = 11;
const int32_t NZ_INDEX_12 = 12;
const int32_t NZ_INDEX_13 = 13;
const int32_t NZ_INDEX_14 = 14;
const int32_t NZ_INDEX_15 = 15;
const int32_t NZ_INDEX_16 = 16;
const int32_t NZ_INDEX_17 = 17;
const int32_t NZ_INDEX_18 = 18;
const int32_t NZ_INDEX_19 = 19;
const int32_t NZ_INDEX_20 = 20;
const int32_t NZ_INDEX_21 = 21;
const int32_t NZ_INDEX_22 = 22;
const int32_t NZ_INDEX_23 = 23;
const int32_t NZ_INDEX_48 = 48;
const int32_t NZ_INDEX_49 = 49;
const int32_t NZ_INDEX_117 = 117;
const int32_t NZ_INDEX_118 = 118;
const int32_t NZ_INDEX_119 = 119;
const int32_t NZ_INDEX_120 = 120;
const int32_t NZ_INDEX_121 = 121;
const int32_t NZ_INDEX_122 = 122;
const int32_t NZ_INDEX_123 = 123;
const int32_t NZ_INDEX_124 = 124;
const int32_t NZ_INDEX_125 = 125;
const int32_t NZ_INDEX_126 = 126;
const int32_t NZ_INDEX_127 = 127;

// Tiling Index For 910A
const int32_t NZ_INDEX_141 = 141;
const int32_t NZ_INDEX_182 = 182;
const int32_t NZ_INDEX_183 = 183;
const int32_t NZ_INDEX_184 = 184;
const int32_t NZ_INDEX_185 = 185;
const int32_t NZ_INDEX_186 = 186;
const int32_t NZ_INDEX_187 = 187;
const int32_t NZ_INDEX_188 = 188;
const int32_t NZ_INDEX_189 = 189;
const int32_t NZ_INDEX_190 = 190;
const int32_t NZ_INDEX_191 = 191;

const int32_t BLOCK_LIMIT = 128;
static const std::unordered_map<PlatformType, int> TILING_OFFSET_MAP{
    {PlatformType::ASCEND_910A, 192},
    {PlatformType::ASCEND_310P, 128},
};
static const std::unordered_map<PlatformType, uint32_t> TILING_KEY_INDEX_MAP{
    {PlatformType::ASCEND_910A, 180},
    {PlatformType::ASCEND_310P, 49},
};

int32_t GetNzRealCoreTilingOffset() { return TILING_OFFSET_MAP.at(PlatformInfo::Instance().GetPlatformType()); }
uint32_t GetTilingKeyIndex() { return TILING_KEY_INDEX_MAP.at(PlatformInfo::Instance().GetPlatformType()); }

constexpr std::array<int32_t, PP_MM_NUM> PP_MM = {16, 32, 48, 64, 80, 96, 112, 128};
inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / NZ_BLOCK_SIZE - 1);
}
const int32_t PP_NN_NUM = 16;
constexpr std::array<int32_t, PP_NN_NUM> PP_NN = {16,  32,  48,  64,  80,  96,  112, 128,
                                                  144, 160, 176, 192, 208, 224, 240, 256};
//                                    index:  0,  1,  2,  3,  4,  5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15
inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound)
{
    return (val > PP_NN[idxBound]) ? idxBound : (val / NZ_BLOCK_SIZE - 1);
}

const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;

inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> BIT_SHIFT); }
inline uint32_t GetLow32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

void FillSplitBatchPtr(const UnpadFlashAttentionNzInfo &mmInfo, int32_t batchOffset, int32_t seqIdx,
                       uint32_t *tilingParam)
{
    uint64_t kCachePtr = reinterpret_cast<uint64_t>(mmInfo.kTensorList[seqIdx].data);
    uint64_t vCachePtr = reinterpret_cast<uint64_t>(mmInfo.vTensorList[seqIdx].data);
    tilingParam[batchOffset + NZ_INDEX_16] = GetHigh32Bit(kCachePtr);
    tilingParam[batchOffset + NZ_INDEX_17] = GetLow32Bit(kCachePtr);
    tilingParam[batchOffset + NZ_INDEX_18] = GetHigh32Bit(vCachePtr);
    tilingParam[batchOffset + NZ_INDEX_19] = GetLow32Bit(vCachePtr);
}

void GetTilingKey(UnpadFlashAttentionNzInfo mmInfo, uint32_t *tilingParam)
{
    if (mmInfo.type != OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER) {
        // tiling key
        tilingParam[GetTilingKeyIndex()] = (mmInfo.precType << NZ_INDEX_3);
    }
    MKI_LOG(INFO) << "tilingKey: " << tilingParam[GetTilingKeyIndex()];
}

Status FillTilingParamRealCore(const UnpadFlashAttentionNzInfo &mmInfo, const uint32_t &torUptr,
                               AddrOffsetsNz &addrOffsets, uint32_t *tilingParam, const uint32_t nzRealCoreNum)
{
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batchSize; seqIdx++) {
        int32_t qSeqLen = *(mmInfo.qSeq + seqIdx);
        int32_t qSeqLenAligned = (qSeqLen + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
        int32_t kvSeqLen = *(mmInfo.kvSeq + seqIdx);
        int32_t kvSeqLenAligned = (kvSeqLen + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
        int32_t tilingK = mmInfo.embeddingSize < BLOCK_LIMIT ? BLOCK_LIMIT : mmInfo.embeddingSize;
        int32_t nUbd = std::min((PP_BLOCK_BUFFER_SIZE / tilingK / NZ_BLOCK_SIZE) * NZ_BLOCK_SIZE,
                                kvSeqLenAligned); // hidden_size = 40, ==> n_blk=256, n_blk=64
        int32_t nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
        int32_t mUbd = std::min((PP_BLOCK_BUFFER_SIZE / std::max(mmInfo.embeddingSize, PP_NN[nIbd]) / NZ_BLOCK_SIZE) *
                                    NZ_BLOCK_SIZE,
                                qSeqLenAligned);
        mUbd =
            mUbd > PP_MM[NZ_INDEX_3] && mmInfo.precType == OpParam::UnpadFlashAttentionNz::PrecType::BMM1_FP32_EXP_FP32
                ? PP_MM[NZ_INDEX_3]
                : mUbd;
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
        int32_t curQBlockNum = ((qSeqLen + PP_MM[mIbd] - 1) / PP_MM[mIbd]); // 对齐处理
        addrOffsets.totalQBlkNum += curQBlockNum;
        MKI_CHECK(addrOffsets.totalQBlkNum > 0, "totalQBlkNum overflow",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        uint32_t currTotalProc = static_cast<uint32_t>(mmInfo.innerBatchSize * addrOffsets.totalQBlkNum);
        uint32_t batchProc = static_cast<uint32_t>(mmInfo.innerBatchSize * curQBlockNum);
        MKI_CHECK(qSeqLenAligned > 0, "qSeqLenAligned is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(kvSeqLenAligned > 0, "kvSeqLenAligned is invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
        int32_t batchOffset = seqIdx * NZ_REAL_CORE_TILING_SIZE + GetNzRealCoreTilingOffset();
        tilingParam[batchOffset] = static_cast<uint32_t>(qSeqLenAligned);
        tilingParam[batchOffset + NZ_INDEX_1] = static_cast<uint32_t>(kvSeqLenAligned);
        tilingParam[batchOffset + NZ_INDEX_2] = static_cast<uint32_t>(PP_MM[mIbd]);
        tilingParam[batchOffset + NZ_INDEX_3] = static_cast<uint32_t>(PP_NN[nIbd]);
        tilingParam[batchOffset + NZ_INDEX_4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
        tilingParam[batchOffset + NZ_INDEX_5] = GetLow32Bit(addrOffsets.addrQSeqOffset);
        tilingParam[batchOffset + NZ_INDEX_6] = mmInfo.batchContinuous ? GetHigh32Bit(addrOffsets.addrKSeqOffset) : 0;
        tilingParam[batchOffset + NZ_INDEX_7] = mmInfo.batchContinuous ? GetLow32Bit(addrOffsets.addrKSeqOffset) : 0;
        tilingParam[batchOffset + NZ_INDEX_8] = mmInfo.batchContinuous ? GetHigh32Bit(addrOffsets.addrVSeqOffset) : 0;
        tilingParam[batchOffset + NZ_INDEX_9] = mmInfo.batchContinuous ? GetLow32Bit(addrOffsets.addrVSeqOffset) : 0;
        tilingParam[batchOffset + NZ_INDEX_10] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
        tilingParam[batchOffset + NZ_INDEX_11] = GetLow32Bit(addrOffsets.addrOSeqOffset);
        tilingParam[batchOffset + NZ_INDEX_12] = currTotalProc;
        tilingParam[batchOffset + NZ_INDEX_13] = static_cast<uint32_t>(qSeqLen);
        tilingParam[batchOffset + NZ_INDEX_14] = static_cast<uint32_t>(kvSeqLen);
        tilingParam[batchOffset + NZ_INDEX_15] = batchProc;
        if (!mmInfo.batchContinuous) {
            FillSplitBatchPtr(mmInfo, batchOffset, seqIdx, tilingParam);
        }
        if (mmInfo.dataDimOrder == OpParam::UnpadFlashAttentionNz::TYPE_BNSD &&
            mmInfo.type != OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER) {
            addrOffsets.addrQSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxSeqLen * mmInfo.innerBatchSize * mmInfo.embeddingSize);
            addrOffsets.addrKSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxKVSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrVSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxKVSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrOSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxSeqLen * mmInfo.innerBatchSize * mmInfo.embeddingSize);
        } else if (mmInfo.dataDimOrder == OpParam::UnpadFlashAttentionNz::TYPE_BNSD &&
                   mmInfo.type == OpParam::UnpadFlashAttentionNz::UNPAD_FLASH_ATTENTION_NZ_DECODER) {
            uint32_t seq = 16;
            addrOffsets.addrQSeqOffset += static_cast<uint64_t>(seq * mmInfo.innerBatchSize * mmInfo.embeddingSize);
            addrOffsets.addrKSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxKVSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrVSeqOffset +=
                static_cast<uint64_t>(mmInfo.maxKVSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrOSeqOffset += static_cast<uint64_t>(seq * mmInfo.innerBatchSize * mmInfo.embeddingSize);
        } else {
            addrOffsets.addrQSeqOffset += mmInfo.qTight != 0 ? static_cast<uint64_t>(qSeqLen * NZ_BLOCK_SIZE)
                                                             : static_cast<uint64_t>(qSeqLenAligned * NZ_BLOCK_SIZE);
            addrOffsets.addrKSeqOffset +=
                !mmInfo.isCache ? static_cast<uint64_t>(qSeqLen * NZ_BLOCK_SIZE)
                                : static_cast<uint64_t>(mmInfo.maxSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrVSeqOffset +=
                !mmInfo.isCache ? static_cast<uint64_t>(qSeqLen * NZ_BLOCK_SIZE)
                                : static_cast<uint64_t>(mmInfo.maxSeqLen * mmInfo.kvHeads * mmInfo.embeddingSize);
            addrOffsets.addrOSeqOffset += mmInfo.qTight != 0 ? static_cast<uint64_t>(qSeqLen * NZ_BLOCK_SIZE)
                                                             : static_cast<uint64_t>(qSeqLenAligned * NZ_BLOCK_SIZE);
        }
        MKI_LOG(INFO) << "m tiling :" << PP_MM[mIbd] << ", n tiling: " << PP_NN[nIbd];
        MKI_LOG(INFO) << "batch_id :" << seqIdx << ", q_block_num: " << curQBlockNum;
    }
    FillCoreAttribs(mmInfo, torUptr, addrOffsets, tilingParam, nzRealCoreNum);
    return Status::OkStatus();
}

void FillMmInfoTilingParam(const UnpadFlashAttentionNzInfo &mmInfo, uint32_t *tilingParam, const uint32_t &torUptr)
{
    tilingParam[NZ_INDEX_0] = static_cast<uint32_t>(mmInfo.batchSize);
    tilingParam[NZ_INDEX_1] = static_cast<uint32_t>(mmInfo.qTokens);
    tilingParam[NZ_INDEX_2] = static_cast<uint32_t>(mmInfo.kvHeads);
    tilingParam[NZ_INDEX_3] = static_cast<uint32_t>(mmInfo.innerBatchSize);
    tilingParam[NZ_INDEX_4] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[NZ_INDEX_5] = static_cast<uint32_t>(mmInfo.maxSeqLen);
    tilingParam[NZ_INDEX_6] = torUptr;
    tilingParam[NZ_INDEX_7] = static_cast<uint32_t>(mmInfo.maskStride);
    tilingParam[NZ_INDEX_8] = static_cast<uint32_t>(mmInfo.headMaskStride);
    tilingParam[NZ_INDEX_9] = static_cast<uint32_t>(mmInfo.batchMaskStride);
    tilingParam[NZ_INDEX_10] = mmInfo.qTight;
    tilingParam[NZ_INDEX_11] = mmInfo.isTriu;
    if (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910A) {
        tilingParam[NZ_INDEX_141] = mmInfo.dataDimOrder;
        tilingParam[NZ_INDEX_182] = mmInfo.alibiLeftAlign;
        tilingParam[NZ_INDEX_183] = mmInfo.windowLen;
        tilingParam[NZ_INDEX_184] = mmInfo.cacheType;
        tilingParam[NZ_INDEX_185] = mmInfo.scaleType;
        tilingParam[NZ_INDEX_186] = mmInfo.maxKVSeqLen;
        tilingParam[NZ_INDEX_187] = static_cast<uint32_t>(mmInfo.batchContinuous);
        tilingParam[NZ_INDEX_188] = mmInfo.isAlibiMaskSqrt;
        tilingParam[NZ_INDEX_189] = mmInfo.alibiCompressOffset;
        tilingParam[NZ_INDEX_190] = mmInfo.maskType;
        tilingParam[NZ_INDEX_191] = mmInfo.isLongSeq;
    } else {
        tilingParam[NZ_INDEX_48] = mmInfo.dataDimOrder;
        tilingParam[NZ_INDEX_118] = mmInfo.alibiLeftAlign;
        tilingParam[NZ_INDEX_119] = mmInfo.windowLen;
        tilingParam[NZ_INDEX_120] = mmInfo.cacheType;
        tilingParam[NZ_INDEX_121] = mmInfo.scaleType;
        tilingParam[NZ_INDEX_122] = mmInfo.maxKVSeqLen;
        tilingParam[NZ_INDEX_123] = static_cast<uint32_t>(mmInfo.batchContinuous);
        tilingParam[NZ_INDEX_124] = mmInfo.isAlibiMaskSqrt;
        tilingParam[NZ_INDEX_125] = mmInfo.alibiCompressOffset;
        tilingParam[NZ_INDEX_126] = mmInfo.maskType;
        tilingParam[NZ_INDEX_127] = mmInfo.isLongSeq;
    }
}

void FillCoreAttribs(const UnpadFlashAttentionNzInfo &mmInfo, const uint32_t &torUptr, const AddrOffsetsNz &addrOffsets,
                     uint32_t *tilingParam, const uint32_t nzRealCoreNum)
{
    FillMmInfoTilingParam(mmInfo, tilingParam, torUptr);
    uint32_t procNum = static_cast<uint32_t>((mmInfo.innerBatchSize * addrOffsets.totalQBlkNum));
    uint32_t procPerBlk = procNum / nzRealCoreNum;
    uint32_t tailBlks = procNum % nzRealCoreNum;
    int32_t startBatch = 0;
    uint32_t startBlk = 0;
    uint32_t endBlk = 0;
    uint32_t blockDim = procNum < nzRealCoreNum ? procNum : nzRealCoreNum;
    for (uint32_t i = 0; i < blockDim; i++) {
        startBlk = endBlk;
        endBlk += procPerBlk;
        if (tailBlks > i) {
            endBlk += 1;
        }
        tilingParam[i * NZ_INDEX_4 + NZ_INDEX_12] = static_cast<uint32_t>(startBatch);
        while (tilingParam[AtbOps::GetNzRealCoreTilingOffset() + startBatch * NZ_REAL_CORE_TILING_SIZE + NZ_INDEX_12] <=
                   endBlk &&
               startBatch < mmInfo.batchSize - 1) {
            startBatch++;
        }
        tilingParam[i * NZ_INDEX_4 + NZ_INDEX_13] = static_cast<uint32_t>(startBatch);
        tilingParam[i * NZ_INDEX_4 + NZ_INDEX_14] = static_cast<uint32_t>(startBlk);
        tilingParam[i * NZ_INDEX_4 + NZ_INDEX_15] = static_cast<uint32_t>(endBlk);
    }
    GetTilingKey(mmInfo, tilingParam);
    return;
}

Status GetUnpadFlashAttentionTilingParam(const UnpadFlashAttentionNzInfo mmInfo, uint32_t &blockDim,
                                         uint32_t *tilingParam, uint32_t tilingParamSize)
{
    if (tilingParam == nullptr) {
        MKI_LOG(ERROR) << "pointer tilingParam or seq is nullptr.";
        return Status::FailStatus(ERROR_INVALID_VALUE, "pointer tilingParam or seq is nullptr");
    }
    uint32_t nzRealCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (nzRealCoreNum <= CORE_NUM_LIMIT) {
        nzRealCoreNum = nzRealCoreNum <= USE_MAX_CORE_NUM ? nzRealCoreNum : USE_MAX_CORE_NUM;
    }
    uint32_t initSize = static_cast<uint32_t>(mmInfo.batchSize * NZ_REAL_CORE_TILING_SIZE * sizeof(uint32_t) +
                                              GetNzRealCoreTilingOffset());
    auto ret = memset_s(tilingParam, tilingParamSize, 0, initSize);
    MKI_CHECK(ret == EOK, "Failed to clear the array", return Status::FailStatus(-1));
    AddrOffsetsNz addrOffsets;
    float tor = mmInfo.tor;
    uint32_t *torUptr = static_cast<uint32_t *>(static_cast<void *>(&tor));
    Status ret1 = FillTilingParamRealCore(mmInfo, *torUptr, addrOffsets, tilingParam, nzRealCoreNum);
    OP_TILING_CHECK_STATUS_RETURN(ret1);
    int64_t procNum = static_cast<int64_t>(mmInfo.innerBatchSize) * addrOffsets.totalQBlkNum;
    MKI_CHECK(procNum > 0 && procNum <= UINT32_MAX, "blockDim overflow",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    blockDim = static_cast<uint32_t>(procNum) < nzRealCoreNum ? static_cast<uint32_t>(procNum) : nzRealCoreNum;
    MKI_LOG(INFO) << "blockDim: " << blockDim;
    return Status::OkStatus();
}

} // namespace AtbOps