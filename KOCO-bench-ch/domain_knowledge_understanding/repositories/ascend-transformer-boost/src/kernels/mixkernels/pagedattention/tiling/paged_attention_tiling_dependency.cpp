/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <securec.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "atbops/params/params.h"
#include "paged_attention_tiling.h"

namespace AtbOps {
const int32_t TILING_BATCH = 0;
const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_FORMER_BATCH = 8;
const int32_t TILING_FORMER_HEAD = 9;
const int32_t TILING_TAIL_BATCH = 10;
const int32_t TILING_TAIL_HEAD = 11;
const int32_t TILING_HEADNUM_MOVE = 12;
const int32_t TILING_MASK_MAX_LEN = 13;
const int32_t TILING_BATCH_STRIDE = 14;
const int32_t TILING_HEAD_STRIDE = 15;
const int32_t TILING_KEY = 16;
const int32_t TILING_HEADSIZE = 17;
const int32_t TILING_PARASIZE = 18;
const int32_t TILING_GROUPNUM = 19;
const int32_t TILING_FORMER_GROUP_MOVE = 20;
const int32_t TILING_TAIL_GROUP_MOVE = 21;
const int32_t TILING_MAX_KVSEQLEN = 22;
const int32_t TILING_KVSPLIT = 23;
const int32_t TILING_KVCORENUM = 24;
const int32_t TILING_BLOCKSIZE_CALC = 25;
const int32_t TILING_TOTAL_BLOCK_NUM = 26;
const int32_t TILING_PREFILL_BS = 27;
const int32_t TILING_DECODER_BS = 28;
const int32_t TILING_HEADDIM_V = 29;
const int32_t TILING_MODCOEF = 30;
const int32_t TILING_DIVCOEF = 31;
const int32_t TILING_QHEADORIGINAL = 32;
const int32_t TILING_COMPRESSHEAD = 33;
const int32_t TILING_QUANTYPE = 34;
const int32_t TILING_DATA_SHAPE_TYPE = 35;
const int32_t TILING_SCALETYPE = 36;
const int32_t TILING_MASK_TYPE_ND = 37;
const int32_t TILING_HEADDIM_K_SPLIT = 38;
const int32_t TILING_HEADDIM_V_SPLIT = 39;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;
const int32_t TILING_MTP_HEAD_SPLIT_SIZE = 42;
const int32_t TILING_MTP_HEAD_SPLIT_NUM = 43;

const int32_t TILING_SCALE_TYPE = 121;
const int32_t TILING_HEAD_NUM_LIMIT = 122;
const int32_t TILING_MASK_TYPE = 123;
const int32_t TILING_MASK_STRIDE = 124;
const int32_t TILING_DECODE_TYPE = 125;

const int32_t TILING_SCALE_TYPE_910A = 185;
const int32_t TILING_HEAD_NUM_LIMIT_910A = 186;
const int32_t TILING_MASK_TYPE_910A = 187;
const int32_t TILING_MASK_STRIDE_910A = 188;
const int32_t TILING_DECODE_TYPE_910A = 189;
const int32_t TILING_UNUSED_INDEX = -1;

const int32_t NUM0 = 0;
const int32_t NUM1 = 1;
const int32_t NUM2 = 2;
const int32_t NUM3 = 3;
const int32_t NUM4 = 4;
const int32_t NUM5 = 5;
const int32_t NUM6 = 6;
const int32_t NUM7 = 7;
const int32_t NUM8 = 8;
const int32_t NUM9 = 9;
const int32_t NUM10 = 10;
const int32_t NUM11 = 11;
const int32_t NUM12 = 12;
const int32_t NUM13 = 13;
const int32_t NUM14 = 14;
const int32_t NUM15 = 15;
const int32_t NUM16 = 16;
const int32_t NUM17 = 17;
const int32_t NUM18 = 18;
const int32_t NUM19 = 19;
const int32_t NUM20 = 20;
const int32_t NUM21 = 21;
const int32_t NUM32 = 32;
const int32_t NUM64 = 64;
const int32_t NUM128 = 128;
const int32_t NUM256 = 256;
const int32_t NUM512 = 512;
const int32_t NUM576 = 576;
const int32_t INDEX125 = 125;
const int32_t INDEX126 = 126;
const int32_t INDEX127 = 127;
const int32_t INDEX190 = 190;
const int32_t INDEX191 = 191;
const int32_t SPECIALNUM_TOKENS = 16;
const int32_t SPECIALNUM_HEADS = 32;
const int32_t EMBEDDING_LIMIT = 128;
const int32_t HIGH_32BIT = 32;
const float SPLITKV_RATION = 0.8;
const float SPLITHEAD_RATION = 0.9;
const int32_t KV_SEQLEN_SLICE = 128;
const int32_t KV_SEQLEN_SLICE_256 = 256;
const int32_t KV_SEQLEN_SLICE_512 = 512;
const uint32_t HEADNUM_LIMIT = 128;
const uint32_t HEADNUM_LIMIT_REGU = 32;
const uint32_t HEADNUM_NZ_LIMIT = 128;
const uint32_t DEQUANT_HIDDEN_LIMIT = 2048; // 8 * 128 * 2

const int32_t SPECIALNUMBATCH = 16;
const int32_t SPECIALNUMHEADS = 8;
const int32_t PP_INDEX = 16;
const int32_t PP_MM_NUM = 8;
const int32_t PP_NN_NUM = 16;
const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;
constexpr std::array<int32_t, PP_MM_NUM> PP_MM = { 16, 32, 48, 64, 80, 96, 112, 128 };
constexpr std::array<int32_t, NUM6> QN_TILE_LIST = { 128, 64, 32, 16, 8, 1 };

using IndexArr = std::array<int32_t, NUM4>;

void SplitCoreND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim,
    uint32_t *tilingParam, uint32_t decoderBatchSize, bool isMLA);
Status GetNdPagedAttentionTiling(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam);
void CalcuHeadNd(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam);
uint32_t SplitCoreBNSND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam, bool isLongSeq,
                        const OpParam::PagedAttention &param);
uint32_t SplitCoreBNND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                       const OpParam::PagedAttention &param, const bool isMLA);
void GetBlockSizeCalc(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, const bool isMLA,
                      const bool& mlaOptimization);
void CalcuEmbedSplitNd(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, const bool& mlaOptimization);

inline uint32_t GetHigh32Bit(uint64_t v) { return static_cast<uint32_t>(v >> HIGH_32BIT); }
inline uint32_t GetLoww32Bit(uint64_t v) { return static_cast<uint32_t>(v); }

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

static const std::unordered_map<PlatformType, IndexArr> TILING_HEAD_OFFSET_MAP{
    { PlatformType::ASCEND_910A, { INDEX190, INDEX191, TILING_MASK_TYPE_910A, TILING_SCALE_TYPE_910A } },
    { PlatformType::ASCEND_310P, { INDEX126, INDEX127, TILING_MASK_TYPE, TILING_SCALE_TYPE } }
};

static const std::unordered_map<PlatformType, IndexArr> g_nzTilingOffsetMap{
    { PlatformType::ASCEND_910A,
      { TILING_HEAD_NUM_LIMIT_910A, TILING_MASK_STRIDE_910A, TILING_DECODE_TYPE_910A, TILING_UNUSED_INDEX } },
    { PlatformType::ASCEND_310P,
      { TILING_HEAD_NUM_LIMIT, TILING_MASK_STRIDE, TILING_DECODE_TYPE, TILING_UNUSED_INDEX } }
};

inline IndexArr GetTilingHeadOffset()
{
    return TILING_HEAD_OFFSET_MAP.at(PlatformInfo::Instance().GetPlatformType());
}
inline IndexArr GetNzTilingOffset()
{
    return g_nzTilingOffsetMap.at(PlatformInfo::Instance().GetPlatformType());
}

void SplitCoreND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim,
                 uint32_t *tilingParam, uint32_t decoderBatchSize, bool isMLA, const OpParam::PagedAttention &param)
{
    bool isLongSeq = (tilingParam[TILING_MAX_KVSEQLEN] >= blockDim * KV_SEQLEN_SLICE * NUM2) &&
        (decoderBatchSize <= blockDim * SPLITKV_RATION);
    uint32_t decoderTaskNum = 0;
    if (isMLA || (decoderBatchSize * mmInfo.numHeads >= blockDim * SPLITKV_RATION && !isLongSeq)) {
        MKI_LOG(INFO) << "split Core BN begin";
        decoderTaskNum = SplitCoreBNND(mmInfo, blockDim, tilingParam, param, isMLA);
    } else {
        decoderTaskNum = SplitCoreBNSND(mmInfo, blockDim, tilingParam, isLongSeq, param);
    }
    int32_t kvRealHeads = (mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads;
    if (mmInfo.embeddingSize % mmInfo.tBlockAlign == 0 && mmInfo.embeddingSize <= EMBEDDING_LIMIT &&
        mmInfo.embeddingSizeV % mmInfo.tBlockAlign == 0 && mmInfo.embeddingSizeV <= EMBEDDING_LIMIT &&
        kvRealHeads == mmInfo.numHeads && mmInfo.type != TilingKeyType::TILING_INT8_CUBE_QUANT) {
        tilingParam[TILING_HEADNUM_MOVE] = 2; // 2 improve bandwidth utilization
        if ((mmInfo.type == TilingKeyType::TILING_QUANT_FP16OUT ||
            mmInfo.type == TilingKeyType::TILING_QUANT_BF16OUT)) {
            tilingParam[TILING_HEADNUM_MOVE] = 4; // 4 improve bandwidth utilization
        }
    } else {
        tilingParam[TILING_HEADNUM_MOVE] = 1;
    }
    MKI_LOG(INFO) <<"tiling headnum move is : "<< tilingParam[TILING_HEADNUM_MOVE];
    MKI_LOG(INFO) << "total task num:" << mmInfo.numHeads * tilingParam[TILING_TOTAL_BLOCK_NUM] + decoderTaskNum;
}

uint32_t SetTilingKeyMlaMtp(const PagedAttentionInfo &mmInfo, uint32_t maxQseqlen)
{
    uint32_t embeddingKey = (mmInfo.embeddingSize <= MLA_THRESHOLD) ? 1 : 0;
    uint32_t typeKey = (mmInfo.type == TilingKeyType::TILING_BF16_DATA) ? 1 : 0;
    uint32_t highPKey = typeKey;
    // workModeKey: case 1(prefill mode), only tile qS. case 0(mtp mode), only tile qN
    uint32_t workModeKey = (maxQseqlen > NUM16) ? 1 : 0;
    uint32_t tilingKey = (workModeKey << DIM_4) + (highPKey << DIM_3) + (typeKey << DIM_2) + embeddingKey;
    return tilingKey;
}

void SetTilingKey(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, bool isMLA,
    const OpParam::PagedAttention &param, const bool &mlaOptimization, uint32_t maxQseqlen)
{
    constexpr uint32_t dataShapeTypeMask = 0x03;
    uint32_t isParallePa = isMLA ? 0 : (tilingParam[TILING_PREFILL_BS] >= 1);
    uint32_t isSplitKey = isMLA ? 0 : (tilingParam[TILING_KVCORENUM] > 1);
    uint32_t isSplitBlock = 0;
    if (mmInfo.blockSize >= KV_SEQLEN_SLICE && (mmInfo.embeddingSize == KV_SEQLEN_SLICE_256 &&
        mmInfo.embeddingSizeV == KV_SEQLEN_SLICE_256)) {
            isSplitBlock = 1;
        }
    uint32_t tilingKey = 0;
    if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND) {
    // 目前9bit数据 0 00 00 00 00
    // optfeature (compressHead isMLA):2bit (dataFormat isParallePa splitkv):4bit quantmode:2bit (dequantCube, dequantVec, dequantMsd, quant)
    // qdatatype:2bit(FP16/BF16/INT8)
        tilingKey = (static_cast<uint32_t>(isMLA) << NUM8) + (isParallePa << NUM5) + (isSplitKey << NUM4) +
                    static_cast<uint32_t>(mmInfo.type);
        if (mlaOptimization) {
            tilingKey += 1 << NUM10;
        }
    } else if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
    // 目前5bit数据 0 0 00 0
    // workmode : 1bit, vector part precision : 1bit, date type : 2bit, embed spec : 1bit
        tilingKey = SetTilingKeyMlaMtp(mmInfo, maxQseqlen);
    } else {
        tilingKey = (mmInfo.compressHead << NUM9) + (isSplitBlock << NUM7) +
                    ((static_cast<uint32_t>(mmInfo.dataShapeType) & dataShapeTypeMask) << NUM6) +
                    (isParallePa << NUM5) + (isSplitKey << NUM4) + static_cast<uint32_t>(mmInfo.type);
    }
    tilingParam[TILING_KEY] = tilingKey;
    MKI_LOG(INFO) << "tilingKey: " << tilingKey;
}

void CalcuEmbedSplitNd(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, const bool& mlaOptimization)
{
    bool isQuant = (mmInfo.type == TilingKeyType::TILING_QUANT_FP16OUT ||
        mmInfo.type == TilingKeyType::TILING_QUANT_BF16OUT);
    uint32_t embedQKSplit = 0;
    uint32_t embedVOSplit = 0;
    uint32_t embedVOSplitVectorFormer = 0;
    uint32_t embedVOSplitVectorTail = 0;
    if (isQuant) {
        embedQKSplit = mmInfo.embeddingSize > MLA_THRESHOLD * NUM2 ? MLA_THRESHOLD * NUM2 : mmInfo.embeddingSize;
        if (tilingParam[TILING_FORMER_GROUP_MOVE] <= NUM64) {
            embedVOSplit = mmInfo.embeddingSizeV > MLA_THRESHOLD * NUM2 ? MLA_THRESHOLD * NUM2 : mmInfo.embeddingSizeV;
        } else {
            embedVOSplit = mmInfo.embeddingSizeV > MLA_THRESHOLD ? MLA_THRESHOLD : mmInfo.embeddingSizeV;
        }
    } else {
        embedQKSplit = mmInfo.embeddingSize > MLA_THRESHOLD ? MLA_THRESHOLD : mmInfo.embeddingSize;
        embedVOSplit = mmInfo.embeddingSizeV > MLA_THRESHOLD ? MLA_THRESHOLD : mmInfo.embeddingSizeV;
    }
    if (tilingParam[TILING_FORMER_GROUP_MOVE] <= NUM64) {
        embedVOSplitVectorFormer = mmInfo.embeddingSizeV > MLA_THRESHOLD ? MLA_THRESHOLD : mmInfo.embeddingSizeV;
    } else {
        embedVOSplitVectorFormer = mmInfo.embeddingSizeV > EMBEDDING_LIMIT ? EMBEDDING_LIMIT : mmInfo.embeddingSizeV;
    }
    if (tilingParam[TILING_TAIL_GROUP_MOVE] <= NUM64) {
        embedVOSplitVectorTail = mmInfo.embeddingSizeV > MLA_THRESHOLD ? MLA_THRESHOLD : mmInfo.embeddingSizeV;
    } else {
        embedVOSplitVectorTail = mmInfo.embeddingSizeV > EMBEDDING_LIMIT ? EMBEDDING_LIMIT : mmInfo.embeddingSizeV;
    }
    if (mlaOptimization) {
        embedQKSplit = NUM128;
    }
    tilingParam[TILING_HEADDIM_K_SPLIT] = embedQKSplit;
    tilingParam[TILING_HEADDIM_V_SPLIT] = embedVOSplit;
    tilingParam[TILING_HEADDIM_V_SPLIT_VECTOR_FORMER] = embedVOSplitVectorFormer;
    tilingParam[TILING_HEADDIM_V_SPLIT_VECTOR_TAIL] = embedVOSplitVectorTail;
    MKI_LOG(INFO) << "embedQKSplit: " << embedQKSplit
        << " embedVOSplit: " << embedVOSplit << " embedVOSplitVectorFormer: " << embedVOSplitVectorFormer
        << " embedVOSplitVectorTail: " << embedVOSplitVectorTail;
}

void InitEmbedQKVOSplit(bool isMLA, const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, uint32_t &embedQKSplit,
    uint32_t &embedVOSplit)
{
    if (isMLA) {
        embedQKSplit = tilingParam[TILING_HEADDIM_K_SPLIT];
        embedVOSplit = tilingParam[TILING_HEADDIM_V_SPLIT];
    } else {
        embedQKSplit = mmInfo.embeddingSize;
        embedVOSplit = mmInfo.embeddingSizeV;
    }
}

void InitEmbeddingSize(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, uint32_t embedQKSplit,
    uint32_t embedVOSplit)
{
    if (mmInfo.blockSize <= KV_SEQLEN_SLICE / NUM2 && mmInfo.blockSize * NUM2 * embedQKSplit <= BLOCK_LIMIT &&
        mmInfo.blockSize * NUM2 * embedVOSplit <= BLOCK_LIMIT) {
        tilingParam[TILING_BLOCKSIZE_CALC] = mmInfo.blockSize * NUM2;
    } else if (mmInfo.blockSize >= KV_SEQLEN_SLICE && (mmInfo.embeddingSize == KV_SEQLEN_SLICE_256 &&
        mmInfo.embeddingSizeV == KV_SEQLEN_SLICE_256)) {
        tilingParam[TILING_BLOCKSIZE_CALC] = KV_SEQLEN_SLICE;
    } else {
        tilingParam[TILING_BLOCKSIZE_CALC] = mmInfo.blockSize;
    }
}

void GetBlockSizeCalc(const PagedAttentionInfo &mmInfo, uint32_t *tilingParam, const bool isMLA,
                      const bool &mlaOptimization)
{
    uint32_t embedQKSplit = 0;
    uint32_t embedVOSplit = 0;
    InitEmbedQKVOSplit(isMLA, mmInfo, tilingParam, embedQKSplit, embedVOSplit);
    InitEmbeddingSize(mmInfo, tilingParam, embedQKSplit, embedVOSplit);
    bool isBlockSatisfy =
        ((mmInfo.blockSize == KV_SEQLEN_SLICE_256 / NUM4) || (mmInfo.blockSize == KV_SEQLEN_SLICE_256 / NUM2));
    uint32_t l0Limit = mmInfo.tBlockAlign * BLOCK_LIMIT_NO_PINGPONG_UINT8 / BLOCK_SIZE_32;
    if (mmInfo.type != TilingKeyType::TILING_INT8_CUBE_QUANT &&
        tilingParam[TILING_MAX_KVSEQLEN] >= KV_SEQLEN_SLICE_256 && tilingParam[TILING_KVSPLIT] >= KV_SEQLEN_SLICE_256 &&
        tilingParam[TILING_FORMER_HEAD] <= BLOCK_LIMIT / KV_SEQLEN_SLICE_256 &&
        tilingParam[TILING_TAIL_HEAD] <= BLOCK_LIMIT / KV_SEQLEN_SLICE_256 &&
        KV_SEQLEN_SLICE_256 * embedQKSplit <= l0Limit && KV_SEQLEN_SLICE_256 * embedVOSplit <= l0Limit &&
        isBlockSatisfy) {
        tilingParam[TILING_BLOCKSIZE_CALC] = KV_SEQLEN_SLICE_256;
    }
    bool isQuant =
        (mmInfo.type == TilingKeyType::TILING_QUANT_FP16OUT || mmInfo.type == TilingKeyType::TILING_QUANT_BF16OUT);
    bool isSeqlenLarge =
        tilingParam[TILING_MAX_KVSEQLEN] >= KV_SEQLEN_SLICE_512 && tilingParam[TILING_KVSPLIT] >= KV_SEQLEN_SLICE_512;
    bool isSpaceEnough = tilingParam[TILING_FORMER_HEAD] <= BLOCK_LIMIT / KV_SEQLEN_SLICE_512 &&
                         tilingParam[TILING_TAIL_HEAD] <= BLOCK_LIMIT / KV_SEQLEN_SLICE_512 &&
                         KV_SEQLEN_SLICE_512 * embedQKSplit <= BLOCK_LIMIT_NO_PINGPONG_UINT8 &&
                         KV_SEQLEN_SLICE_512 * embedVOSplit <= BLOCK_LIMIT_NO_PINGPONG_UINT8 &&
                         tilingParam[TILING_HEADNUM_MOVE] < DIM_4 &&
                         tilingParam[TILING_FORMER_GROUP_MOVE] < BLOCK_SIZE_32;
    if (isQuant && isSeqlenLarge && isSpaceEnough && isBlockSatisfy) {
        tilingParam[TILING_BLOCKSIZE_CALC] = KV_SEQLEN_SLICE_512;
    }
    if (mlaOptimization) {
        tilingParam[TILING_BLOCKSIZE_CALC] = NUM256;
    }
    MKI_LOG(INFO) << "tiling block size calc: " << tilingParam[TILING_BLOCKSIZE_CALC];
}

void CalcuHeadNd(const PagedAttentionInfo &mmInfo, uint32_t &formerHeadSplit, uint32_t &tailHeadSplit,
    uint32_t *tilingParam)
{
    uint32_t groupNum = mmInfo.numHeads / ((mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads);
    uint32_t formerGroupNumMove = 1;
    uint32_t tailGroupNumMove = 1;

    if (formerHeadSplit % groupNum == 0) {
        formerGroupNumMove = groupNum;
    } else if ((formerHeadSplit < groupNum) &&
        ((mmInfo.kvHeads == 1) || (groupNum % formerHeadSplit == 0))) {
        formerGroupNumMove = formerHeadSplit;
    }

    if ((tailHeadSplit != 0) && (tailHeadSplit % groupNum == 0)) {
        tailGroupNumMove = groupNum;
    } else if ((tailHeadSplit != 0) && (tailHeadSplit < groupNum) &&
        ((mmInfo.kvHeads == 1) || (groupNum % tailHeadSplit == 0))) {
        tailGroupNumMove = tailHeadSplit;
    }

    tilingParam[TILING_GROUPNUM] = static_cast<uint32_t>(groupNum);
    tilingParam[TILING_FORMER_GROUP_MOVE] = static_cast<uint32_t>(formerGroupNumMove);
    tilingParam[TILING_TAIL_GROUP_MOVE] = static_cast<uint32_t>(tailGroupNumMove);
    MKI_LOG(INFO) <<"groupNum: "<< groupNum <<
    " formerGroupNumMove: " << formerGroupNumMove << " tailGroupNumMove:" << tailGroupNumMove;
}

uint32_t GetHeadLimit(
    const PagedAttentionInfo &mmInfo, const OpParam::PagedAttention &param, const uint32_t &blockDim)
{
    uint32_t kvheadLimit = std::min(DEQUANT_HIDDEN_LIMIT / mmInfo.embeddingSize, HEADNUM_LIMIT);
    uint32_t taskNumMLA = mmInfo.numTokens * (mmInfo.numHeads + HEADNUM_LIMIT - 1) / HEADNUM_LIMIT;
    uint32_t headLimitMLA = (taskNumMLA >= blockDim) ? HEADNUM_LIMIT : HEADNUM_LIMIT_REGU;
    uint32_t regularLimit =
        (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND) ? HEADNUM_LIMIT_REGU : headLimitMLA;
    uint32_t headLimit =
        (mmInfo.type == TilingKeyType::TILING_INT8_VEC_QUANT || mmInfo.type == TilingKeyType::TILING_INT8_VEC_QUANTBF16)
            ? kvheadLimit
            : regularLimit;
    return headLimit;
}

uint32_t SplitCoreBNSND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam, bool isLongSeq,
                        const OpParam::PagedAttention &param)
{
    uint32_t decoderBatch = tilingParam[TILING_DECODER_BS];
    MKI_LOG(INFO) << "split Core BNS begin";
    uint32_t kvSeqklenMaxAlign =
        Utils::RoundUp(tilingParam[TILING_MAX_KVSEQLEN], static_cast<uint32_t>(mmInfo.blockSize));
    uint32_t kvSeqBlockNum = kvSeqklenMaxAlign / mmInfo.blockSize;
    uint32_t kvBlockPreCore = 0;
    if (isLongSeq) {
        // split kv first
        kvBlockPreCore = Utils::CeilDiv(kvSeqBlockNum, blockDim);
    } else {
        // split bath first
        uint32_t coreNumPerBatch = Utils::CeilDiv(blockDim, static_cast<uint32_t>(decoderBatch));
        kvBlockPreCore = Utils::CeilDiv(kvSeqBlockNum, coreNumPerBatch);
    }
    uint32_t kvSplitPerCore = kvBlockPreCore * mmInfo.blockSize;
    uint32_t kvSplitCoreNum = Utils::CeilDiv(kvSeqklenMaxAlign, kvSplitPerCore);
    uint32_t coreNumPerKV = 1;
    if (decoderBatch * kvSplitCoreNum < blockDim * SPLITHEAD_RATION) {
        coreNumPerKV = Utils::CeilDiv(blockDim, (decoderBatch * kvSplitCoreNum));
    }
    uint32_t headSplit = Utils::CeilDiv(static_cast<uint32_t>(mmInfo.numHeads), coreNumPerKV);
    uint32_t headLitmit = GetHeadLimit(mmInfo, param, blockDim);
    headSplit = headSplit > headLitmit ? headLitmit : headSplit;
    uint32_t headCoreNum = Utils::CeilDiv(static_cast<uint32_t>(mmInfo.numHeads), headSplit);
    uint32_t block = headCoreNum * decoderBatch * kvSplitCoreNum;
    uint32_t formerBatch = decoderBatch;
    uint32_t tailBatch = 0;
    uint32_t formerHeadSplit = headSplit;
    uint32_t tailHeadSplit = 0;
    tilingParam[TILING_FORMER_BATCH] = formerBatch;
    tilingParam[TILING_FORMER_HEAD] = formerHeadSplit;
    tilingParam[TILING_TAIL_BATCH] = tailBatch;
    tilingParam[TILING_TAIL_HEAD] = tailHeadSplit;
    tilingParam[TILING_KVSPLIT] = kvSplitPerCore;
    tilingParam[TILING_KVCORENUM] = kvSplitCoreNum;
    blockDim = (block < blockDim) ? block : blockDim;
    MKI_LOG(INFO) << "split Core BNS end formerbatch: " << formerBatch << " formerHeadSplit: " << formerHeadSplit <<
        " tailBatch:" << tailBatch << " kvSplitPerCore: " << kvSplitPerCore << " kvSplitCoreNum:" << kvSplitCoreNum <<
        " blockDim:" << blockDim;
    CalcuHeadNd(mmInfo, formerHeadSplit, tailHeadSplit, tilingParam);
    return block;
}

uint32_t SplitCoreBNND(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                       const OpParam::PagedAttention &param, const bool isMLA)
{
    uint32_t decoderBatch = tilingParam[TILING_DECODER_BS];
    uint32_t coreNumPerBatch = (blockDim + decoderBatch - 1) / decoderBatch;
    bool isQuant = (mmInfo.type == TilingKeyType::TILING_QUANT_FP16OUT ||
        mmInfo.type == TilingKeyType::TILING_QUANT_BF16OUT);
    if (blockDim * SPLITKV_RATION <= decoderBatch && decoderBatch <= blockDim && isQuant && mmInfo.kvHeads == 1) {
        coreNumPerBatch = 1;
    }
    uint32_t headSplit = (mmInfo.numHeads + coreNumPerBatch - 1) / coreNumPerBatch;
    uint32_t headLitmit = GetHeadLimit(mmInfo, param, blockDim);
    headSplit = headSplit > headLitmit ? headLitmit : headSplit;

    if (decoderBatch == SPECIALNUM_TOKENS && mmInfo.numHeads == SPECIALNUM_HEADS) {
        headSplit = 8; // 实测该场景切8性能更优
    }
    uint32_t loopLen = (mmInfo.numHeads + headSplit - 1) / headSplit; // 2

    uint64_t block64 = static_cast<uint64_t>(loopLen) * static_cast<uint64_t>(decoderBatch);
    if (block64 >= std::numeric_limits<uint32_t>::max()) {
        block64 = std::numeric_limits<uint32_t>::max();
        MKI_LOG(ERROR) << "batch: " << decoderBatch << " is too large, only calculate part of it.";
    }

    uint32_t block = static_cast<uint32_t>(block64);
    uint32_t formerBatch = decoderBatch;
    uint32_t tailBatch = 0;
    uint32_t formerHeadSplit = headSplit;
    uint32_t tailHeadSplit = 0;

    if (block > blockDim) {
        uint32_t processLoop = block / blockDim;
        formerBatch = processLoop * blockDim / loopLen;
        tailBatch = decoderBatch - formerBatch;
        uint32_t processRemain = tailBatch * loopLen;
        // 当最后一轮任务数小于16时调整
        bool adjLastHead = 0;
        if ((mmInfo.kvHeads == 0) || (mmInfo.numHeads == mmInfo.kvHeads)) {
            adjLastHead = (processRemain < SPECIALNUM_TOKENS) && (tailBatch > 0);
        } else {
            adjLastHead = (processRemain < SPECIALNUM_TOKENS) && (tailBatch > 0) && (tailBatch <= blockDim / NUM2);
        }
        if (adjLastHead) {
            if (isMLA && isQuant) {
                coreNumPerBatch = blockDim /  tailBatch;
            } else {
                coreNumPerBatch = (blockDim + tailBatch - 1) / tailBatch;
            }
            tailHeadSplit = (mmInfo.numHeads + coreNumPerBatch - 1) / coreNumPerBatch;
            tailHeadSplit = tailHeadSplit > headLitmit ? headLitmit : tailHeadSplit;
        } else {
            formerBatch = decoderBatch;
            tailBatch = 0;
        }
    }
    blockDim = (block < blockDim) ? block : blockDim;
    tilingParam[TILING_FORMER_BATCH] = static_cast<uint32_t>(formerBatch);
    tilingParam[TILING_FORMER_HEAD] = static_cast<uint32_t>(formerHeadSplit);
    tilingParam[TILING_TAIL_BATCH] = static_cast<uint32_t>(tailBatch);
    tilingParam[TILING_TAIL_HEAD] = static_cast<uint32_t>(tailHeadSplit);
    tilingParam[TILING_KVSPLIT] =
        Utils::RoundUp(tilingParam[TILING_MAX_KVSEQLEN], static_cast<uint32_t>(mmInfo.blockSize));
    tilingParam[TILING_KVCORENUM] = 1;
    MKI_LOG(INFO) << "split Core BN end formerbatch: " << formerBatch << " formerHeadSplit: " << formerHeadSplit <<
        " tailBatch:" << tailBatch << " kvSplitPerCore: " << tilingParam[TILING_KVSPLIT] << " kvSplitCoreNum:" << 1 <<
        " blockDim:" << blockDim;
    CalcuHeadNd(mmInfo, formerHeadSplit, tailHeadSplit, tilingParam);
    return block;
}

void GetLOffsetInfo(uint32_t *tilingParam, const PagedAttentionInfo &mmInfo, uint32_t prefillBatchSize,
                    AddrOffsets addrOffsets)
{
    uint32_t curdecoderbatch = 0;
    uint32_t paOnly = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqlen = (paOnly == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqlen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqlen;
        int32_t tilingOffset = static_cast<int32_t>(prefillBatchSize + curdecoderbatch) *
            TILING_PARA_SIZE + TILING_HEAD_SIZE;
        if (qSeqlen <= 1) {
            uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqlen);
            tilingParam[tilingOffset + NUM11] = GetHigh32Bit(addrOffsets.addrLSeqOffset);
            tilingParam[tilingOffset + NUM12] = GetLoww32Bit(addrOffsets.addrLSeqOffset);
            addrOffsets.addrLSeqOffset += static_cast<uint64_t>(static_cast<int64_t>(tilingParam[TILING_KVCORENUM]) *
                mmInfo.numHeads * qSeqlen);
            tilingParam[tilingOffset + NUM15] = GetHigh32Bit(addrOffsets.addrOFdSeqOffset);
            tilingParam[tilingOffset + NUM16] = GetLoww32Bit(addrOffsets.addrOFdSeqOffset);
            addrOffsets.addrOFdSeqOffset += addressOffset;
            curdecoderbatch += 1;
        }
    }
}

void GetLookaheadTilingHead(uint32_t *tilingParam, const PagedAttentionInfo &mmInfo)
{
    tilingParam[TILING_MASK_MAX_LEN] = static_cast<uint32_t>(mmInfo.maxPromptLen);
    tilingParam[TILING_BATCH_STRIDE] = static_cast<uint32_t>(mmInfo.batchStride);
    tilingParam[TILING_HEAD_STRIDE] = static_cast<uint32_t>(mmInfo.headStride);
    tilingParam[TILING_MODCOEF] = static_cast<uint32_t>(mmInfo.modCoef);
    tilingParam[TILING_DIVCOEF] = static_cast<uint32_t>(mmInfo.divCoef);
}

uint64_t GetLookaheadMaskOffset(const PagedAttentionInfo &mmInfo, const OpParam::PagedAttention &param,
    uint32_t kvSeqLen, uint32_t qSeqLen, uint32_t preqSeqLen)
{
    uint32_t maskOffset = 0;
    auto maskType = param.maskType;
    if (mmInfo.isMaskSquare == 1) {
        maskOffset = (kvSeqLen - qSeqLen) * mmInfo.maxPromptLen;
    } else if (maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD) {
        maskOffset = (preqSeqLen) * mmInfo.maxPromptLen;
    } else {
        maskOffset = 0;
    }
    return static_cast<uint64_t>(maskOffset);
}

void GetAddrOffset(uint32_t *tilingParam, const AddrOffsets addrOffsets, const int32_t tilingOffset)
{
    tilingParam[tilingOffset + NUM4] = GetHigh32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM5] = GetLoww32Bit(addrOffsets.addrQSeqOffset);
    tilingParam[tilingOffset + NUM6] = GetHigh32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + NUM7] = GetLoww32Bit(addrOffsets.addrOSeqOffset);
    tilingParam[tilingOffset + NUM11] = 0;
    tilingParam[tilingOffset + NUM12] = 0;
}

void GetLookaheadBatchTiling(uint32_t *tilingParam, const OpParam::PagedAttention &param,
    const PagedAttentionInfo &mmInfo, std::vector<uint32_t> indices, AddrOffsets addrOffsets)
{
    uint32_t totalQBlkNum = 0;
    uint32_t curPrefillBatchId = 0;
    uint32_t curDecoderBatchId = 0;
    int32_t maxKVseqLen = 0;
    uint32_t preqSeqlen = 0;
    uint32_t paOnly = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = (paOnly == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t qSeqlenAligned = (qSeqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        maxKVseqLen = std::max(maxKVseqLen, kvSeqlen);
        int32_t mUbd =
            std::min((PP_BLOCK_BUFFER_SIZE / std::max(mmInfo.embeddingSize, mmInfo.blockSize) /
            BLOCK_SIZE) * BLOCK_SIZE, qSeqlenAligned);
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
        int32_t tilingOffset = TILING_HEAD_SIZE;
        if (qSeqLen > 1) {
            int32_t curQBlockNum = ((qSeqLen + PP_MM[mIbd] - 1) / PP_MM[mIbd]); // 对齐处理
            totalQBlkNum += static_cast<uint32_t>(curQBlockNum);
            tilingOffset += static_cast<int32_t>(curPrefillBatchId * TILING_PARA_SIZE);
            curPrefillBatchId += 1;
        } else {
            tilingOffset += static_cast<int32_t>((tilingParam[TILING_PREFILL_BS] + curDecoderBatchId) *
                TILING_PARA_SIZE);
            tilingParam[tilingOffset + NUM13] = tilingParam[TILING_PREFILL_BS] + indices.at(curDecoderBatchId);
            curDecoderBatchId++;
        }
        tilingParam[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingParam[tilingOffset + 1] = static_cast<uint32_t>(kvSeqlen);
        tilingParam[tilingOffset + NUM2] = static_cast<uint32_t>(PP_MM[mIbd]);
        tilingParam[tilingOffset + NUM3] = static_cast<uint32_t>(mmInfo.blockSize);
        tilingParam[tilingOffset + NUM8] = static_cast<uint32_t>(seqIdx);
        tilingParam[tilingOffset + NUM9] = totalQBlkNum;
        tilingParam[tilingOffset + NUM10] = GetHigh32Bit(GetLookaheadMaskOffset(mmInfo, param, kvSeqlen,
        qSeqLen, preqSeqlen));
        tilingParam[tilingOffset + NUM14] = GetLoww32Bit(GetLookaheadMaskOffset(mmInfo, param, kvSeqlen,
        qSeqLen, preqSeqlen));
        GetAddrOffset(tilingParam, addrOffsets, tilingOffset);
        uint64_t addressQffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSizeV * qSeqLen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        preqSeqlen += static_cast<uint32_t>(qSeqLen);
    }
    tilingParam[TILING_MAX_KVSEQLEN] = static_cast<uint32_t>(maxKVseqLen);
    tilingParam[TILING_TOTAL_BLOCK_NUM] = totalQBlkNum;
}

int32_t GetQNBlockTile(const PagedAttentionInfo &mmInfo, int32_t qSeqLen)
{
    int32_t tileListIdx = static_cast<int32_t>(std::ceil(std::log2(qSeqLen)));
    tileListIdx = (tileListIdx > NUM5) ? NUM5 : tileListIdx;
    int32_t qNBlockTile = QN_TILE_LIST[tileListIdx];
    int32_t group = mmInfo.numHeads / mmInfo.kvHeads;
    qNBlockTile = (qNBlockTile > group) ? group : qNBlockTile;
    return qNBlockTile;
}

void GetMlaMtpBatchTiling(uint32_t *tilingParam, const OpParam::PagedAttention &param,
    const PagedAttentionInfo &mmInfo, AddrOffsets addrOffsets, int32_t maxQseqlen)
{
    uint32_t totalQBlkNum = 0; // tile S or N
    int32_t maxKVseqLen = 0;
    uint32_t preqSeqlen = 0;
    int32_t curQNBlockTile = GetQNBlockTile(mmInfo, maxQseqlen);
    int32_t curQNBlockNum = (mmInfo.numHeads + curQNBlockTile - 1) / curQNBlockTile;
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        int32_t qSeqlenAligned = (qSeqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvSeqlen = *(mmInfo.kvSeqLen + seqIdx);
        maxKVseqLen = std::max(maxKVseqLen, kvSeqlen);
        int32_t mUbd = qSeqlenAligned;
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
        int32_t tilingOffset = TILING_HEAD_SIZE;
        int32_t curQSBlockTile = PP_MM[mIbd];
        int32_t curQSBlockNum = (qSeqLen + curQSBlockTile - 1) / curQSBlockTile; // 对齐处理

        int32_t curQBlockNum = curQSBlockNum * curQNBlockNum;
        totalQBlkNum += static_cast<uint32_t>(curQBlockNum);
        tilingOffset += static_cast<int32_t>(seqIdx * TILING_PARA_SIZE);
        tilingParam[tilingOffset] = static_cast<uint32_t>(qSeqLen);
        tilingParam[tilingOffset + 1] = static_cast<uint32_t>(kvSeqlen);
        tilingParam[tilingOffset + NUM2] = static_cast<uint32_t>(PP_MM[mIbd]);
        tilingParam[tilingOffset + NUM3] = static_cast<uint32_t>(mmInfo.blockSize);
        tilingParam[tilingOffset + NUM8] = static_cast<uint32_t>(seqIdx);
        tilingParam[tilingOffset + NUM9] = totalQBlkNum;
        tilingParam[tilingOffset + NUM10] = GetHigh32Bit(GetLookaheadMaskOffset(mmInfo, param, kvSeqlen,
        qSeqLen, preqSeqlen));
        tilingParam[tilingOffset + NUM14] = GetLoww32Bit(GetLookaheadMaskOffset(mmInfo, param, kvSeqlen,
        qSeqLen, preqSeqlen));
        GetAddrOffset(tilingParam, addrOffsets, tilingOffset);
        uint64_t addressQffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSize * qSeqLen);
        uint64_t addressOffset = static_cast<uint64_t>(mmInfo.numHeads * mmInfo.embeddingSizeV * qSeqLen);
        addrOffsets.addrQSeqOffset += addressQffset;
        addrOffsets.addrOSeqOffset += addressOffset;
        preqSeqlen += static_cast<uint32_t>(qSeqLen);
    }
    tilingParam[TILING_MTP_HEAD_SPLIT_SIZE] = static_cast<uint32_t>(curQNBlockTile);
    tilingParam[TILING_MTP_HEAD_SPLIT_NUM] = static_cast<uint32_t>(curQNBlockNum);
    tilingParam[TILING_MAX_KVSEQLEN] = static_cast<uint32_t>(maxKVseqLen);
    tilingParam[TILING_TOTAL_BLOCK_NUM] = totalQBlkNum;
}

bool MLAJudge(const PagedAttentionInfo &mmInfo)
{
    bool isMLA = (mmInfo.embeddingSize > MLA_THRESHOLD || mmInfo.embeddingSizeV > MLA_THRESHOLD ||
                  mmInfo.embeddingSize != mmInfo.embeddingSizeV);
    return isMLA;
}

std::vector<int32_t> CalcBatchRelatedFactors(const PagedAttentionInfo &mmInfo, uint32_t paOnly,
                                             std::vector<uint32_t> &decoderBatches)
{
    int32_t maxQseqlen = 0;
    int32_t prefillBatchSize = 0;
    int32_t decoderBatchSize = 0;
    std::vector<int32_t> factors{};
    for (int32_t seqIdx = 0; seqIdx < mmInfo.batch; seqIdx++) {
        int32_t qSeqLen = (paOnly == 1) ? 1 : *(mmInfo.qSeqLen + seqIdx);
        qSeqLen = (*(mmInfo.kvSeqLen + seqIdx) == 0) ? 0 : qSeqLen;
        maxQseqlen = std::max(qSeqLen, maxQseqlen);
        prefillBatchSize = qSeqLen > 1 ? prefillBatchSize + 1 : prefillBatchSize;
        if (qSeqLen <= 1) {
            decoderBatchSize++;
            decoderBatches.push_back(seqIdx);
        }
    }
    factors.push_back(maxQseqlen);
    factors.push_back(prefillBatchSize);
    factors.push_back(decoderBatchSize);
    return factors;
}

Status GetNdPagedAttentionTiling(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
                                 const OpParam::PagedAttention &param)
{
    uint32_t paOnly = (mmInfo.qSeqLen == nullptr) ? 1 : 0;
    AddrOffsets addrOffsets;
    std::vector<uint32_t> decoderBatches{};
    std::vector<int32_t> factors = CalcBatchRelatedFactors(mmInfo, paOnly, decoderBatches);
    int32_t maxQseqlen = factors[0];
    uint32_t prefillBatchSize = static_cast<uint32_t>(factors[1]);
    uint32_t decoderBatchSize = static_cast<uint32_t>(factors[2]);
    MKI_CHECK(!(prefillBatchSize > 0 && param.scaleType == OpParam::PagedAttention::SCALE_LOGN_FP32),
              "lookahead does not support logN", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingParam[TILING_PREFILL_BS] = prefillBatchSize;
    tilingParam[TILING_DECODER_BS] = decoderBatchSize;
    std::vector<uint32_t> indices(decoderBatchSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
        [&mmInfo, &decoderBatches](uint32_t firstBatchIdx, uint32_t secondBatchIdx) {
            return *(mmInfo.kvSeqLen + decoderBatches.at(firstBatchIdx)) <
            *(mmInfo.kvSeqLen + decoderBatches.at(secondBatchIdx));
            });
    bool isMLA = MLAJudge(mmInfo);
    if (param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        GetMlaMtpBatchTiling(tilingParam, param, mmInfo, addrOffsets, maxQseqlen);
    } else {
        GetLookaheadBatchTiling(tilingParam, param, mmInfo, indices, addrOffsets);
    }
    GetLookaheadTilingHead(tilingParam, mmInfo);

    bool mlaOptimization = false;
    if (mmInfo.blockSize == NUM256 && (mmInfo.numHeads == NUM32 || mmInfo.numHeads == NUM16) &&
        mmInfo.kvHeads == NUM1 && mmInfo.embeddingSize == NUM576 &&
        mmInfo.embeddingSizeV == NUM512) {
        mlaOptimization = true;
    }

    MKI_CHECK(!(isMLA && param.scaleType == OpParam::PagedAttention::SCALE_LOGN_FP32),
              "MLA does not support logN", return Status::FailStatus(ERROR_INVALID_VALUE));
    if (decoderBatchSize > 0 && param.type !=
        OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        SplitCoreND(mmInfo, blockDim, tilingParam, decoderBatchSize, isMLA, param);
        if (isMLA) {
            CalcuEmbedSplitNd(mmInfo, tilingParam, mlaOptimization);
        }
        GetBlockSizeCalc(mmInfo, tilingParam, isMLA, mlaOptimization);
        GetLOffsetInfo(tilingParam, mmInfo, prefillBatchSize, addrOffsets);
    }
    SetTilingKey(mmInfo, tilingParam, isMLA, param, mlaOptimization, maxQseqlen);
    return AtbOps::Status::OkStatus();
}

void GetTilingHead(const PagedAttentionInfo &mmInfo, const OpParam::PagedAttention &param, uint32_t *tilingParam,
    const uint32_t *torPtr, uint32_t nzFlag)
{
    if (nzFlag == 1) {
        auto [headStrideOffset, batchStrideOffset, maskTypeOffset, scaleTypeOffset] = GetTilingHeadOffset();
        tilingParam[TILING_BATCH] = mmInfo.numTokens > mmInfo.batch ?
                                static_cast<uint32_t>(mmInfo.numTokens) : static_cast<uint32_t>(mmInfo.batch);
        tilingParam[headStrideOffset] = static_cast<uint32_t>(mmInfo.headStride);
        tilingParam[batchStrideOffset] = static_cast<uint32_t>(mmInfo.batchStride);
        tilingParam[NUM8] = static_cast<uint32_t>(mmInfo.maxPromptLen);
        tilingParam[maskTypeOffset] = static_cast<uint32_t>(param.maskType);
        tilingParam[scaleTypeOffset] = static_cast<uint32_t>(param.scaleType);
    } else {
        tilingParam[TILING_BATCH] = static_cast<uint32_t>(mmInfo.batch);
        tilingParam[TILING_HEADSIZE] = static_cast<uint32_t>(TILING_HEAD_SIZE);
        tilingParam[TILING_PARASIZE] = static_cast<uint32_t>(TILING_PARA_SIZE);
        tilingParam[TILING_SCALETYPE] = param.scaleType;
        tilingParam[TILING_MASK_TYPE_ND] = static_cast<uint32_t>(param.maskType);
    }
    tilingParam[TILING_NUMHEADS] = static_cast<uint32_t>(mmInfo.numHeads);
    tilingParam[TILING_HEADDIM] = static_cast<uint32_t>(mmInfo.embeddingSize);
    tilingParam[TILING_HEADDIM_V] = static_cast<uint32_t>(mmInfo.embeddingSizeV);
    tilingParam[TILING_NUMBLOKS] = static_cast<uint32_t>(mmInfo.numBlocks);
    tilingParam[TILING_BLOCKSIZE] = static_cast<uint32_t>(mmInfo.blockSize);
    tilingParam[TILING_MAXBLOCKS] = static_cast<uint32_t>(mmInfo.maxNumBlocksPerQuery);
    tilingParam[TILING_TOR] = *torPtr;
    tilingParam[TILING_KVHEADS] = (mmInfo.kvHeads == 0) ? mmInfo.numHeads : mmInfo.kvHeads;
    tilingParam[TILING_QHEADORIGINAL] = static_cast<uint32_t>(mmInfo.qHeadOriginal);
    tilingParam[TILING_COMPRESSHEAD] = static_cast<uint32_t>(mmInfo.compressHead);
    tilingParam[TILING_QUANTYPE] = static_cast<uint32_t>(param.quantType);
    tilingParam[TILING_DATA_SHAPE_TYPE] = static_cast<uint32_t>(mmInfo.dataShapeType);
}

void GetPaBatchTiling(const OpParam::PagedAttention &param, const PagedAttentionInfo &mmInfo, uint32_t *tilingParam,
                      uint32_t &qBlkNum)
{
    if (mmInfo.qSeqLen == nullptr) {
        return;
    }
    auto maskType = param.maskType;
    uint64_t maskOffset = 0;
    uint64_t qOffset = 0;
    int32_t maxQ = 0;
    int32_t maxMSlice = 0;
    int32_t numTokensPad = (mmInfo.numTokens + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
    for (int32_t batchIdx = 0; batchIdx < mmInfo.batch; batchIdx++) {
        int32_t qSeqLen = *(mmInfo.qSeqLen + batchIdx);
        int32_t qSeqLenAligned = (qSeqLen + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
        maxQ = qSeqLenAligned > maxQ ? qSeqLenAligned : maxQ;
        int32_t mSlice = std::min((BLOCK_LIMIT / std::max(mmInfo.embeddingSize, mmInfo.blockSize) /
                                  NZ_BLOCK_SIZE) * NZ_BLOCK_SIZE, qSeqLenAligned);
        mSlice = std::min(M_LIMIT, mSlice);
        maxMSlice = std::max(maxMSlice, mSlice);
        uint32_t curQBlockNum = static_cast<uint32_t>(((qSeqLen + mSlice - 1) / mSlice));
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ ] = static_cast<uint32_t>(qSeqLen);
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM1] = curQBlockNum * mmInfo.numHeads;
        qBlkNum += curQBlockNum;
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM2] = qBlkNum * mmInfo.numHeads;
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM3] = GetHigh32Bit(qOffset);
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM4] = GetLoww32Bit(qOffset);
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM5] = GetHigh32Bit(maskOffset);
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM6] = GetLoww32Bit(maskOffset);
        tilingParam[TILING_HEAD_SIZE_NZ + batchIdx * TILING_PARA_SIZE_NZ + NUM7] = static_cast<uint32_t>(mSlice);
        uint64_t incOffset = static_cast<uint64_t>(qSeqLen) * NZ_BLOCK_SIZE;
        qOffset += incOffset;
        maskOffset += incOffset;
    }
    if (maskType == OpParam::PagedAttention::MASK_TYPE_MASK_FREE) {
        tilingParam[TILING_MASK_STRIDE] = 128;
        tilingParam[TILING_DECODE_TYPE] = maxQ >= maxMSlice ? static_cast<uint32_t>(CalcType::CALC_TYPE_PREFILL) :
                                      static_cast<uint32_t>(CalcType::CALC_TYPE_MIX);
    } else {
        tilingParam[TILING_MASK_STRIDE] = (maskType == OpParam::PagedAttention::MASK_TYPE_LOOK_AHEAD) ?
                                    static_cast<uint32_t>(numTokensPad) : static_cast<uint32_t>(maxQ);
        tilingParam[TILING_DECODE_TYPE] = maxQ > maxMSlice ? static_cast<uint32_t>(CalcType::CALC_TYPE_PREFILL) :
                                      static_cast<uint32_t>(CalcType::CALC_TYPE_MIX);
    }
}

void GetPaBlockTilingDefault(uint32_t *tilingParam, const PagedAttentionInfo &mmInfo,
                             const uint32_t taskNum, const uint32_t blockDim)
{
    uint32_t taskNumPerCore = taskNum / blockDim;
    uint32_t tailTaskNum = taskNum % blockDim;
    uint32_t taskStart = 0;
    uint32_t taskEnd = 0;
    for (uint32_t blockIdx = 0; blockIdx < blockDim; blockIdx++) {
        taskStart = taskEnd;
        taskEnd = taskEnd + (blockIdx < tailTaskNum ? taskNumPerCore + 1 : taskNumPerCore);
        tilingParam[NUM10 + blockIdx * NUM4] =  taskStart;
        tilingParam[NUM11 + blockIdx * NUM4] = taskEnd;
        tilingParam[NUM12 + blockIdx * NUM4] = static_cast<uint32_t>(taskStart / mmInfo.numHeads);
        tilingParam[NUM13 + blockIdx * NUM4] = static_cast<uint32_t>((taskEnd - 1) / mmInfo.numHeads);
    }
}

void GetPaBlockTilingParallel(uint32_t *tilingParam, const PagedAttentionInfo &mmInfo,
                              const uint32_t taskNum, const uint32_t blockDim)
{
    uint32_t taskNumPerCore = taskNum / blockDim;
    uint32_t tailTaskNum = taskNum % blockDim;
    uint32_t taskStart = 0;
    uint32_t taskEnd = 0;
    uint32_t startBatch = 0;
    for (uint32_t blockIdx = 0; blockIdx < blockDim; blockIdx++) {
        taskStart = taskEnd;
        taskEnd = taskEnd + (blockIdx < tailTaskNum ? taskNumPerCore + 1 : taskNumPerCore);
        tilingParam[NUM10 + blockIdx * NUM4] = taskStart;
        tilingParam[NUM11 + blockIdx * NUM4] = taskEnd;
        tilingParam[NUM12 + blockIdx * NUM4] = startBatch;
        while (
            tilingParam[TILING_HEAD_SIZE_NZ + startBatch * TILING_PARA_SIZE_NZ + NUM2] <= taskEnd &&
            static_cast<int32_t>(startBatch) < mmInfo.batch - 1) {
            startBatch++;
        }
        tilingParam[NUM13 + blockIdx * NUM4] = startBatch;
    }
}

Status GetNzPagedAttentionTiling(const PagedAttentionInfo &mmInfo, uint32_t &blockDim, uint32_t *tilingParam,
    const OpParam::PagedAttention &param, bool is910A)
{
    int64_t taskNumI64 = mmInfo.numHeads * mmInfo.batch;
    auto [tilingHeadNumLimitOffset, tilingMaskStrideOffset, tilingDecodeTypeOffset, unused] = GetNzTilingOffset();
    (void)unused;

    if (is910A) {
        tilingParam[tilingHeadNumLimitOffset] = 1u;
    } else {
        tilingParam[tilingHeadNumLimitOffset] = std::max(1u, HEADNUM_NZ_LIMIT /
                                                    ((mmInfo.embeddingSize + FLOAT_LIMIT - 1) / FLOAT_LIMIT));
    }
    if (mmInfo.qSeqLen != nullptr) {
        uint32_t qBlkNum = 0;
        GetPaBatchTiling(param, mmInfo, tilingParam, qBlkNum);
        taskNumI64 = mmInfo.numHeads * static_cast<int64_t>(qBlkNum);
    } else {
        tilingParam[tilingMaskStrideOffset] = BLOCK_SIZE;
        tilingParam[tilingDecodeTypeOffset] = static_cast<uint32_t>(CalcType::CALC_TYPE_DEFAULT);
    }
    MKI_CHECK(taskNumI64 <= UINT32_MAX && taskNumI64 >= 0, "param is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t taskNum = static_cast<uint32_t>(taskNumI64);
    blockDim = taskNum < blockDim ? taskNum : blockDim;
    if (tilingParam[tilingDecodeTypeOffset] == static_cast<uint32_t>(CalcType::CALC_TYPE_PREFILL)) {
        GetPaBlockTilingParallel(tilingParam, mmInfo, taskNum, blockDim);
    } else {
        GetPaBlockTilingDefault(tilingParam, mmInfo, taskNum, blockDim);
    }
    return AtbOps::Status::OkStatus();
}

Status GetPagedAttentionTilingParam(const LaunchParam &launchParam, const PagedAttentionInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize)
{
    bool is910A = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910A ? true : false;
    MKI_CHECK(tilingParam != nullptr, "param is nullptr", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.batch > 0  && mmInfo.numTokens > 0 && mmInfo.numHeads > 0 && mmInfo.embeddingSize > 0,
        "param must > 0", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(mmInfo.numBlocks >= 0 && mmInfo.blockSize >= 0 && mmInfo.maxNumBlocksPerQuery >= 0, "param must >= 0",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    auto param = AnyCast<OpParam::PagedAttention>(launchParam.GetParam());
    uint64_t curTilingParamSize = 0;
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        curTilingParamSize = (TILING_HEAD_SIZE + TILING_PARA_SIZE * mmInfo.batch) * sizeof(uint32_t);
        MKI_CHECK(memset_s(tilingParam, tilingParamSize, 0,
            curTilingParamSize) == EOK, "init tiling failed",
            return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        auto tilingHeadSize = is910A ? TILING_HEAD_SIZE_910A : TILING_HEAD_SIZE_NZ;
        curTilingParamSize = (tilingHeadSize + TILING_PARA_SIZE_NZ * param.qSeqLen.size()) * sizeof(uint32_t);
        MKI_CHECK(memset_s(tilingParam, tilingParamSize, 0,
            curTilingParamSize) == EOK,
            "init tiling failed", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    float tor = mmInfo.tor;
    uint32_t *torPtr = reinterpret_cast<uint32_t *>(&tor);
    int32_t nzFlag = 0;
    if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND ||
        param.type == OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND) {
        GetTilingHead(mmInfo, param, tilingParam, torPtr, nzFlag);
        OP_TILING_CHECK_STATUS_RETURN(GetNdPagedAttentionTiling(mmInfo, blockDim, tilingParam, param));
    } else if (param.type == OpParam::PagedAttention::PAGED_ATTENTION_NZ_MASK) {
        nzFlag = 1;
        GetTilingHead(mmInfo, param, tilingParam, torPtr, nzFlag);
        OP_TILING_CHECK_STATUS_RETURN(GetNzPagedAttentionTiling(mmInfo, blockDim, tilingParam, param, is910A));
    }
    return AtbOps::Status::OkStatus();
}
} // end paged_attention namespace
