/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef UNPAD_FLASHATTENTION_TILING_DEPENDENCY_H
#define UNPAD_FLASHATTENTION_TILING_DEPENDENCY_H

#include <mki/tensor.h>
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/status/status.h>
#include <map>
#include "atbops/params/kvcache.h"
#include "atbops/params/unpad_flash_attention.h"

const int32_t SWA_COMPRESS_MASK_SIZE = 512;
const uint32_t TILING_KEY_INDEX = 16;

namespace AtbOps {
using namespace Mki;
constexpr int32_t DOUBLE_PING_PONG_SIZE = 32768 * 8;
constexpr int32_t SPLITM_DOUBLE_PING_PONG_SIZE = 32768 * 16;
constexpr int32_t TILING_PARA_SIZE = 27;
constexpr int32_t HARDWARE_AICORE_NUM = 20;
constexpr int32_t TILING_HEAD_SIZE = 37;
constexpr int32_t TILING_RELAY_HEAD_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_SIZE_INT8 = 32;
constexpr int32_t ND_BATCH_LIMIT = INT32_MAX;
constexpr int32_t LONG_SEQ_LEN = 128;
constexpr int32_t LONG_SEQ_ALIBI_LEN = 256;
constexpr int32_t MAX_EMBEDDING = 576;
constexpr int32_t MLA_THRESHOLD = 256;
constexpr int32_t MLA_COMBINE_NUM = 9;
constexpr int32_t TILING_KEY_INDEX = 16;
constexpr int32_t RELAY_BLOCK_TILING = 8;
constexpr uint32_t SPEC_TILING_KEY = 1 << 20;
constexpr int32_t SPLIT_M_THRESHOLD = 4096;
constexpr int32_t MAX_BATCH = 60;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_IN_BF16_OUT = 2,
    TILING_INT8_IN_HALF_OUT = 3,
    TILING_SWA = 4,
    TILING_SWA_CMP = 5
};

enum class ScaleType {
        SCALE_TOR = 0,
        SCALE_LOGN = 1,
        SCALE_LOGN_FP32 = 2
};


using FaTensor = struct FaTensor {
    Tensor query;
    Tensor kCache;
    Tensor vCache;
    Tensor kShareCache;
    Tensor vShareCache;
    Tensor layerId;
    Tensor mask;
    Tensor alibiCoeff;
    Tensor blockTable;
};

using UnpadFlashAttentionInfo = struct UnpadFlashAttentionTilingParams {
    OpParam::UnpadFlashAttention::Type type;
    int32_t batchSize{0};
    int32_t innerBatchSize{0};
    int32_t embeddingSize{0};
    int32_t blockSize{0};
    int32_t maxNumBlocks{0};
    int32_t embeddingSizeV{0};
    int32_t maxQSeqLen{0};
    int32_t maxSeqLen{0};
    int32_t maxKvSeqLen{0};
    int32_t maxKvShareSeqLen{0};
    int32_t razorLen;
    int32_t preTokens;
    int32_t nextTokens;
    int32_t *qSeq{nullptr};
    int32_t *kvSeq{nullptr};
    int32_t tileQ;
    int32_t tileKv;
    int32_t textQLen;
    int32_t textKvLen;
    float *logNs{nullptr};
    int32_t *batchRunStatus{nullptr};
    float tor{0};
    int32_t kvHead{0};
    int32_t maskStride{0};
    uint32_t blockDim{0};
    int32_t isClamp{0};
    float clampMin{0};
    float clampMax{0};
    uint32_t isTriuMask{0};
    uint32_t headStride{0};
    uint32_t tilingKey{0};
    uint32_t isLongSeq{0};
    uint32_t isAlibiMaskSqrt{0};
    uint32_t alibiCompressOffset{0};
    uint32_t maskType{0};
    uint32_t quantType{0};
    uint32_t alibiLeftAlign{0};
    uint32_t cacheType{0};
    uint32_t windowSize{0};
    bool isNoCache{false};
    bool batchContinuous{true};
    bool isMLA{false};
    bool splitm{false};
    std::vector<Tensor> kTensorList;
    std::vector<Tensor> vTensorList;
    std::vector<Tensor> kShareTensorList;
    std::vector<Tensor> vShareTensorList;
    std::map<int32_t, std::vector<int>, std::greater<int>> batchShareMap;
    std::vector<int> kvShareMap;
    std::vector<int> kvShareLen;
    uint32_t dataShapeType{0};
    OpParam::UnpadFlashAttention::ScaleType scaleType = OpParam::UnpadFlashAttention::SCALE_TOR;
    FaTensor tensors;
};

using AddrOffsets = struct AddressOffsetInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrKSeqOffset = 0;
    uint64_t addrVSeqOffset = 0;
    uint64_t addrKshareSeqOffset = 0;
    uint64_t addrVshareSeqOffset = 0;
    uint64_t addrSSeqOffset = 0;
    uint64_t addrPSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    int32_t totalQBlkNum = 0;
    int64_t block = 0;
};

Status FillTilingParam(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr,
    AddrOffsets& addrOffsets, int32_t kvRealHeads, uint32_t *tilingParam);
Status EncoderFillTilingParam(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                              int32_t kvRealHeads, uint32_t *tilingParam);
void FillTilingOffsetParam(int32_t seqIdx, AddrOffsets &addrOffsets, uint32_t *tilingParam);
void FillAddrOffsets(const AtbOps::UnpadFlashAttentionInfo &mmInfo, AtbOps::AddrOffsets &addrOffsets,
                     int32_t kvRealHeads, int32_t qSeqlen, int32_t kvFactor);
Status DecoderFillTilingParam(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr, AddrOffsets &addrOffsets,
                              int32_t kvRealHeads, uint32_t *tilingParam);
void FillAddr(AtbOps::AddrOffsets &addrOffsets, int32_t qSeqlen, const AtbOps::UnpadFlashAttentionInfo &mmInfo,
              int32_t kvRealHeads);
Status GetUnpadFlashAttentionTilingParam(const UnpadFlashAttentionInfo &mmInfo, uint32_t &blockDim,
                                         uint32_t *tilingParam, uint32_t tilingParamSize);
Status CheckSeqlen(const UnpadFlashAttentionInfo &mmInfo, const AddrOffsets &addrOffsets, int32_t qSeqlen,
                   int32_t kvSeqlen, int32_t seqIdx);
void FillTilingHead(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr,
    AddrOffsets &addrOffsets, int32_t kvRealHeads, uint32_t *tilingParam);
void DecoderSplitCore(const UnpadFlashAttentionInfo &mmInfo, AddrOffsets &addrOffsets,
    int32_t kvRealHeads, uint32_t *tilingParam);
Status GetFillTilingParam(const UnpadFlashAttentionInfo &mmInfo,
                          uint32_t *tilingParam, AddrOffsets &addrOffsets);
Status DecoderFillTilingParamRelay(const UnpadFlashAttentionInfo &mmInfo, const uint32_t &torUptr,
                                   AddrOffsets &addrOffsets, int32_t kvRealHeads, uint32_t *tilingParam);
void SplitCoreRelay(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, uint32_t shareBlockTiling);
void SplitTaskRelay(const UnpadFlashAttentionInfo &mmInfo, uint32_t *tilingParam, int32_t groupNum,
                    uint32_t blockIdx, uint32_t shareBlockTiling);
} // namespace AtbOps
#endif
// UNPAD_FLASHATTENTION_TILING_DEPENDENCY_H
