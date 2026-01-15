/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTENTION_NZ_TILING_DEP_H
#define FLASH_ATTENTION_NZ_TILING_DEP_H

#include <mki/tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/status/status.h>
#include <unordered_map>
#include "atbops/params/kvcache.h"
#include "atbops/params/unpad_flash_attention_nz.h"

namespace AtbOps {
using namespace Mki;
const int32_t NZ_BLOCK_SIZE = 16;
const int32_t MAX_EMBEDDING = 256;
const int32_t NZ_REAL_CORE_TILING_SIZE = 20;
const int32_t LONG_SEQ_LEN = 128;
const size_t NZ_BATCH_LIMIT = 2000;
constexpr int32_t LONG_SEQ_ALIBI_LEN = 256;
constexpr int32_t SWA_MASK_COMPRESS_SIZE = 512;
constexpr uint32_t USE_MAX_CORE_NUM = 9;

using UnpadFlashAttentionNzInfo = struct UnpadFlashAttentionNzTilingParams {
    OpParam::UnpadFlashAttentionNz::Type type;
    OpParam::UnpadFlashAttentionNz::ScaleType scaleType;
    OpParam::UnpadFlashAttentionNz::PrecType precType;
    int32_t batchSize;      // 实际的网络 batch_size，sequence 数量
    int32_t innerBatchSize; // head size
    int32_t embeddingSize;
    int32_t maxSeqLen;
    int32_t maxKVSeqLen;
    int32_t qTokens;
    int32_t kvHeads;
    int32_t *qSeq;  // q矩阵sequence length 序列，序列长度 = batch_size
    int32_t *kvSeq; // kv矩阵sequence length 序列，序列长度 = batch_size
    int32_t maskStride = 0;
    bool isCache = true;
    float tor;
    uint32_t headMaskStride = 0;
    uint32_t batchMaskStride = 0;
    uint32_t qTight = 0; // temporary flag, will be removed after models adapt to kernel changes
    uint32_t isTriu = 0;
    uint32_t isLongSeq = 0;
    uint32_t alibiCompressOffset = 0;
    uint32_t alibiLeftAlign = 0;
    uint32_t maskType = 0;
    uint32_t isAlibiMaskSqrt = 0;
    uint32_t dataDimOrder = 0;
    bool batchContinuous = true;
    std::vector<Tensor> kTensorList;
    std::vector<Tensor> vTensorList;
    uint32_t windowLen = 0; // SWA window 大小，默认为0
    uint32_t cacheType = 0; // Decoder cache类型，默认为Norm
};

using AddrOffsetsNz = struct AddressOffsetNzInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrKSeqOffset = 0;
    uint64_t addrVSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
    int32_t totalQBlkNum = 0;
};

int32_t GetNzRealCoreTilingOffset();
uint32_t GetTilingKeyIndex();
void FillCoreAttribs(const UnpadFlashAttentionNzInfo &mmInfo, const uint32_t &torUptr, const AddrOffsetsNz &addrOffsets,
                     uint32_t *tilingParam, const uint32_t nzRealCoreNum);
Status FillTilingParamRealCore(const UnpadFlashAttentionNzInfo &mmInfo, const uint32_t &torUptr,
                               AddrOffsetsNz &addrOffsets, uint32_t *tilingParam, const uint32_t nzRealCoreNum);
Status GetUnpadFlashAttentionTilingParam(const UnpadFlashAttentionNzInfo mmInfo, uint32_t &blockDim,
                                         uint32_t *tilingParam, uint32_t tilingParamSize);

} // namespace AtbOps
#endif
