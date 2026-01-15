/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_H
#define ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/tensor.h"
#include "mki/utils/compare/compare.h"

namespace AtbOps {
namespace OpParam {
struct UnpadFlashAttention {
    enum Type {
        UNPAD_FLASH_ATTENTION_ND = 1,
        UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION = 4,
        UNPAD_FLASH_ATTENTION_ENCODER_ND = 10,
        UNPAD_ALIBI_FLASH_ATTENTION_ND = 11,
        UNPAD_FLASH_ATTENTION_DECODER_ND = 12,
        UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION_DECODER = 13,
        UNPAD_FLASH_ATTENTION_FP32_ND = 2001,
        MULTI_LATENT_ATTENTION_COMBINE_CACHE = 2008,
        MULTI_LATENT_ATTENTION_HIGH_PRECISION_COMBINE_CACHE = 2010,
        UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND = 2012,
        UNPAD_FLASH_ATTENTION_RAZOR_FUSION = 2014,
        RELAY_ATTENTION_DECODER_ND = 14
    };
    enum ScaleType {
        SCALE_TOR = 0,
        SCALE_LOGN = 1,
        SCALE_LOGN_FP32 = 2
    };
    Type type;

    enum QuantType {
        TYPE_QUANT_UNDEFINED = 0,
        TYPE_DEQUANT_FUSION,
        TYPE_QUANT_QKV_OFFLINE,
        TYPE_QUANT_QKV_ONLINE
    };
    QuantType quantType = TYPE_QUANT_UNDEFINED;
    Mki::TensorDType outDataType = Mki::TensorDType::TENSOR_DTYPE_FLOAT16;

    ScaleType scaleType = SCALE_TOR;
    // UNPAD_FLASH_ATTENTION
    int32_t headSize = 0;
    int32_t headDimV = 0;
    int32_t razorLen = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    int32_t tileQ = 0;
    int32_t tileKv = 0;
    int32_t textQLen = 0;
    int32_t textKvLen = 0;
    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;
    std::vector<Mki::Tensor> kTensorList;
    std::vector<Mki::Tensor> vTensorList;
    std::vector<Mki::Tensor> kShareTensorList;
    std::vector<Mki::Tensor> vShareTensorList;
    std::vector<int32_t> kvShareMap;
    std::vector<int32_t> kvShareLen;
    float tor = 0;
    int32_t kvHead = 0;
    // UNPAD_BATCH_DYNAMIC_FLASH_ATTENTION
    std::vector<int32_t> batchRunStatus;
    // clamp 算子
    int32_t isClamp = 0;
    float clampMin = 0;
    float clampMax = 0;

    // swa  feature
    uint32_t windowSize = 0;

    uint32_t isTriuMask = 0;
    // UNPAD_FLASH_ATTENTION_ENCODER_ND
    enum MaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_ALIBI = 2,
        MASK_TYPE_LOOK_AHEAD = 3,
        MASK_TYPE_SWA_NORM = 4,
        MASK_TYPE_SWA_COMPRESS = 5,
        MASK_TYPE_ALIBI_COMPRESS = 6,
        MASK_TYPE_ALIBI_COMPRESS_SQRT = 7,
        MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN = 8,
        MASK_TYPE_ALIBI_COMPRESS_128 = 9,
        MASK_TYPE_CAUSAL_MASK = 10
    };
    MaskType maskType = MASK_TYPE_NORM;
    enum CacheType {
        CACHE_TYPE_NORM = 0,
        CACHE_TYPE_CYCLE = 1
    };
    CacheType cacheType = CACHE_TYPE_NORM;

    uint32_t isAlibiMaskSqrt = 0;
    uint32_t alibiLeftAlign = 0;

    bool compressHead = false;

    enum DataShapeType : int {
        TYPE_BSND = 0,
        TYPE_BNSD
    };
    DataShapeType dataShapeType = TYPE_BSND;

    bool operator==(const UnpadFlashAttention &other) const
    {
        return this->type == other.type && this->headSize == other.headSize && this->qSeqLen == other.qSeqLen &&
               this->kvSeqLen == other.kvSeqLen && Mki::Utils::Compare<float>::IsEqual(this->tor, other.tor) &&
               this->kvHead == other.kvHead && this->batchRunStatus == other.batchRunStatus &&
               this->isClamp == other.isClamp && Mki::Utils::Compare<float>::IsEqual(this->clampMin, other.clampMin) &&
               Mki::Utils::Compare<float>::IsEqual(this->clampMax, other.clampMax) &&
               this->windowSize == other.windowSize && this->cacheType == other.cacheType &&
               this->isTriuMask == other.isTriuMask && this->maskType == other.maskType &&
               this->isAlibiMaskSqrt == other.isAlibiMaskSqrt && this->alibiLeftAlign == other.alibiLeftAlign &&
               this->compressHead == other.compressHead && this->dataShapeType == other.dataShapeType &&
               this->quantType == other.quantType && this->outDataType == other.outDataType &&
               this->scaleType == other.scaleType && this->razorLen  == other.razorLen &&
               this->preTokens == other.preTokens && this->nextTokens  == other.nextTokens &&
               this->tileQ == other.tileQ && this->tileKv == other.tileKv &&
               this->textQLen  == other.textQLen && this->textKvLen == other.textKvLen;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_H