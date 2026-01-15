/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_NZ_H
#define ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_NZ_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/tensor.h"
#include "mki/utils/compare/compare.h"

namespace AtbOps {
namespace OpParam {
struct UnpadFlashAttentionNz {
    enum Type {
        UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION = 4,
        UNPAD_FLASH_ATTENTION_NZ = 6,
        UNPAD_ALIBI_FLASH_ATTENTION_NZ = 7,
        UNPAD_FLASH_ATTENTION_NZ_ENCODER = 17,
        UNPAD_FLASH_ATTENTION_NZ_DECODER = 18,
        PAGED_ATTENTION_NZ_MASK = 2003,
        UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE = 2005
    };
    Type type;

    enum ScaleType {
        SCALE_TOR = 0,
        SCALE_LOGN = 1
    };
    ScaleType scaleType = SCALE_TOR;
    enum PrecType {
        BMM1_FP16_EXP_FP32 = 0,
        BMM1_FP32_EXP_FP32 = 1,
        BMM2_ONLINE_SOFTMAX_FP16 = 4
    };
    PrecType precType = BMM1_FP16_EXP_FP32;
    // DATA
    enum DataDimOrder : int {
        TYPE_BSND = 0,
        TYPE_BNSD
    };
    DataDimOrder dataDimOrder = TYPE_BSND;

    // UNPAD_FLASH_ATTENTION
    int32_t headSize = 0;
    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;
    std::vector<Mki::Tensor> kTensorList;
    std::vector<Mki::Tensor> vTensorList;

    float tor = 0;
    int32_t kvHead = 0;
    int32_t windowSize = 0;
    // UNPAD_BATCH_DYNAMIC_FLASH_ATTENTION
    std::vector<int32_t> batchRunStatus;

    uint32_t isTriuMask = 0;
    // PAGED_ATTENTION_MASK_NZ, UNPAD_FLASH_ATTENTION_ENCODER_NZ
    enum MaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_ALIBI = 2,
        MASK_TYPE_LOOK_AHEAD = 3,
        MASK_TYPE_SWA_NORM = 4,
        MASK_TYPE_SWA_COMPRESS = 5
    };
    // MaskType is also need to be added in op_kernel/unpad_flash_attention_common.h to avoid devil num in .cce
    MaskType maskType = MASK_TYPE_NORM;
    enum CacheType {
        CACHE_TYPE_NORM = 0,
        CACHE_TYPE_CYCLE = 1
    };
    CacheType cacheType = CACHE_TYPE_NORM;
    uint32_t alibiLeftAlign = 0;

    uint32_t isAlibiMaskSqrt = 0;
    bool compressHead = false;
    bool operator==(const UnpadFlashAttentionNz &other) const
    {
        return this->type == other.type && this->headSize == other.headSize && this->qSeqLen == other.qSeqLen &&
               this->kvSeqLen == other.kvSeqLen && this->kvHead == other.kvHead &&
               Mki::Utils::Compare<float>::IsEqual(this->tor, other.tor) && this->isTriuMask == other.isTriuMask &&
               this->maskType == other.maskType && this->isAlibiMaskSqrt == other.isAlibiMaskSqrt &&
               this->compressHead == other.compressHead && this->batchRunStatus == other.batchRunStatus &&
               this->alibiLeftAlign == other.alibiLeftAlign && this->dataDimOrder == other.dataDimOrder &&
               this->scaleType == other.scaleType && this->windowSize == other.windowSize &&
               this->precType == other.precType && this->cacheType == other.cacheType;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_UNPAD_FLASH_ATTENTION_NZ_H