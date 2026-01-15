/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_PAGEDATTENTION_H
#define ATBOPS_PARAMS_PAGEDATTENTION_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/types.h"
#include "mki/utils/compare/compare.h"
namespace AtbOps {
namespace OpParam {
struct PagedAttention {
    enum Type {
        PAGED_ATTENTION_MASK_ND = 2002,
        PAGED_ATTENTION_NZ_MASK = 2003,
        PAGED_ATTENTION_NZ = 2004,
        PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND = 2005,
        PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND = 2006,
        PAGED_ATTENTION_ND = 8
    };
    Type type;
    enum ScaleType {
        SCALE_TOR = 0,
        SCALE_LOGN = 1,
        SCALE_LOGN_FP32 = 2
    };
    ScaleType scaleType = SCALE_TOR;

    uint32_t isTriuMask = 0;
    std::vector<int8_t> identityM = {0};
    int32_t headSize = 0;
    float tor = 0;
    int32_t kvHead = 0;
    int32_t headDimV = 0;

    enum MaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_ALIBI = 2,
        MASK_TYPE_LOOK_AHEAD = 3,
        MASK_TYPE_MASK_FREE = 4
    };

    MaskType maskType = MASK_TYPE_NORM;

    enum QuantType {
        TYPE_QUANT_UNDEFINED = 0,   //!< 默认值，不与量化融合
        TYPE_DEQUANT_FUSION = 1,    //!< 与反量化融合, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_OFFLINE = 2, //!< 离线INT8量化, 只支持Atlas 800I A2
        TYPE_QUANT_QKV_ONLINE = 3   //!< 在线INT8量化, 只支持Atlas 800I A2
    };
    QuantType quantType = TYPE_QUANT_UNDEFINED;

    Mki::TensorDType outDataType = Mki::TENSOR_DTYPE_FLOAT16; // 只有量化能用， 可选FLOAT16：1  BFLOAT16:27

    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;
    std::vector<int32_t> batchRunStatus;
    bool compressHead = false;
    enum DataShapeType {
        BSND = 0, // BSND
        BNSD = 1  // BNSD
    };
    DataShapeType dataShapeType = BSND;

    bool operator==(const PagedAttention &other) const
    {
        return this->headSize == other.headSize && this->scaleType == other.scaleType &&
               this->qSeqLen == other.qSeqLen && this->kvSeqLen == other.kvSeqLen && this->type == other.type &&
               this->maskType == other.maskType && this->compressHead == other.compressHead &&
               Mki::Utils::Compare<float>::IsEqual(this->tor, other.tor) && this->kvHead == other.kvHead &&
               this->batchRunStatus == other.batchRunStatus && this->identityM == other.identityM &&
               this->isTriuMask == other.isTriuMask && this->quantType == other.quantType &&
               this->outDataType == other.outDataType && this->dataShapeType == other.dataShapeType &&
               this->scaleType == other.scaleType && this->headDimV == other.headDimV;
    }
};
} // namespace OpParam
} // namespace AtbOps
#endif // ATBOPS_PARAMS_PAGEDATTENTION_H