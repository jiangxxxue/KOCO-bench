/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATBOPS_PARAMS_RINGMLA_H
#define ATBOPS_PARAMS_RINGMLA_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/types.h"
#include "mki/utils/compare/compare.h"
namespace AtbOps {
namespace OpParam {
struct RINGMLA {
    enum Type {
        SPLIT_CACHE = 0,
        PREFILL_SPLIT_CACHE = 1,
    };
    Type type;
    int32_t headSize = 0;
    float tor = 0;
    int32_t kvHead = 0;
    uint32_t isTriuMask = 0;
    std::vector<Mki::Tensor> kTensorList;
    std::vector<Mki::Tensor> vTensorList;

    enum MaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_ALIBI = 2,
        MASK_TYPE_LOOK_AHEAD = 3,
        MASK_TYPE_MASK_FREE = 4
    };

    MaskType maskType = MASK_TYPE_NONE;

    enum QuantType {
        TYPE_QUANT_UNDEFINED = 0,
        TYPE_DEQUANT_FUSION,
        TYPE_QUANT_QKV_OFFLINE,
        TYPE_QUANT_QKV_ONLINE
    };
    QuantType quantType = TYPE_QUANT_UNDEFINED;

    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;

    int32_t isRing = 0;

    bool operator==(const RINGMLA &other) const
    {
        return this->headSize == other.headSize && this->isTriuMask == other.isTriuMask &&
               this->qSeqLen == other.qSeqLen && this->kvSeqLen == other.kvSeqLen && this->type == other.type &&
               Mki::Utils::Compare<float>::IsEqual(this->tor, other.tor) && this->kvHead == other.kvHead &&
               this->isRing == other.isRing;
    }
};
} // namespace OpParam
} // namespace AtbOps
#endif // ATBOPS_PARAMS_RINGMLA_H