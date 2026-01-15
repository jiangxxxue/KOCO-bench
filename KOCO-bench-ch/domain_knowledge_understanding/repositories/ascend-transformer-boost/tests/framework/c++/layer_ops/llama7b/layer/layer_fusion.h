/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLAMA_7BLAYER_FUSION_OPERATION_H
#define LLAMA_7BLAYER_FUSION_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/svector.h>

namespace atb_speed {
namespace llama_7b {
struct LayerFusionParam {
    float rmsNormEps = 1e-5;
    int headNum = 1;
    int dk = 0;
    int layerId = 0;
    float qScale = 1.0;
    atb::SVector<int32_t> seqLen;
    atb::SVector<int32_t> tokenOffset;
    std::string model = "llama7b";
    int rotaryCoeff = 2;
};
} // namespace llama_7b
} // namespace atb_speed
#endif