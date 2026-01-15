/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_SPEED_MODELS_LLAMA7B_POSITION_EMBEDDING_1DSPLIT_FUSION_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA7B_POSITION_EMBEDDING_1DSPLIT_FUSION_OPERATION_H
#include <atb/atb_infer.h>

namespace atb_speed {
namespace llama_7b {
struct PositionEmbedding1dFusionParam {
    std::string model = "llama";
    int64_t rotaryCoeff = 2;
    int64_t headNum = 0;
};

atb::Status PositionEmbeddingFusionOperation(const PositionEmbedding1dFusionParam &param,
                                             atb::Operation **operation);
} // namespace llama_7b
} // namespace atb_speed
#endif