/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_SPEED_MODELS_CHATGLM6BLAYER_ENCODER_OPERATION_H
#define ATB_SPEED_MODELS_CHATGLM6BLAYER_ENCODER_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/utils/log.h>

namespace atb_speed {
struct ChatGlm6BLayerEncoderParam {
    double layerNormEps = 0;
    int headNum = 0;
    int headDim = 0;
    float qScale = 1;
    float qkScale = 1;
    float residualAddScale = 0;
};

atb::Status CreateChatGlm6BLayerEncoderOperation(const ChatGlm6BLayerEncoderParam &param, atb::Operation **operation);
} // namespace atb_speed
#endif