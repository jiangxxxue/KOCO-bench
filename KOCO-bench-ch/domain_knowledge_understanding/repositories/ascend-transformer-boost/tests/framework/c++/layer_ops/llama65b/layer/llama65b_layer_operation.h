/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_SPEED_MODELS_LLAMA65BLAYER_OPERATION_H
#define ATB_SPEED_MODELS_LLAMA65BLAYER_OPERATION_H

#include <atb/atb_infer.h>
#include <atb/utils/log.h>

namespace atb_speed {
enum LlamaLayerFusionParallelTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_POSITIONIDS,
    IN_COS_SIN_TABLE,
    IN_ATTENTIONMASK,
    IN_CACHE_K,
    IN_CACHE_V,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    IN_BATCH_STATUS,
    OUT_LLAMA13BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERNAL_CAST_COS_SIN_TABLE,
    INTERMIDATE_CASTCOS,
    INTERMIDATE_CASTSIN,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPLINEARPARALLELOUT,
};

struct LlamaLayerFusionParallelParam {
    float rmsNormEps = 1e-5;
    int headNum = 1;
    int headDim = 0;
    int rank = 0;
    int rankSize = 1;
    float qkScale = 1.0;
    int rotaryCoeff = 2;
    bool transpose = true;
    void *hcclComm = nullptr;          // only effect when hcclComm is not null
    bool batchRunStatusEnable = false; // enable dynamic batch
    int coderType = 0;                 // 区分encoder还是decoder
    int isTriuMask = 0;
};

atb::Status CreateLlama65BLayerOperation(const LlamaLayerFusionParallelParam &param, atb::Operation **operation);
} // namespace atb_speed
#endif