/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_NORM_H
#define ASDOPS_PARAMS_NORM_H

#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/compare/compare.h>

namespace AsdOps {
namespace OpParam {
struct Norm {
    enum NormType {
        NORM_UNDEFINED = 0,
        LAYER_NORM,
        RMS_NORM,
        RMS_NORM_FORWARD,
        RMS_NORM_BACKWARD,
        GATHER_PRE_RMS_NORM,
    };
    NormType normType;
    // layernorm
    int32_t beginNormAxis = 0;
    int32_t beginParamsAxis = 0;
    // postlayernorm
    // opsMode = 0 : high precision
    // opsMode = 1 : high performance
    size_t opsMode = 0;
    float epsilon = 0.1f;
    float zoomScaleValue = 1.0f;
    // post/pre rmsnorm
    // precisionMode = 0 : high precision(weight fp32)
    // precisionMode = 1 : high performance(weight fp16)
    uint32_t precisionMode = 0;
    uint32_t gemmaMode = 0;
    bool inGamma = false; // CohereLayerNorm, LayernormF16Kernel, LayernormBF16Kernel, LayernormF32Kernel,
                          // PostLayernormF16Kernel, LayernormF16QuantKernel, PostLayernormF16QuantKernel,
                          // RmsPreNormQuantKernel, RmsNormKernel, RmsNormQuantKernel, PreRmsNormKernel,
                          // PostRmsNormKernel, RmsPostNormQuantKernel, AdalayernormKernel
    bool inBeta = false;  // LayernormF16Kernel, LayernormBF16Kernel, LayernormF32Kernel, PostLayernormF16Kernel,
                          // LayernormF16QuantKernel, PostLayernormF16QuantKernel, RmsNormQuantKernel,
                          // PreRmsNormKernel, PostRmsNormKernel, AdalayernormKernel
    bool inRes = false;   // PostLayernormF16Kernel, PostLayernormF16QuantKernel, RmsPreNormQuantKernel,
                          // PreRmsNormKernel, PostRmsNormKernel, RmsPostNormQuantKernel
    bool inNormBias = false;  // RmsPreNormQuantKernel
    bool outMean = false;     // LayernormF16Kernel, LayernormBF16Kernel, LayernormF32Kernel
    bool outVarience = false; // LayernormF16Kernel, LayernormBF16Kernel, LayernormF32Kernel
    bool outResQuant = false; // LayernormF16QuantKernel, PostLayernormF16QuantKernel
    bool outRes = false;      // RmsPreNormQuantKernel, PreRmsNormKernel, RmsPostNormQuantKernel
    bool isDynamicQuant = false; // rmsnorm + dynamicquant、layernorm + dynamicquant
    bool isSymmetric = true; // symmetric or asymmetric

    bool operator==(const Norm &other) const
    {
        return this->normType == other.normType && this->beginNormAxis == other.beginNormAxis &&
               this->beginParamsAxis == other.beginParamsAxis && this->opsMode == other.opsMode &&
               Mki::Utils::Compare<float>::IsEqual(this->epsilon, other.epsilon) &&
               Mki::Utils::Compare<float>::IsEqual(this->zoomScaleValue, other.zoomScaleValue) &&
               this->inGamma == other.inGamma &&
               this->inBeta == other.inBeta &&
               this->inRes == other.inRes &&
               this->inNormBias == other.inNormBias &&
               this->outMean == other.outMean &&
               this->outVarience == other.outVarience &&
               this->outResQuant == other.outResQuant &&
               this->outRes == other.outRes &&
               this->precisionMode == other.precisionMode &&
               this->gemmaMode == other.gemmaMode &&
               this->isDynamicQuant == other.isDynamicQuant;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif