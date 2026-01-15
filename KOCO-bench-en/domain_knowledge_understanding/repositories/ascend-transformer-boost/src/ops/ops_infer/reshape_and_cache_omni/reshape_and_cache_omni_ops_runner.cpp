/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_omni_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
ReshapeAndCacheOmniOpsRunner::ReshapeAndCacheOmniOpsRunner(const infer::ReshapeAndCacheOmniParam &param)
    : OpsRunner("ReshapeAndCacheOmniOpsRunner", RUNNER_TYPE_RESHAPE_AND_CACHE_OMNI), param_(param)
{
    ATB_LOG(INFO) << "ReshapeAndCacheOmniOpsRunner::ReshapeAndCacheOmniOpsRunner called";
    const std::size_t intensorSize = 8;
    const std::size_t outtensorSize = 2;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    size_t inTensorStart = 0;
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &slotsTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &winsTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &offsetIndex = kernelGraph_.inTensors.at(inTensorStart++);

    size_t outTensorStart = 0;
    Mki::Tensor &outKeyCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &outValueCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);

    kernelGraph_.nodes.resize(1);
    auto &reshapeAndCacheOmniNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::ReshapeAndCache reshapeAndCacheParam;
    reshapeAndCacheParam.type = AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_OMNI_COMPRESS;
    reshapeAndCacheOmniNode.opDesc = {0, "ReshapeAndCacheOperation", reshapeAndCacheParam};

    reshapeAndCacheOmniNode.inTensors = {&keyTensor, &valueTensor, &keyCacheTensor, &valueCacheTensor,
                                         &slotsTensor, &winsTensor, &seqLen, &offsetIndex};
    reshapeAndCacheOmniNode.outTensors = {&outKeyCacheTensor, &outValueCacheTensor};

    reshapeAndCacheOmniNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

ReshapeAndCacheOmniOpsRunner::~ReshapeAndCacheOmniOpsRunner() {}

void ReshapeAndCacheOmniOpsRunner::SetParam(const Mki::Any &param)
{
    infer::ReshapeAndCacheOmniParam newParam = Mki::AnyCast<infer::ReshapeAndCacheOmniParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "ReshapeAndCacheOmniOpsRunner param changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb
