/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_ops_runner_SISO.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
ReshapeAndCacheOpsRunnerSISO::ReshapeAndCacheOpsRunnerSISO(const infer::ReshapeAndCacheParam &param)
    : OpsRunner("ReshapeAndCacheOpsRunnerSISO", RUNNER_TYPE_RESHAPE_AND_CACHE_SISO), param_(param)
{
    ATB_LOG(INFO) << "ReshapeAndCacheOpsRunnerSISO::ReshapeAndCacheOpsRunnerSISO called";
    const std::size_t intensorSize = 3;
    const std::size_t outtensorSize = 1;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    size_t inTensorStart = 0;
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &slotsTensor = kernelGraph_.inTensors.at(inTensorStart++);

    size_t outTensorStart = 0;
    Mki::Tensor &outKeyCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);

    kernelGraph_.nodes.resize(1);
    auto &reshapeAndCacheNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::ReshapeAndCache reshapeAndCacheParam;
    reshapeAndCacheParam.type = AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND_SISO;
    reshapeAndCacheNode.opDesc = {0, "ReshapeAndCacheOperation", reshapeAndCacheParam};

    reshapeAndCacheNode.inTensors = {&keyTensor, &keyCacheTensor, &slotsTensor};
    reshapeAndCacheNode.outTensors = {&outKeyCacheTensor};

    reshapeAndCacheNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

ReshapeAndCacheOpsRunnerSISO::~ReshapeAndCacheOpsRunnerSISO() {}
} // namespace atb
