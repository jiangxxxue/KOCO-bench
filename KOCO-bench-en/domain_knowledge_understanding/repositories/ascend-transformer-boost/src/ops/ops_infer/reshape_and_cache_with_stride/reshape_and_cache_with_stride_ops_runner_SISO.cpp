/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_with_stride_ops_runner_SISO.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT_THREE = 3;
static const uint32_t IN_TENSOR_COUNT_FOUR = 4;
ReshapeAndCacheWithStrideOpsRunnerSISO::ReshapeAndCacheWithStrideOpsRunnerSISO(
    const infer::ReshapeAndCacheWithStrideParam &param)
    : OpsRunner("ReshapeAndCacheWithStrideOpsRunnerSISO", RUNNER_TYPE_RESHAPE_AND_CACHE_WITH_STRIDE_SISO), param_(param)
{
    ATB_LOG(INFO) << "ReshapeAndCacheWithStrideOpsRunnerSISO::ReshapeAndCacheWithStrideOpsRunnerSISO called";
    const size_t intensorSize = 5;
    const size_t outtensorSize = 1;
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

Status ReshapeAndCacheWithStrideOpsRunnerSISO::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    Mki::Tensor &keytensor = kernelGraph_.inTensors.at(0);
    size_t kstrideSize = static_cast<size_t>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).Numel());
    keytensor.desc.strides.resize(kstrideSize);
    for (size_t i = 0; i < kstrideSize; ++i) {
        keytensor.desc.strides.at(i) =
            static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).hostData)[i];
        ATB_LOG(INFO) << "kstrides: "
                      << static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).hostData)[i];
    }
    keytensor.desc.offset = static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_FOUR).hostData)[0];
    ATB_LOG(INFO) << "koffset: " << keytensor.desc.offset;
    return NO_ERROR;
}

ReshapeAndCacheWithStrideOpsRunnerSISO::~ReshapeAndCacheWithStrideOpsRunnerSISO() {}
} // namespace atb
