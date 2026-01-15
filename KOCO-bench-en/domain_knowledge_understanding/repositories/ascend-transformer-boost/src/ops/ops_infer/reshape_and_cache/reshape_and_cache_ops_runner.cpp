/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
ReshapeAndCacheOpsRunner::ReshapeAndCacheOpsRunner(const infer::ReshapeAndCacheParam &param)
    : OpsRunner("ReshapeAndCacheOpsRunner", RUNNER_TYPE_RESHAPE_AND_CACHE), param_(param)
{
    ATB_LOG(INFO) << "ReshapeAndCacheOpsRunner::ReshapeAndCacheOpsRunner called";
    const std::size_t intensorSize =
        param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED ?
            5 :
            (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD ? 7 : 8);
    const std::size_t outtensorSize = 2;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    size_t inTensorStart = 0;
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &slotsTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *winsTensor = param_.compressType != infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED ?
                                  &kernelGraph_.inTensors.at(inTensorStart++) :
                                  &nullTensor_;
    Mki::Tensor *seqLen = param_.compressType != infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED ?
                              &kernelGraph_.inTensors.at(inTensorStart++) :
                              &nullTensor_;
    Mki::Tensor *offsetIndex = param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD_ROPE ?
                                   &kernelGraph_.inTensors.at(inTensorStart++) :
                                   &nullTensor_;

    size_t outTensorStart = 0;
    Mki::Tensor &outKeyCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &outValueCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);

    kernelGraph_.nodes.resize(1);
    auto &reshapeAndCacheNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::ReshapeAndCache reshapeAndCacheParam;
    reshapeAndCacheParam.type = param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED ?
                                    AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_ND :
                                    (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD ?
                                         AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS :
                                         AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_WINS_ROPE);
    reshapeAndCacheNode.opDesc = {0, "ReshapeAndCacheOperation", reshapeAndCacheParam};

    if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD) {
        reshapeAndCacheNode.inTensors = {&keyTensor,   &valueTensor, &keyCacheTensor, &valueCacheTensor,
                                         &slotsTensor, winsTensor,   seqLen};
    } else if (param_.compressType == infer::ReshapeAndCacheParam::COMPRESS_TYPE_KVHEAD_ROPE) {
        reshapeAndCacheNode.inTensors = {&keyTensor,   &valueTensor, &keyCacheTensor, &valueCacheTensor,
                                         &slotsTensor, winsTensor,   seqLen,          offsetIndex};
    } else {
        reshapeAndCacheNode.inTensors = {&keyTensor, &valueTensor, &keyCacheTensor, &valueCacheTensor, &slotsTensor};
    }
    reshapeAndCacheNode.opDesc = {0, "ReshapeAndCacheOperation", reshapeAndCacheParam};
    reshapeAndCacheNode.outTensors = {&outKeyCacheTensor, &outValueCacheTensor};

    reshapeAndCacheNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
}

ReshapeAndCacheOpsRunner::~ReshapeAndCacheOpsRunner() {}
} // namespace atb
