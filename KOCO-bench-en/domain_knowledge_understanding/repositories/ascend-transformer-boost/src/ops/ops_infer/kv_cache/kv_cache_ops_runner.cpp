/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kv_cache_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
KvCacheOpsRunner::KvCacheOpsRunner(const infer::KvCacheParam &param)
    : OpsRunner("KvCacheOpsRunner", RUNNER_TYPE_KV_CACHE), param_(param)
{
    ATB_LOG(INFO) << "KvCacheOpsRunner::KvCacheOpsRunner called";
    const std::size_t intensorSize = 5;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(0);

    int inTensorStart = 0;
    Mki::Tensor &newKvTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &layerIdTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &pastTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &tokenOffsetTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &seqLenTensor = kernelGraph_.inTensors.at(inTensorStart++);

    kernelGraph_.nodes.resize(1);
    auto &kvCacheNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::KVCache kvCacheParam;
    kvCacheParam.type = AtbOps::OpParam::KVCache::KVCACHE_ND;
    kvCacheParam.qSeqLen = {};
    kvCacheParam.kvSeqLen = {};
    kvCacheParam.batchRunStatus = {};
    kvCacheNode.opDesc = {0, "KVCacheOperation", kvCacheParam};
    kvCacheNode.inTensors = {&newKvTensor, &layerIdTensor, &pastTensor, &tokenOffsetTensor, &seqLenTensor};
    kvCacheNode.outTensors = {&pastTensor};
    kvCacheNode.inTensorViewFuncs.resize(kvCacheNode.inTensors.size());
    kvCacheNode.inTensorViewFuncs[0] = [](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        if (oldDims.size() == 2) { // 2: intensor维数为2时
            newDims = oldDims;
        } else if (oldDims.size() == 4) {                                             // 4: intensor维数为4时
            newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)}; // 2, 3: 设置张量形状
        } else {
            ATB_LOG(ERROR) << "kvcache operation intensor[0] only support dimNum = 2 or 4.";
        }
    };
}

KvCacheOpsRunner::~KvCacheOpsRunner() {}
} // namespace atb