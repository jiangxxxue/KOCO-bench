/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "block_copy_ops_runner.h"
namespace atb {
BlockCopyOpsRunner::BlockCopyOpsRunner(const infer::BlockCopyParam &param)
    : OpsRunner("BlockCopyOpsRunner", RUNNER_TYPE_BLOCK_COPY), param_(param)
{
    ATB_LOG(INFO) << "BlockCopyOpsRunner::BlockCopyOpsRunner called";
    kernelGraph_.inTensors.resize(5); // dim:5
    size_t inTensorId = 0;
    Mki::Tensor &kCache = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &vCache = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &srcBlockIndices = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &dstBlockIndices = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &cumSum = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.nodes.resize(1);
    auto &blockCopyNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::BlockCopy blockCopyNodeParam = {};

    blockCopyNode.opDesc = {0, "BlockCopyOperation", blockCopyNodeParam};
    blockCopyNode.inTensors = {&kCache, &vCache, &srcBlockIndices, &dstBlockIndices, &cumSum};
    blockCopyNode.outTensors = {&kCache, &vCache};
}

BlockCopyOpsRunner::~BlockCopyOpsRunner() {}

} // namespace atb
