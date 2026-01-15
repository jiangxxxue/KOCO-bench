/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_value_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT = 2;

SetValueOpsRunner::SetValueOpsRunner(const infer::SetValueParam &param)
    : OpsRunner("SetValueOpsRunner", RUNNER_TYPE_SET_VALUE), param_(param)
{
    ATB_LOG(INFO) << "SetValueOpsRunner::SetValueOpsRunner called";
}

SetValueOpsRunner::~SetValueOpsRunner() {}

Status SetValueOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(0);

    int64_t inTensorNum = 0;
    Mki::Tensor &dst = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &src = kernelGraph_.inTensors.at(inTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &copyNode = kernelGraph_.nodes.at(0);

    AsdOps::OpParam::Copy copyParam;
    size_t dstdimSize = opsTensorPack.inTensors[0].desc.dims.size();

    for (size_t i = 0; i < dstdimSize; ++i) {
        copyParam.dstSize.push_back(opsTensorPack.inTensors[1].desc.dims.at(i));
    }

    int64_t dim = 1;
    SVector<int64_t> dimMatul;
    dimMatul.resize(dstdimSize);
    ATB_LOG(INFO) << "dstdimSize:" << dstdimSize;
    for (size_t i = dstdimSize; i > 0; --i) {
        dim *= opsTensorPack.inTensors[0].desc.dims.at(i - 1);
        dimMatul.at(i - 1) = dim;
    }

    copyParam.dstStride.resize(dstdimSize);
    for (size_t i = 0; i < dstdimSize - 1; ++i) {
        copyParam.dstStride.at(i) = dimMatul.at(i + 1) * param_.strides.at(i);
        ATB_LOG(INFO) << "copyParam.dstStride.at(i):" << copyParam.dstStride.at(i);
    }
    copyParam.dstStride.at(dstdimSize - 1) = param_.strides.at(dstdimSize - 1);
    ATB_LOG(INFO) << "copyParam.dstStride.at(i):" << copyParam.dstStride.at(dstdimSize - 1);

    int64_t dstOffset = param_.starts.at(dstdimSize - 1);
    for (size_t i = 0; i < dstdimSize - 1; ++i) {
        if ((std::numeric_limits<int64_t>::max() - dstOffset) / dimMatul.at(i + 1) < param_.starts.at(i)) {
            ATB_LOG(ERROR) << " dstOffset Overflow.";
            return ERROR_INVALID_PARAM;
        }
        dstOffset = dstOffset + param_.starts.at(i) * dimMatul.at(i + 1);
    }
    ATB_LOG(INFO) << "dstOffset" << dstOffset;
    copyParam.dstOffset.push_back(dstOffset);

    copyNode.opDesc = {0, "CopyOperation", copyParam};
    copyNode.inTensors = {&dst, &src};
    copyNode.outTensors = {&dst};
    return NO_ERROR;
}
} // namespace atb