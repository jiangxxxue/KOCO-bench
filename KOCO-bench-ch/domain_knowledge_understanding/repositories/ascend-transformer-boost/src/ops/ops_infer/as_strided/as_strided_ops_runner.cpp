/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "as_strided_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
AsStridedOpsRunner::AsStridedOpsRunner(const infer::AsStridedParam &param)
    : OpsRunner("AsStridedOpsRunner", RUNNER_TYPE_ASSTRIDED), param_(param)
{
    ATB_LOG(INFO) << "AsStridedOpsRunner::AsStridedOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &aTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &asstridedNode = kernelGraph_.nodes.at(0);

    Mki::SVector<int64_t> size;
    Mki::SVector<int64_t> stride;
    Mki::SVector<int64_t> offset;
    TensorUtil::AtbSVector2OpsSVector(param.size, size);
    TensorUtil::AtbSVector2OpsSVector(param.stride, stride);
    TensorUtil::AtbSVector2OpsSVector(param.offset, offset);
    AsdOps::OpParam::AsStrided opParam = {size, stride, offset};
    asstridedNode.opDesc = {0, "AsStridedOperation", opParam};
    asstridedNode.inTensors = {&aTensor};
    asstridedNode.outTensors = {&operationOutTensor};
}

AsStridedOpsRunner::~AsStridedOpsRunner() {}
} // namespace atb