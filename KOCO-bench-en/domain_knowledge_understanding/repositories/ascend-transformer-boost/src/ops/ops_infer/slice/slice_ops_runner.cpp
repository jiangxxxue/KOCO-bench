/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_ops_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"

namespace atb {
SliceOpsRunner::SliceOpsRunner(const infer::SliceParam &param)
    : OpsRunner("SliceOpsRunner", RUNNER_TYPE_SLICE), param_(param)
{
    ATB_LOG(INFO) << "SliceOpsRunner::SliceOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &sliceNode = kernelGraph_.nodes[0];

    Mki::SVector<int64_t> offsets;
    Mki::SVector<int64_t> size;
    TensorUtil::AtbSVector2OpsSVector(param_.offsets, offsets);
    TensorUtil::AtbSVector2OpsSVector(param_.size, size);
    AsdOps::OpParam::Slice sliceNodeParam = {offsets, size};

    sliceNode.opDesc = {0, "SliceOperation", sliceNodeParam};
    sliceNode.inTensors = {&xTensor};
    sliceNode.outTensors = {&outTensor};
}

SliceOpsRunner::~SliceOpsRunner() {}
} // namespace atb