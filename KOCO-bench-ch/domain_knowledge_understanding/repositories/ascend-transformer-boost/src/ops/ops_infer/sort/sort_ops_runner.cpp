/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_ops_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"

namespace atb {
static const uint64_t OUT_TENSOR_COUNT = 2;

SortOpsRunner::SortOpsRunner(const infer::SortParam &param)
    : OpsRunner("SortOpsRunner", RUNNER_TYPE_SORT), param_(param)
{
    ATB_LOG(INFO) << "SortOpsRunner::SortOpsRunner called";
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);
    int64_t outTensorNum = 0;
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &indices = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &sortNode = kernelGraph_.nodes[0];

    Mki::SVector<int32_t> num;
    TensorUtil::AtbSVector2OpsSVector(param_.num, num);
    AsdOps::OpParam::Sort sortNodeParam = {num};

    sortNode.opDesc = {0, "SortOperation", sortNodeParam};
    sortNode.inTensors = {&xTensor};
    sortNode.outTensors = {&outTensor, &indices};
}

SortOpsRunner::~SortOpsRunner() {}
} // namespace atb