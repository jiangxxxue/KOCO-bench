/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "repeat_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"
#include "atb/utils/tensor_util.h"

namespace atb {
RepeatOpsRunner::RepeatOpsRunner(const infer::RepeatParam &param)
    : OpsRunner("RepeatOpsRunner", RUNNER_TYPE_REPEAT), param_(param)
{
    ATB_LOG(INFO) << "RepeatOpsRunner::RepeatOpsRunner called, param_.multiples:" << param_.multiples;
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);

    Mki::Tensor &operationInTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &repeatNode = kernelGraph_.nodes.at(0);

    repeatNode.opDesc = {0, "ExpandOperation", {}};
    repeatNode.inTensors = {&operationInTensor};
    repeatNode.outTensors = {&operationOutTensor};
    repeatNode.inTensorViewFuncs.resize(repeatNode.inTensors.size());
    repeatNode.inTensorViewFuncs.at(0) = [=](const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims) {
        InTensorViewFunc(oldDims, newDims);
    };
    repeatNode.inferShapePreFunc = [this](Mki::LaunchParam &launchParam) {
        launchParam.SetParam(AsdOps::OpParam::Expand({repeatParam_}));
    };
}

void RepeatOpsRunner::InTensorViewFunc(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    repeatParam_.clear();
    if (param_.multiples.size() < oldDims.size()) {
        ATB_LOG(ERROR) << "RepeatOpsRunner InTensorViewFunc: unexpected param size: " << param_.multiples.size();
        return;
    }

    uint64_t diff = param_.multiples.size() - oldDims.size();
    for (uint64_t i = 0; i < diff; i++) {
        newDims.push_back(1);
        repeatParam_.push_back(param_.multiples.at(i));
    }
    for (uint64_t i = diff; i < param_.multiples.size(); i++) {
        int64_t curShape = oldDims.at(i - diff);
        if (param_.multiples.at(i) != 1 && curShape != 1) {
            newDims.push_back(1);
            repeatParam_.push_back(param_.multiples.at(i));
            newDims.push_back(curShape);
            repeatParam_.push_back(1);
        } else {
            newDims.push_back(curShape);
            repeatParam_.push_back(param_.multiples.at(i));
        }
    }
    for (uint64_t i = 0; i < newDims.size(); i++) {
        if (std::numeric_limits<int64_t>::max() / repeatParam_.at(i) < newDims.at(i)) {
            ATB_LOG(ERROR) << "Repeat Results Size Overflow.";
            return;
        }
        repeatParam_.at(i) *= newDims.at(i);
    }
}

RepeatOpsRunner::~RepeatOpsRunner() {}
} // namespace atb