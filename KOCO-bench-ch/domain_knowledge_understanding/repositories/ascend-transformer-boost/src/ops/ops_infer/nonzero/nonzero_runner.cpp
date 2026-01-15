/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nonzero_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;
NonzeroRunner::NonzeroRunner(const infer::NonzeroParam &param)
    : OpsRunner("NonzeroRunner", RUNNER_TYPE_NONZERO), param_(param)
{
    ATB_LOG(INFO) << "NonzeroRunner::NonzeroRunner called:";
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);

    Mki::Tensor &intensor = kernelGraph_.inTensors.at(0);

    Mki::Tensor &yOutTensor = kernelGraph_.outTensors.at(0);
    Mki::Tensor &numTruesOutTensor = kernelGraph_.outTensors.at(1);

    kernelGraph_.nodes.resize(1);
    auto &nonzeroNode = kernelGraph_.nodes.at(0);

    nonzeroNode.opDesc = {0, "NonzeroOperation", param_};
    nonzeroNode.inTensors = {&intensor};
    nonzeroNode.outTensors = {&yOutTensor, &numTruesOutTensor};
}

NonzeroRunner::~NonzeroRunner() {}
} // namespace atb
