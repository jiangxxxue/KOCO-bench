/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_hccl_runner.h"
#include <hccl/hccl.h>
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
SendHcclRunner::SendHcclRunner(const infer::SendParam &param, bool useRankTableFile)
    : HcclRunner(!useRankTableFile ?
                     HcclRunner("SendHcclRunner", RUNNER_TYPE_SEND, param.rank, param.rankSize, param.rankRoot,
                                param.commDomain) :
                     HcclRunner("SendHcclRunner", RUNNER_TYPE_SEND, param.rank, param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "SendHcclRunner::SendHcclRunner called";
}

SendHcclRunner::SendHcclRunner(const infer::SendParam &param, HcclComm hcclComm)
    : HcclRunner("SendHcclRunner", hcclComm, RUNNER_TYPE_SEND), param_(param)
{
    ATB_LOG(INFO) << "SendHcclRunner::SendHcclRunner ext called";
}

Status SendHcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!hcclComm_) {
        ATB_LOG(ERROR) << "hcclComm is null, rank: " << param_.rank;
        return ERROR_COMM_EMPTY;
    }

    if (!runnerVariantPack.inTensors[0].deviceData) {
        ATB_LOG(ERROR) << " device tensor is null";
        return ERROR_INVALID_PARAM;
    }

    HcclResult ret =
        HcclSend(runnerVariantPack.inTensors[0].deviceData, Utils::GetTensorNumel(runnerVariantPack.inTensors[0]),
                 GetHcclDtype(runnerVariantPack.inTensors[0].desc.dtype), param_.destRank, hcclComm_.get(),
                 GetExecuteStream(runnerVariantPack.context));
    if (ret != HCCL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "HcclSend Execute failed, HcclResult:" << ret;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

SendHcclRunner::~SendHcclRunner() {}
} // namespace atb