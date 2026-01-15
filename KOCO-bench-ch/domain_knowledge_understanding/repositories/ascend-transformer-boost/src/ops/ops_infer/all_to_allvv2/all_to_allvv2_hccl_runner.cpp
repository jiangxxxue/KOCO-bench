/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_allvv2_hccl_runner.h"
#include <hccl/hccl.h>
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
AllToAllVV2HcclRunner::AllToAllVV2HcclRunner(const infer::AllToAllVV2Param &param, bool useRankTableFile)
    : HcclRunner(!useRankTableFile ? HcclRunner("AllToAllVV2HcclRunner", RUNNER_TYPE_ALL_TO_ALLVV2, param.rank,
                                                param.rankSize, param.rankRoot, param.commDomain) :
                                     HcclRunner("AllToAllVV2HcclRunner", RUNNER_TYPE_ALL_TO_ALLVV2, param.rank,
                                                param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "AllToAllVV2HcclRunner::AllToAllVV2HcclRunner called";
}

AllToAllVV2HcclRunner::AllToAllVV2HcclRunner(const infer::AllToAllVV2Param &param, HcclComm hcclComm)
    : HcclRunner("AllToAllVV2HcclRunner", hcclComm, RUNNER_TYPE_ALL_TO_ALLVV2),
      param_(param)
{
    ATB_LOG(INFO) << "AllToAllVV2HcclRunner::AllToAllVV2HcclRunner ext called";
}

Status AllToAllVV2HcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!hcclComm_) {
        ATB_LOG(ERROR) << "hcclComm is null, rank: " << param_.rank;
        return ERROR_INTERNAL_ERROR;
    }
    if (!runnerVariantPack.inTensors[0].deviceData || !runnerVariantPack.outTensors[0].deviceData) {
        ATB_LOG(ERROR) << " device tensor is null";
        return ERROR_INVALID_PARAM;
    }
    HcclResult ret = HcclAlltoAllV (
        runnerVariantPack.inTensors[0].deviceData,
        runnerVariantPack.inTensors[1].hostData,
        // 2 : input2
        runnerVariantPack.inTensors[2].hostData,
        GetHcclDtype(runnerVariantPack.inTensors[0].desc.dtype),
        runnerVariantPack.outTensors[0].deviceData,
        // 3 : input3
        runnerVariantPack.inTensors[3].hostData,
        // 4 : input4
        runnerVariantPack.inTensors[4].hostData,
        GetHcclDtype(runnerVariantPack.outTensors[0].desc.dtype),
        hcclComm_.get(),
        GetExecuteStream(runnerVariantPack.context));
    if (ret != HCCL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "AlltoAllV Execute failed, HcclResult:" << ret;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

AllToAllVV2HcclRunner::~AllToAllVV2HcclRunner() {}
}  // namespace atb