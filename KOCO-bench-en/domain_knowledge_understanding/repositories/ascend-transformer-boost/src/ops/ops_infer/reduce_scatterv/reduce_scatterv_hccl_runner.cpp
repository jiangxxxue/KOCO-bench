/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatterv_hccl_runner.h"
#include <hccl/hccl.h>
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
ReduceScatterVHcclRunner::ReduceScatterVHcclRunner(const infer::ReduceScatterVParam &param, bool useRankTableFile)
    : HcclRunner(!useRankTableFile ? HcclRunner("ReduceScatterVHcclRunner", RUNNER_TYPE_REDUCE_SCATTERV, param.rank,
                                                param.rankSize, param.rankRoot, param.commDomain) :
                                     HcclRunner("ReduceScatterVHcclRunner", RUNNER_TYPE_REDUCE_SCATTERV, param.rank,
                                                param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "ReduceScatterVHcclRunner::ReduceScatterVHcclRunner called";
}

ReduceScatterVHcclRunner::ReduceScatterVHcclRunner(const infer::ReduceScatterVParam &param, HcclComm hcclComm)
    : HcclRunner("ReduceScatterVHcclRunner", hcclComm, RUNNER_TYPE_REDUCE_SCATTERV), param_(param)
{
    ATB_LOG(INFO) << "ReduceScatterVHcclRunner::ReduceScatterVHcclRunner ext called";
}

Status ReduceScatterVHcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!hcclComm_) {
        ATB_LOG(ERROR) << "hcclComm is null, rank: " << param_.rank;
        return ERROR_INTERNAL_ERROR;
    }

    if (!runnerVariantPack.inTensors[0].deviceData || !runnerVariantPack.outTensors[0].deviceData) {
        ATB_LOG(ERROR) << " device tensor is null";
        return ERROR_INVALID_PARAM;
    }
    HcclResult ret = HcclReduceScatterV(runnerVariantPack.inTensors[0].deviceData,
        runnerVariantPack.inTensors[1].hostData,  // sendCounts
        runnerVariantPack.inTensors[2].hostData,  // sdispls
        runnerVariantPack.outTensors.at(0).deviceData,
        *(static_cast<int64_t *>(runnerVariantPack.inTensors[3].hostData)),  // recvCount
        GetHcclDtype(runnerVariantPack.inTensors[0].desc.dtype),
        GetAllReduceType(param_.reduceType),
        hcclComm_.get(),
        GetExecuteStream(runnerVariantPack.context));
    if (ret == HCCL_E_PARA) {
        ATB_LOG(ERROR) << "ReduceScatterV hccl Parameter error, ret: " << ret;
        return ERROR_CANN_ERROR;
    }
    if (ret != HCCL_SUCCESS) {
        ATB_LOG(ERROR) << "ReduceScatterV hccl Execute failed, ret: " << ret;
        return ERROR_CANN_ERROR;
    }
    return NO_ERROR;
}

ReduceScatterVHcclRunner::~ReduceScatterVHcclRunner()
{}
} // namespace atb