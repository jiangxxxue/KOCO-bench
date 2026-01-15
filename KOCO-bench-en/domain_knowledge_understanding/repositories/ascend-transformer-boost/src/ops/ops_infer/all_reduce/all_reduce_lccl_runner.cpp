/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_lccl_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
AllReduceLcclRunner::AllReduceLcclRunner(const infer::AllReduceParam &param, Context &context)
    : LcclRunner("AllReduceLcclRunner", RUNNER_TYPE_ALL_REDUCE, param.rank, param.rankSize, param.commMode,
                 context, param.commDomain),
      param_(param)
{
    ATB_LOG(INFO) << "AllReduceLcclRunner::AllReduceLcclRunner called";
}

AllReduceLcclRunner::~AllReduceLcclRunner() {}

Status AllReduceLcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!lccl_) {
        ATB_LOG(ERROR) << "lccl_ is null, rank: " << param_.rank;
        return ERROR_COMM_EMPTY;
    }
    const int32_t offsetPosition = 1;
    const int32_t scalePosition = 2;
    int ret;
    if (runnerVariantPack.inTensors.size() > 1) {
        ret = lccl_->AllReduce(
            runnerVariantPack.inTensors.at(0).deviceData, runnerVariantPack.outTensors.at(0).deviceData,
            Utils::GetTensorNumel(runnerVariantPack.inTensors.at(0)),
            GetHcclDtype(runnerVariantPack.inTensors.at(0).desc.dtype), GetAllReduceType(param_.allReduceType),
            GetExecuteStream(runnerVariantPack.context), GetHcclDtype(runnerVariantPack.outTensors.at(0).desc.dtype),
            runnerVariantPack.inTensors.at(offsetPosition).deviceData,
            Utils::GetTensorNumel(runnerVariantPack.inTensors.at(offsetPosition)),
            runnerVariantPack.inTensors.at(scalePosition).deviceData);
    } else {
        ret = lccl_->AllReduce(runnerVariantPack.inTensors.at(0).deviceData,
                               runnerVariantPack.outTensors.at(0).deviceData,
                               Utils::GetTensorNumel(runnerVariantPack.inTensors.at(0)),
                               GetHcclDtype(runnerVariantPack.inTensors.at(0).desc.dtype),
                               GetAllReduceType(param_.allReduceType), GetExecuteStream(runnerVariantPack.context));
    }
    if (ret == Lcal::LCAL_ERROR_PARA_CHECK_FAIL) {
        ATB_LOG(ERROR) << "ret: " << ret << " LCCL_PARALLEL should be 0 or fasle";
        return ERROR_INVALID_SINGLE_OPERATION_PARAM;
    }
    if (ret != 0) {
        ATB_LOG(ERROR) << "AllReduce failed, ret: " << ret;
        return ERROR_RT_FAIL;
    }
    return NO_ERROR;
}
} // namespace atb