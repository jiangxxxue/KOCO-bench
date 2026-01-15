/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_lccl_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
ReduceScatterLcclRunner::ReduceScatterLcclRunner(const infer::ReduceScatterParam &param, Context &context)
    : LcclRunner("ReduceScatterLcclRunner", RUNNER_TYPE_REDUCE_SCATTER, param.rank, param.rankSize, param.commMode,
                 context, param.commDomain),
      param_(param)
{
    ATB_LOG(INFO) << "ReduceScatterLcclRunner::ReduceScatterLcclRunner called";
}

ReduceScatterLcclRunner::~ReduceScatterLcclRunner() {}

Status ReduceScatterLcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!lccl_) {
        ATB_LOG(ERROR) << "lccl_ is null, rank: " << param_.rank;
        return ERROR_COMM_EMPTY;
    }
    int ret = lccl_->ReduceScatter(runnerVariantPack.inTensors.at(0).deviceData,
                                   runnerVariantPack.outTensors.at(0).deviceData,
                                   Utils::GetTensorNumel(runnerVariantPack.outTensors.at(0)),
                                   GetHcclDtype(runnerVariantPack.inTensors.at(0).desc.dtype),
                                   GetAllReduceType(param_.reduceType), GetExecuteStream(runnerVariantPack.context));
    if (ret == Lcal::LCAL_ERROR_PARA_CHECK_FAIL) {
        ATB_LOG(ERROR) << "ret: " << ret << " LCCL_PARALLEL should be 0 or fasle";
        return ERROR_INVALID_SINGLE_OPERATION_PARAM;
    }
    if (ret != 0) {
        ATB_LOG(ERROR) << "ReduceScatter failed, ret: " << ret;
        return ERROR_RT_FAIL;
    }
    return NO_ERROR;
}
} // namespace atb