/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_to_all_lccl_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"

namespace atb {
AllToAllLcclRunner::AllToAllLcclRunner(const infer::AllToAllParam &param, Context &context)
    : LcclRunner("AllToAllLcclRunner", RUNNER_TYPE_ALL_TO_ALL, param.rank, param.rankSize, param.commMode, context,
                 param.commDomain),
      param_(param)
{
    ATB_LOG(INFO) << "AllToAllLcclRunner::AllToAllLcclRunner called";
}

Status AllToAllLcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!lccl_) {
        ATB_LOG(ERROR) << "lccl_ is null, rank: " << param_.rank;
        return ERROR_COMM_EMPTY;
    }
    if (!runnerVariantPack.inTensors[0].deviceData || !runnerVariantPack.outTensors[0].deviceData) {
        ATB_LOG(ERROR) << " device tensor is null";
        return ERROR_INVALID_TENSOR_ADDR;
    }
    int ret = 0;
    ATB_LOG(INFO) << "lccl alltoall transpose: " << param_.transpose;
    if (!param_.transpose) {
        ret = lccl_->All2All(runnerVariantPack.inTensors[0].deviceData, runnerVariantPack.outTensors.at(0).deviceData,
                             Utils::GetTensorNumel(runnerVariantPack.inTensors.at(0)),
                             GetHcclDtype(runnerVariantPack.inTensors.at(0).desc.dtype),
                             GetExecuteStream(runnerVariantPack.context));
    } else {
        int64_t width = runnerVariantPack.inTensors[0].desc.shape.dims[1];
        int64_t burstlen = width / param_.rankSize;
        ret = lccl_->All2All(runnerVariantPack.inTensors[0].deviceData, runnerVariantPack.outTensors.at(0).deviceData,
                             Utils::GetTensorNumel(runnerVariantPack.inTensors.at(0)), static_cast<int>(burstlen),
                             static_cast<int>(width), GetHcclDtype(runnerVariantPack.inTensors.at(0).desc.dtype),
                             GetExecuteStream(runnerVariantPack.context));
    }
    if (ret == Lcal::LCAL_ERROR_PARA_CHECK_FAIL) {
        ATB_LOG(ERROR) << "ret: " << ret << " LCCL_PARALLEL should be 0 or fasle";
        return ERROR_INVALID_SINGLE_OPERATION_PARAM;
    }
    if (ret != 0) {
        ATB_LOG(ERROR) << "lccl AlltoAll Execute failed, LcclResult:" << ret;
        return ERROR_RT_FAIL;
    }
    return NO_ERROR;
}

AllToAllLcclRunner::~AllToAllLcclRunner() {}
} // namespace atb