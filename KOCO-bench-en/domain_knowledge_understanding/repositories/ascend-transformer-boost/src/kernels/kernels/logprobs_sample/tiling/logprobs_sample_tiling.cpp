/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "tiling_data.h"
#include "logprobs_sample_tiling.h"

static constexpr uint32_t WORKSPACE_PARAM_INDEX = 5;
static constexpr uint32_t WORKSPACE_SIZE_PER_CORE = 32;

namespace AsdOps {
void FillTilingParam(const LaunchParam &launchParam, LogprobsSampleTilingData &tilingData)
{
    const auto &probsDim = launchParam.GetInTensor(INPUT_SORTED_PROBS_INDEX).desc.dims;
    uint32_t batchSize = static_cast<uint32_t>(probsDim[0]);
    uint32_t probsSize = static_cast<uint32_t>(probsDim[1]);
    tilingData.batchSize = batchSize;
    tilingData.probsSize = probsSize;
}

void SetWorkspaceSize(uint32_t batchSize, KernelInfo &kernelInfo)
{
    uint32_t maxCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t runsPerCore = (batchSize + maxCore - 1) / maxCore;
    uint32_t usedCore = (batchSize + runsPerCore - 1) / runsPerCore;

    kernelInfo.SetBlockDim(usedCore);

    // 根据文档所述，IBWait接口的gmWorkspace申请的空间最少要求为：
    // 核数 * 32Bytes * eventID_max + blockIdx_max * 32Bytes + 32Bytes
    // （eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）
    // 这里eventID只会用0，而blockIdx_max + 1即为核数，所以workSpace的空间即为核数 * 32
    uint64_t workspaceSize = usedCore * WORKSPACE_SIZE_PER_CORE;
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    kernelInfo.SetMemsetInfo(WORKSPACE_PARAM_INDEX, workspaceSize);
}

Status LogprobsSampleTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    LogprobsSampleTilingData *tilingDataPtr =
        reinterpret_cast<AsdOps::LogprobsSampleTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t dimSize = launchParam.GetInTensor(INPUT_SORTED_PROBS_INDEX).desc.dims.size();
    MKI_CHECK(dimSize >= 2, "dimSize should equal or greater than 2",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    FillTilingParam(launchParam, *tilingDataPtr);

    SetWorkspaceSize(tilingDataPtr->batchSize, kernelInfo);

    bool tilingKey = launchParam.GetInTensor(INPUT_SORTED_PROBS_INDEX).desc.dtype == TENSOR_DTYPE_BF16;
    kernelInfo.SetTilingId((uint64_t)tilingKey);

    MKI_LOG(INFO) << " usedCore: " << kernelInfo.GetBlockDim()
                  << ", batchSize: " << tilingDataPtr->batchSize
                  << ", probsSize: "<< tilingDataPtr->probsSize;
    return Status::OkStatus();
}
} // namespace AsdOps