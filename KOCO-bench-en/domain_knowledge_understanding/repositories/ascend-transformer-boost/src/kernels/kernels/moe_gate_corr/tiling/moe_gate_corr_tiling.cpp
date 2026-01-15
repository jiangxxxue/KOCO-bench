/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_gate_corr_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "tiling_data.h"

namespace AsdOps {
void PrintMoeGateCorrTiling(const KernelInfo &kernelInfo)
{
    MoeGateCorrTilingData *tilingData =
        reinterpret_cast<AsdOps::MoeGateCorrTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_LOG(INFO) << "block dim = " << kernelInfo.GetBlockDim();
    MKI_LOG(INFO) << "m, k, n = " << tilingData->m << " " << tilingData->k << " " << tilingData->n;
    return;
}

Status MoeGateCorrTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);

    const auto &inputADim = launchParam.GetInTensor(0).desc.dims;
    const auto &inputBDim = launchParam.GetInTensor(1).desc.dims;
    MKI_CHECK(inputADim.size() == 2, "inTensor0 dims invalid.", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(inputBDim.size() == 2, "inTensor1 dims invalid.", return Status::FailStatus(ERROR_INVALID_VALUE));

    // NT
    int64_t m = inputADim.at(0);
    int64_t n = inputBDim.at(0);
    int64_t k = inputADim.at(1);

    uint8_t *tiling = kernelInfo.GetTilingHostAddr();
    MKI_CHECK(tiling, "TilingHostAddr should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));
    auto tilingDataPointer = reinterpret_cast<MoeGateCorrTilingData *>(tiling);
    MKI_CHECK(tilingDataPointer, "TilingData should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));

    MKI_CHECK(m >= 0 && m <= UINT32_MAX, "Invalid m value.", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPointer->m = static_cast<uint32_t>(m);
    MKI_CHECK(n >= 0 && n <= UINT32_MAX, "Invalid n value.", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPointer->n = static_cast<uint32_t>(n);
    MKI_CHECK(k >= 0 && k <= UINT32_MAX, "Invalid k value.", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPointer->k = static_cast<uint32_t>(k);

    kernelInfo.SetBlockDim(coreNum);
    PrintMoeGateCorrTiling(kernelInfo);
    return Status::OkStatus();
}
} // namespace AsdOps
