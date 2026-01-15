/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>

#include "atbops/params/params.h"

#include "mm_deq_swiglu_quant_mm_deq_tiling_data.h"
#include "mm_deq_swiglu_quant_mm_deq_tiling.h"

namespace AtbOps {
Status MmDeqSwigluQuantMmDeqTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);

    const size_t x1Index = 0;
    const size_t scale1Index = 2;

    const size_t mDimInX1 = 0;
    const size_t kDimInX1 = 1;
    const size_t nDimInScale1 = 0;
    int64_t m = launchParam.GetInTensor(x1Index).desc.dims.at(mDimInX1);
    int64_t n = launchParam.GetInTensor(scale1Index).desc.dims.at(nDimInScale1);
    int64_t k = launchParam.GetInTensor(x1Index).desc.dims.at(kDimInX1);
    int64_t nOut = n / 2;

    int64_t k2 = nOut;

    size_t workspaceSize =
        Utils::RoundUp(m * k2 * sizeof(int8_t), GM_ALIGN_BYTE) +  // size of X2
        Utils::RoundUp(m * sizeof(float), GM_ALIGN_BYTE) +        // size of PerTokenScale2
        Utils::RoundUp(MAX_TILE_SIZE * coreNum * WORKSPACE_STAGES * sizeof(int32_t), GM_ALIGN_BYTE) +
        Utils::RoundUp(m * nOut * sizeof(float), GM_ALIGN_BYTE);  // sizeof swiglu out

    auto tilingDataPointer = reinterpret_cast<MmDeqSwigluQuantMmDeqTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPointer, "tilingData should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPointer->m = static_cast<uint32_t>(m);
    tilingDataPointer->n = static_cast<uint32_t>(n);
    tilingDataPointer->k = static_cast<uint32_t>(k);

    kernelInfo.SetBlockDim(coreNum);
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    return Status::OkStatus();
}
}  // namespace AtbOps
