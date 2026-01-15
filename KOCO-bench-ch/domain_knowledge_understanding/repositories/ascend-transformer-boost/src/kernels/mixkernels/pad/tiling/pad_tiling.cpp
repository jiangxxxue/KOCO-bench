/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/status/status.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "pad_tiling.h"

namespace AtbOps {
using namespace Mki;
void FillTilingParam(const LaunchParam &launchParam, PadTilingData *tilingDataPtr)
{
    tilingDataPtr->padLength = launchParam.GetInTensor(DIM_3).desc.dims[1];
    tilingDataPtr->batch = launchParam.GetInTensor(DIM_3).desc.dims[0];
    tilingDataPtr->hiddenDim = launchParam.GetInTensor(0).desc.dims[1];
}

Status PadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    PadTilingData *tilingDataPtr = reinterpret_cast<PadTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    FillTilingParam(launchParam, tilingDataPtr);
    kernelInfo.SetBlockDim(1);
    uint64_t sysWorkspaceSize = 16;
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);
    return Status::OkStatus();
}
} // namespace AtbOps