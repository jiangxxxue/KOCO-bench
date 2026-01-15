/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "unpad_tiling.h"
#include <mki/kernel_info.h>
#include "tiling_data.h"

namespace AtbOps {
using namespace Mki;
const uint64_t SYSTEM_WORKSPACE_SIZE = 16;
void FillTilingParam(const LaunchParam &launchParam, UnpadTilingData *tilingDataPtr)
{
    tilingDataPtr->padLength = launchParam.GetInTensor(0).desc.dims[1];
    tilingDataPtr->batch = launchParam.GetInTensor(0).desc.dims[0];
}

Status UnpadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    UnpadTilingData *tilingDataPtr = reinterpret_cast<UnpadTilingData *>(kernelInfo.GetTilingHostAddr());
    FillTilingParam(launchParam, tilingDataPtr);
    kernelInfo.SetBlockDim(1);

    uint64_t sysWorkspaceSize = SYSTEM_WORKSPACE_SIZE;
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize};
    return Status::OkStatus();
}
} // namespace AtbOps