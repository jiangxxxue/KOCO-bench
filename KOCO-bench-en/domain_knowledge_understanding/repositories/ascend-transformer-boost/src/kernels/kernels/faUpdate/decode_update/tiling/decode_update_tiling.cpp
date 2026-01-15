/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "decode_update_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/faUpdate.h"
#include "tiling_data.h"

//  hDim = 128， spAligned = 8时，单核b*hc最大7，所以取3
constexpr uint32_t MIN_BLOCK_LENGTH = 3;

namespace AsdOps {
//  b*s*hc做tiling, 对应sp block的个数
Status DecodeUpdateTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    DecodeUpdateTilingData *tilingDataPointer =
        reinterpret_cast<AsdOps::DecodeUpdateTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPointer != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    if (launchParam.GetParam().Type() != typeid(OpParam::FaUpdate)) {
        return Status::FailStatus(
            ERROR_ATTR_INVALID_TYPE,
            "Failed to check faUpdate param, type of specificParam is not equals to OpParam::FaUpdate");
    }

    // sp, B*s*hc in1
    // sp, B*s*hc, hd in2

    const uint32_t sp = launchParam.GetInTensor(0).desc.dims.at(0);
    const uint32_t totalLength = launchParam.GetInTensor(0).desc.dims.at(1);
    const uint32_t hd = launchParam.GetInTensor(1).desc.dims.at(2);
    MKI_LOG(INFO) << "b*s*hc totalLength is " << totalLength;
    MKI_LOG(INFO) << "sp is " << sp;
    MKI_LOG(INFO) << "hd is " << hd;
    
    //  tiling计算过程
    uint32_t blockDims = 0;
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    blockDims = std::min<uint32_t>((totalLength + MIN_BLOCK_LENGTH - 1) / MIN_BLOCK_LENGTH, coreNum);
    tilingDataPointer->formerNum = static_cast<uint32_t>(totalLength % blockDims);
    tilingDataPointer->tailNum = static_cast<uint32_t>(blockDims - tilingDataPointer->formerNum);
    tilingDataPointer->tailLength = static_cast<uint32_t>(totalLength / blockDims);
    tilingDataPointer->formerLength = static_cast<uint32_t>(
    tilingDataPointer->formerNum == 0 ? 0 : tilingDataPointer->tailLength + 1);
    tilingDataPointer->hDim = static_cast<uint32_t>(hd);
    tilingDataPointer->sp = static_cast<uint32_t>(sp);
    tilingDataPointer->totalLength = static_cast<uint32_t>(totalLength);
    MKI_LOG(INFO) << "BlockDims is " << blockDims;
    MKI_LOG(INFO) << "formerLength is " << tilingDataPointer->formerLength; // 多干活的核心任务数
    MKI_LOG(INFO) << "tailLength is " << tilingDataPointer->tailLength; // 每个核心的任务数
    MKI_LOG(INFO) << "formerNum is " << tilingDataPointer->formerNum; // 多干活的核心数
    MKI_LOG(INFO) << "tailNum is " << tilingDataPointer->tailNum; // 少干活的核心数
    MKI_LOG(INFO) << "hDim is " << tilingDataPointer->hDim;
    MKI_LOG(INFO) << "sp is " << tilingDataPointer->sp;
    MKI_LOG(INFO) << "totalLength is " << tilingDataPointer->totalLength;
    kernelInfo.SetBlockDim(blockDims);
    return Status::OkStatus();
}
} // namespace AsdOps