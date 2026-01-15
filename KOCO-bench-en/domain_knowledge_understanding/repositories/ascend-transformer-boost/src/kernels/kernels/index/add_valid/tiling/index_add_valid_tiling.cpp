/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
#include <mki/utils/math/math.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/index.h"
#include "tiling_data.h"
#include "index_add_valid_tiling.h"

static constexpr uint64_t INDEXADDVALID_WKSP_TENSOR_IDX = 5;

namespace AsdOps {
    constexpr uint32_t INDEX_ADD_VALID_TILING_KEY_BASE = 2000000000;
    constexpr uint32_t INDEX_ADD_VALID_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16

Status IndexAddValidTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    IndexAddValidTilingData *tilingDataPointer =
        reinterpret_cast<AsdOps::IndexAddValidTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPointer != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    if (launchParam.GetParam().Type() != typeid(OpParam::Index)) {
        return Status::FailStatus(
            ERROR_ATTR_INVALID_TYPE,
            "Failed to check index add valid param, type of specificParam is not equals to OpParam::Index");
    }
    const uint64_t indicesValid = launchParam.GetInTensor(0).desc.dims.at(0);
    const uint64_t indicesTotal = launchParam.GetInTensor(1).desc.dims.at(0);
    const uint64_t valueSize = launchParam.GetInTensor(2).desc.dims.at(1);
    MKI_LOG(INFO) << "indicesTotal is " << indicesTotal
                  << " indicesValid is " << indicesValid
                  << " valueSize is " << valueSize;
    MKI_CHECK(indicesValid > 0 && indicesValid <= std::numeric_limits<uint32_t>::max(), "indicesValid is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "indicesValid should not be zero"));
    MKI_CHECK(indicesTotal <= std::numeric_limits<uint32_t>::max(), "indicesTotal is invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "indicesTotal is invalid"));
    MKI_CHECK(valueSize <= MAX_VALUE_SIZE, "valueSize exceeds limit value",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "valueSize exceeds limit value"));
    tilingDataPointer->indicesValid = static_cast<uint32_t>(indicesValid);
    tilingDataPointer->indicesTotal = static_cast<uint32_t>(indicesTotal);
    tilingDataPointer->valueSize = static_cast<uint32_t>(valueSize);
    uint64_t sysWorkspaceSize = static_cast<uint64_t>(indicesValid) * valueSize * sizeof(float);
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize};
    kernelInfo.SetMemsetInfo(INDEXADDVALID_WKSP_TENSOR_IDX, sysWorkspaceSize);
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t blockDims = 0;
    if (indicesValid < coreNum) {
        blockDims = indicesValid;
    } else {
        blockDims = coreNum;
    }
    kernelInfo.SetBlockDim(blockDims);

    uint64_t tilingKey = INDEX_ADD_VALID_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ?
        INDEX_ADD_VALID_TILING_KEY_DTYPE : 0;
    kernelInfo.SetTilingId(tilingKey);
   
    return Status::OkStatus();
}
} // namespace AsdOps