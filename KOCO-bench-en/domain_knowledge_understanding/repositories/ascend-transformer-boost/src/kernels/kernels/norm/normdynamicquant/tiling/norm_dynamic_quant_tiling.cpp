/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "norm_dynamic_quant_tiling.h"
#include <cstring>
#include <climits>
#include <mki/utils/env/env.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/norm.h"
#include "norm_dynamic_quant_tiling_data.h"

namespace AsdOps {
constexpr uint32_t NDQ_TILING_KEY_BASE = 2000000000;
constexpr uint32_t NDQ_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16
constexpr uint32_t NDQ_TILING_KEY_ASYMMETRIC_MODE = 10000000; // 0: symmetric; 1: asymmetric
constexpr uint32_t NDQ_TILING_KEY_WITHOUT_BETA = 1000000; // 0: with beta; 1: without beta

constexpr uint32_t SLICE_SIZE_LIMIT = 12288;
constexpr uint32_t BLOCK_ELEM_COUNT = 32;

void PrintTilingInfo(NormDynamicQuantTilingData &tilingData)
{
    MKI_LOG(INFO) << "NormDynamicQuant Tiling Data:"
                  << " numCore " << tilingData.numCore
                  << " numCol " << tilingData.numCol
                  << " numRow "<< tilingData.numRow
                  << " avgFactor " << tilingData.avgFactor
                  << " epsilon " << tilingData.epsilon
                  << " quantMin " << tilingData.quantMin
                  << " numRowPerCore " << tilingData.numRowPerCore;
}

Status NormDynamicQuantTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    NormDynamicQuantTilingData *tilingData =
        reinterpret_cast<NormDynamicQuantTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK_NO_LOG(tilingData != nullptr,
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingData should not be empty"));

    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    tilingData->epsilon = param.epsilon;

    int64_t numRow = 1;
    size_t dimsSize = launchParam.GetInTensor(0).desc.dims.size();
    for (size_t i = 0; i < dimsSize - 1; i++) {
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims[i] > 0 &&
                  numRow <= UINT_MAX / launchParam.GetInTensor(0).desc.dims[i],
                  "numRow is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "numRow is invalid!"));
        numRow *= launchParam.GetInTensor(0).desc.dims[i];
    }
    int64_t numCol = launchParam.GetInTensor(0).desc.dims[dimsSize - 1];
    MKI_CHECK_NO_LOG(numCol > 0 && numCol <= SLICE_SIZE_LIMIT && numCol % BLOCK_ELEM_COUNT == 0,
        return Status::FailStatus(ERROR_INVALID_VALUE, "numCol is invalid!"));

    tilingData->numRow = static_cast<uint32_t>(numRow);
    tilingData->numCol = static_cast<uint32_t>(numCol);

    uint32_t maxCoreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    tilingData->numRowPerCore = static_cast<uint32_t>(Utils::RoundUp(Utils::CeilDiv(tilingData->numRow, maxCoreNum),
                                                                     8U)); // scale offset BLOCKSIZE对齐，float类型
    tilingData->numCore = Utils::CeilDiv(tilingData->numRow, tilingData->numRowPerCore);
    tilingData->avgFactor = float(1.0 / numCol);
    tilingData->quantMin = -128; // default int8 min is -128
    const char *quantMinSetPtr = Mki::GetEnv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinSetPtr != nullptr && strcmp(quantMinSetPtr, "1") == 0) {
        tilingData->quantMin = -127; // set int8 min to -127
    }

    uint64_t tilingKey = NDQ_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? NDQ_TILING_KEY_DTYPE : 0;
    tilingKey += param.isSymmetric ? 0 : NDQ_TILING_KEY_ASYMMETRIC_MODE;
    tilingKey += !CheckEmptyTensor(launchParam.GetInTensor(2)) ? 0 : NDQ_TILING_KEY_WITHOUT_BETA; // 2:beta

    kernelInfo.SetTilingId(tilingKey);
    kernelInfo.SetBlockDim(tilingData->numCore);

    PrintTilingInfo(*tilingData);
    return Status::OkStatus();
}

} // namespace AsdOps