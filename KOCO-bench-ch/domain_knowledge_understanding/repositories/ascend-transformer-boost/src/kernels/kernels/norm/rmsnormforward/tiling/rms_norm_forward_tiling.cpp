/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <climits>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "tiling_data.h"
#include "rms_norm_forward_tiling.h"

namespace AsdOps {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t DTYPE_KEY_WEIGHT = 10;
constexpr uint32_t UB_FACTOR_B16 = 12288;
constexpr uint32_t UB_FACTOR_B32 = 10240;
constexpr uint32_t GAMMA_INDEX = 1;

template <typename T>
static T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}
Status RmsNormForwardTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto tilingDataPointer =
        reinterpret_cast<RmsNormForwardTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPointer != nullptr, "tilingdata should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata should not be empty"));
    uint32_t numCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t ubFactor = UB_FACTOR_B16;
    auto xShape = launchParam.GetInTensor(0).desc.dims;
    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    const float epsilon = param.epsilon;
    MKI_CHECK(launchParam.GetInTensor(GAMMA_INDEX).desc.Numel() <= UINT_MAX, "numel of gamma is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "numel of gamma is invalid!"));
    uint32_t numCol = static_cast<uint32_t>(launchParam.GetInTensor(GAMMA_INDEX).desc.Numel());
    float avgFactor = (numCol == 0) ? 0 : 1.0 / numCol;
    size_t xDimNum = launchParam.GetInTensor(0).desc.dims.size();
    auto gammaShape = launchParam.GetInTensor(GAMMA_INDEX).desc.dims;
    size_t gammaDimNum = launchParam.GetInTensor(GAMMA_INDEX).desc.dims.size();
    int64_t numRowTmp = 1;
    int32_t startDim = xDimNum - gammaDimNum;
    MKI_CHECK(startDim >= 0, "x's dim should great than gamma's dim!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "x's dim should great than gamma's dim!"));
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        MKI_CHECK(xShape[i] > 0 && numRowTmp <= UINT_MAX / xShape[i],
                  "numRowTmp is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "numRowTmp is invalid!"));
        numRowTmp *= xShape[i];
    }
    for (size_t i = startDim; i < xDimNum; i++) {
        MKI_CHECK(xShape[i] == gammaShape[i - startDim],
                  "gamma shape invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "gamma shape invalid!"));
    }
    uint32_t numRow = static_cast<uint32_t>(numRowTmp);
    uint32_t blockFactor = 1;
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    uint32_t useCoreNum = CeilDiv(numRow, blockFactor);

    kernelInfo.SetBlockDim(useCoreNum);

    uint32_t rowFactor = 64;
    auto dataType = launchParam.GetInTensor(0).desc.dtype;
    uint32_t dtypeKey = DTYPE_KEY_FP16;
    if (dataType == TENSOR_DTYPE_FLOAT) {
        dtypeKey = DTYPE_KEY_FP32;
        ubFactor = UB_FACTOR_B32;
    }
    if (dataType == TENSOR_DTYPE_BF16) {
        dtypeKey = DTYPE_KEY_BF16;
    }
    uint32_t splitD = numCol > ubFactor ? 1 : 0;

    uint32_t tilingKey = dtypeKey * DTYPE_KEY_WEIGHT + splitD;
    kernelInfo.SetTilingId(tilingKey);
    tilingDataPointer->numRow = numRow;
    tilingDataPointer->numCol = numCol;
    tilingDataPointer->blockFactor = blockFactor;
    tilingDataPointer->rowFactor = rowFactor;
    tilingDataPointer->ubFactor = ubFactor;
    tilingDataPointer->epsilon = epsilon;
    tilingDataPointer->avgFactor = avgFactor;
    return AsdOps::Status::OkStatus();
    }

} // namespace AsdOps