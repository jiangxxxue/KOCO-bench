/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rms_norm_tiling.h"
#include <cstring>
#include <climits>
#include <limits>
#include <mki/utils/env/env.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"
#include "kernels/norm/common/common_tiling_data.h"

static constexpr uint32_t SLICE_SIZE = 8192; // 用于切分过长的行
static constexpr uint32_t NUM_TWO = 2;

namespace {
template <typename T> T CeilDiv(T x, T y) { return y == 0 ? x : (x + y - 1) / y; }
} // namespace
namespace AsdOps {
constexpr uint32_t RMS_NORM_TILING_KEY_TAILMODE = 100000; // 0:short tail; 1:Long tail
constexpr uint32_t RMS_NORM_TILING_KEY_BASE = 10000;
constexpr uint32_t RMS_NORM_TILING_KEY_GEMMAMODE = 1000;    // 0:gemmamode no; 1：gemmamode yes
constexpr uint32_t RMS_NORM_TILING_KEY_PRECISIONMODE = 100; // 0:precisionmode; 1:performance mode
constexpr uint32_t RMS_NORM_TILING_KEY_DTYPE = 10;          // 0:fp16;1:bf16

void PrintRmsNormTiling(const RmsNormCommonTilingData &tilingDataPtr)
{
    MKI_LOG(INFO) << "RmsNorm Tiling Data:"
                  << " numCore " << tilingDataPtr.numCore
                  << " numCol " << tilingDataPtr.numCol
                  << " numRow " << tilingDataPtr.numRow
                  << " avgFactor " << tilingDataPtr.avgFactor
                  << " epsilon " << tilingDataPtr.epsilon
                  << " sliceSize " << tilingDataPtr.sliceSize
                  << " mode " << tilingDataPtr.mode
                  << " precisionMode  " << tilingDataPtr.precisionMode
                  << " gemmaMode " << tilingDataPtr.gemmaMode;
}
uint64_t ComputeTilingKey(uint32_t gemmaMode, uint32_t precisionMode, bool isShortTail, const LaunchParam &launchParam)
{
    uint64_t tilingKey = RMS_NORM_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? RMS_NORM_TILING_KEY_DTYPE : 0;
    if (isShortTail) {
        tilingKey += (gemmaMode == 1) ? RMS_NORM_TILING_KEY_GEMMAMODE : 0;
        tilingKey += (precisionMode == 0) ? RMS_NORM_TILING_KEY_PRECISIONMODE : 0;
    } else {
        tilingKey += RMS_NORM_TILING_KEY_TAILMODE;
    }
    return tilingKey;
}

void SetNonContiguousTenor(RmsNormCommonTilingData &tilingDataPtr, const LaunchParam &launchParam)
{
    const auto& xStrides = launchParam.GetInTensor(0).desc.strides;
    const auto& shape = launchParam.GetInTensor(0).desc.dims;
    uint32_t dimNum = xStrides.size();
    if (xStrides.empty() || dimNum == 1 || xStrides[dimNum - NUM_TWO] == shape[dimNum - 1]) {
        tilingDataPtr.xDimNum = 0;
    } else {
        for (size_t i = 0; i < xStrides.size(); ++ i) {
            tilingDataPtr.xStrides[i] = xStrides[i];
        }
        tilingDataPtr.xDimNum = dimNum;
        tilingDataPtr.xOffset = launchParam.GetInTensor(0).desc.offset;
    }
}

Status RmsNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    RmsNormCommonTilingData* tilingDataPtr = reinterpret_cast<RmsNormCommonTilingData*>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingdata should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata should not be empty"));
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    const Mki::SVector<int64_t> &shape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(!shape.empty(), "shape should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "shape should not be empty"));
    tilingDataPtr->numRow = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        MKI_CHECK(shape[i] > 0 && tilingDataPtr->numRow <= UINT_MAX / shape[i],
                  "numRow is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "numRow is invalid!"));
        tilingDataPtr->numRow *= shape[i];
    }
    MKI_CHECK(shape[shape.size() - 1] > 0 && shape[shape.size() - 1] <= UINT_MAX, "numCol invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->numCol = static_cast<uint32_t>(shape[shape.size() - 1]);
    tilingDataPtr->numCore = CeilDiv(tilingDataPtr->numRow, CeilDiv(tilingDataPtr->numRow, coreNum));
    MKI_CHECK(tilingDataPtr->numCore <= UINT_MAX - tilingDataPtr->numRow, "numRow + numCore is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "numRow + numCore is invalid!"));
    tilingDataPtr->avgFactor = static_cast<float>(1.0 / tilingDataPtr->numCol);
    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    MKI_CHECK(CheckRmsNormTiling(launchParam), "Failed to Check rmsnormtiling",
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "Failed to Check rmsnormtiling"));
    tilingDataPtr->gemmaMode = param.gemmaMode;
    tilingDataPtr->precisionMode = param.precisionMode;
    tilingDataPtr->epsilon = param.epsilon;
    bool isShortTail = tilingDataPtr->numCol <= SLICE_SIZE;
    tilingDataPtr->sliceSize = isShortTail ? tilingDataPtr->numCol : SLICE_SIZE;
    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
        if (PlatformInfo::Instance().GetPlatformType() != PlatformType::ASCEND_910B) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Only 910B supports BF16.");
        }
        tilingDataPtr->mode = 1;
    } else { tilingDataPtr->mode = 0; }
    tilingDataPtr->quantMin = -128; // default int8 min is -128
    const char *quantMinSetPtr = Mki::GetEnv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinSetPtr != nullptr && strcmp(quantMinSetPtr, "1") == 0) {
        tilingDataPtr->quantMin = -127; // set int8 min to -127
    }
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);
    SetNonContiguousTenor(*tilingDataPtr, launchParam);
    uint64_t tilingKey = ComputeTilingKey(tilingDataPtr->gemmaMode, tilingDataPtr->precisionMode, isShortTail,
                                          launchParam);
    MKI_LOG(INFO) << "post rmsnorm tilingKey is : " << tilingKey;
    kernelInfo.SetTilingId(tilingKey);
    PrintRmsNormTiling(*tilingDataPtr);
    return Status::OkStatus();
}

bool CheckRmsNormTiling(const LaunchParam &launchParam)
{
    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    MKI_CHECK(param.epsilon > 0, "Invalid parameter: epsilon cannot be " + std::to_string(param.epsilon),
        return false);
    MKI_CHECK(param.precisionMode == 0 || param.precisionMode == 1, "Invalid parameter: precisionMode cannot be " +
        std::to_string(param.precisionMode), return false);
    MKI_CHECK(param.gemmaMode == 0 || param.gemmaMode == 1,
        "Invalid parameter: gemmaMode cannot be " + std::to_string(param.gemmaMode), return false);
    MKI_CHECK(!(param.precisionMode == 1 && launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16),
        "Invalid parameter:not support dtype is bfloat16 and precisionMode is 1", return false);
    MKI_CHECK(!(param.gemmaMode == 1 && param.precisionMode == 1),
        "Invalid parameter:not support gemmaMode is 1 and precisionMode is 1", return false);
    return true;
}
} // namespace AsdOps
