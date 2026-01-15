/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NORM_COMMMON_TILING_H
#define NORM_COMMMON_TILING_H

#include <climits>
#include <cstring>
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/env/env.h>
#include <mki/utils/math/math.h>
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/norm.h"
#include "kernels/elewise/quant/quant_tiling/tiling_data.h"
#include "kernels/norm/postlayernorm/tiling/post_layer_norm_tiling.h"
#include "kernels/norm/common/common_tiling_data.h"

namespace AsdOps {
struct NormTilingDataPtrCon {
    uint32_t maxCoreNum{0};
    uint32_t maxUbSize{0};
    uint32_t maxEleFp16{0};
    uint32_t numRow{0};
    uint32_t numCol{0};
    uint32_t numCore{0};
    uint32_t rowWork{0};
    uint32_t nlFirstdimPerCoreNum{0};
};

template <typename T>
inline __attribute__((always_inline)) Status PostLayerNormPtrFunc(T *tilingDataPtr,
    NormTilingDataPtrCon &ptrCon, const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    ptrCon.maxCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    MKI_CHECK(ptrCon.maxCoreNum > 0 && ptrCon.maxCoreNum <= UINT_MAX, "maxCoreNum is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "maxCoreNum is invalid!"));
    ptrCon.maxUbSize = PlatformInfo::Instance().GetUbSize();
    ptrCon.maxEleFp16 = ptrCon.maxUbSize / 2; // buffer * 2
    int64_t tmpNumRow = 1;
    uint32_t dimsSize = launchParam.GetInTensor(0).desc.dims.size();
    MKI_CHECK(dimsSize > 0 && dimsSize <= 8, "dimsSize is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "dimsSize is invalid!"));
    for (size_t i = 0; i < dimsSize - 1; i++) {
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims[i] > 0 &&
                  tmpNumRow <= UINT_MAX / launchParam.GetInTensor(0).desc.dims[i],
                  "tmpNumRow is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "tmpNumRow is invalid!"));
        tmpNumRow *= launchParam.GetInTensor(0).desc.dims[i];
    }
    ptrCon.numRow = static_cast<uint32_t>(tmpNumRow);
    tilingDataPtr->numFirstDim = ptrCon.numRow;
    MKI_CHECK(launchParam.GetInTensor(0).desc.dims[dimsSize - 1] > 0 &&
                 launchParam.GetInTensor(0).desc.dims[dimsSize - 1] <= UINT_MAX, "numCol is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "numCol is invalid!"));
    ptrCon.numCol = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[dimsSize - 1]);

    tilingDataPtr->numLastDim = ptrCon.numCol;
    ptrCon.numCore = Utils::CeilDiv(ptrCon.numRow, Utils::CeilDiv(ptrCon.numRow, ptrCon.maxCoreNum));
    tilingDataPtr->numCore = ptrCon.numCore;
    kernelInfo.SetBlockDim(ptrCon.numCore);
    ptrCon.rowWork = Utils::CeilDiv(ptrCon.numRow, ptrCon.numCore);
    tilingDataPtr->nlFirstdimPerCore = ptrCon.rowWork;
    ptrCon.nlFirstdimPerCoreNum = tilingDataPtr->nlFirstdimPerCore;
    tilingDataPtr->lFirstdimPerCore = (ptrCon.numRow - ptrCon.nlFirstdimPerCoreNum * (ptrCon.numCore - 1));
    return Status::OkStatus();
}

template <typename T>
inline __attribute__((always_inline)) Status LayerNormPtrFunc(NormTilingDataPtrCon &ptrCon,
                                                              const LaunchParam &launchParam,
                                                              KernelInfo &kernelInfo)
{
    ptrCon.maxCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    MKI_CHECK(ptrCon.maxCoreNum > 0 && ptrCon.maxCoreNum <= UINT_MAX, "maxCoreNum is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "maxCoreNum is invalid!"));
    ptrCon.maxUbSize = PlatformInfo::Instance().GetUbSize();
    ptrCon.maxEleFp16 = ptrCon.maxUbSize / 2; // buffer * 2
    int64_t tmpNumRow = 1;
    uint32_t dimsSize = launchParam.GetInTensor(0).desc.dims.size();
    MKI_CHECK(dimsSize > 0 && dimsSize <= 8, "dimsSize is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "dimsSize is invalid!"));
    for (size_t i = 0; i < dimsSize - 1; i++) {
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims[i] > 0 &&
                  tmpNumRow <= UINT_MAX / launchParam.GetInTensor(0).desc.dims[i],
                  "tmpNumRow is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "tmpNumRow is invalid!"));
        tmpNumRow *= launchParam.GetInTensor(0).desc.dims[i];
    }
    ptrCon.numRow = static_cast<uint32_t>(tmpNumRow);
    MKI_CHECK(launchParam.GetInTensor(0).desc.dims[dimsSize - 1] > 0 &&
                 launchParam.GetInTensor(0).desc.dims[dimsSize - 1] <= UINT_MAX, "numCol is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "numCol is invalid!"));
    ptrCon.numCol = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[dimsSize - 1]);
    ptrCon.numCore = Utils::CeilDiv(ptrCon.numRow, Utils::CeilDiv(ptrCon.numRow, ptrCon.maxCoreNum));
    kernelInfo.SetBlockDim(ptrCon.numCore);
    ptrCon.rowWork = Utils::CeilDiv(ptrCon.numRow, ptrCon.numCore);
    ptrCon.nlFirstdimPerCoreNum = ptrCon.rowWork;
    return Status::OkStatus();
}

template <typename T>
inline __attribute__((always_inline)) Status CheckSplit(T *tilingDataPtr, const int32_t &totalMemNeed,
                                                        const int32_t &sumData, const NormTilingDataPtrCon &ptrCon)
{
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    if (totalMemNeed > sumData) {
        MKI_CHECK(sumData > 0, "sumData is invalid!",
                     return Status::FailStatus(ERROR_INVALID_VALUE, "sumData is invalid!"));
        uint32_t timeCopyIn = static_cast<uint32_t>(Utils::CeilDiv(totalMemNeed, sumData));
        tilingDataPtr->firstDimPerTimes =
            (ptrCon.nlFirstdimPerCoreNum / timeCopyIn == 0) ? 1 : ptrCon.nlFirstdimPerCoreNum / timeCopyIn;
    } else {
        tilingDataPtr->firstDimPerTimes = ptrCon.nlFirstdimPerCoreNum;
    }
    MKI_CHECK(tilingDataPtr->firstDimPerTimes < UINT_MAX - tilingDataPtr->nlFirstdimPerCore,
              "row_work + row_step is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "row_work + row_step is invalid!"));
    return Status::OkStatus();
}

template <typename RmsNormQuantTilingDataType>
inline __attribute__((always_inline)) Status RmsNormQuantTilingGen(KernelInfo &kernelInfo,
                                                                   RmsNormQuantTilingDataType *tilingDataPtr,
                                                                   const LaunchParam &launchParam)
{
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    auto attrs = AnyCast<OpParam::Norm>(launchParam.GetParam());
    tilingDataPtr->epsilon = *reinterpret_cast<uint32_t *>(&attrs.epsilon);
    uint32_t maxCoreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    MKI_CHECK(maxCoreNum > 0, "maxCoreNum is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "maxCoreNum is invalid!"));
    int64_t numRow = 1;
    uint32_t dimsSize = launchParam.GetInTensor(0).desc.dims.size();
    MKI_CHECK(dimsSize > 0 && dimsSize <= 8, "dimsSize is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "dimsSize is invalid!"));
    for (size_t i = 0; i < dimsSize - 1; i++) {
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims[i] > 0 &&
                  numRow <= UINT_MAX / launchParam.GetInTensor(0).desc.dims[i], "numRow is invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "numRow is invalid!"));
        numRow *= launchParam.GetInTensor(0).desc.dims[i];
    }
    tilingDataPtr->numRow = static_cast<uint32_t>(numRow);
    MKI_CHECK(launchParam.GetInTensor(0).desc.dims[dimsSize - 1] > 0 &&
                 launchParam.GetInTensor(0).desc.dims[dimsSize - 1] <= UINT_MAX,
                 "numCol is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "numCol is invalid!"));
    uint32_t numCol = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[dimsSize - 1]);
    tilingDataPtr->numCol = numCol;
    uint32_t numCore = static_cast<uint32_t>(
        Utils::CeilDiv(static_cast<uint32_t>(numRow), Utils::CeilDiv(static_cast<uint32_t>(numRow), maxCoreNum)));
    tilingDataPtr->numCore = numCore;
    MKI_CHECK(tilingDataPtr->numCore <= UINT_MAX - tilingDataPtr->numRow, "numCore + numRow is invalid!",
            return Status::FailStatus(ERROR_INVALID_VALUE, "numCore + numRow is invalid!"));
    kernelInfo.SetBlockDim(numCore);
    float tempAve = float(1.0 / numCol);
    tilingDataPtr->avgFactor = *reinterpret_cast<uint32_t *>(&tempAve);
    tilingDataPtr->quantMin = -128; // default int8 min is -128
    const char *quantMinSetPtr = Mki::GetEnv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinSetPtr != nullptr && strcmp(quantMinSetPtr, "1") == 0) {
        tilingDataPtr->quantMin = -127; // set int8 min to -127
    }
    return Status::OkStatus();
}
} // namespace AsdOps

#endif
