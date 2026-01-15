/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "adalayer_norm_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/norm.h"
#include "tiling_data.h"
#include "kernels/norm/common/common_tiling.h"

namespace AsdOps {
constexpr uint32_t LAYER_NORM_TILING_KEY_BASE = 2000000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16
constexpr uint32_t LAYER_NORM_TILING_KEY_FAST = 10000000;   // 0: fast; 1:slice
constexpr uint32_t SCALE_DIM_THREE = 3;

void AdaLayerNormPrintLog(AdaLayerNormTilingData &tilingData)
{
    MKI_LOG(INFO) << " tilingDataPtr->numCore = " << tilingData.numCore
                  << " tilingDataPtr->numFirstDim = " << tilingData.numFirstDim
                  << " tilingDataPtr->numRowDim = " << tilingData.numRowDim
                  << " tilingDataPtr->numLastDim = " << tilingData.numLastDim
                  << " tilingDataPtr->gammaFirstDim = " << tilingData.gammaFirstDim
                  << " tilingDataPtr->gammaRowDim = " << tilingData.gammaRowDim
                  << " tilingDataPtr->epsStr = " << tilingData.epsStr
                  << " tilingDataPtr->aveStr = " << tilingData.aveStr
                  << " tilingDataPtr->coreLoop = " << tilingData.coreLoop
                  << " tilingDataPtr->rowLoop = " << tilingData.rowLoop
                  << " tilingDataPtr->rowSize = " << tilingData.rowSize
                  << " tilingDataPtr->tailRowSize = " << tilingData.tailRowSize
                  << " tilingDataPtr->sliceNum = " << tilingData.sliceNum
                  << " tilingDataPtr->sliceSize = " << tilingData.sliceSize
                  << " tilingDataPtr->tailSliceSize = " << tilingData.tailSliceSize;
}

Status GetTilingSliceInfo(AdaLayerNormTilingData &tilingData, uint32_t maxUbSize, uint32_t numCol,
                          KernelBufferInfoAdaLayerNorm kernelBufferInfo)
{
    uint32_t singleRowSizePerElem =
        kernelBufferInfo.fp32BufNum * sizeof(uint32_t) + kernelBufferInfo.fp16BufNum * sizeof(uint16_t);
    uint32_t singleRowBufferSize = singleRowSizePerElem * numCol;
    MKI_CHECK(numCol <= (UINT_MAX / singleRowSizePerElem), "singleRowBufferSize is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "singleRowBufferSize is invalid!"));
    uint32_t multiRowSizePerElem = kernelBufferInfo.fp16BufNumForMulRow * sizeof(uint16_t);
    uint32_t multiRowBufferSize = multiRowSizePerElem * numCol;
    MKI_CHECK(numCol <= (UINT_MAX / multiRowSizePerElem), "multiRowBufferSize is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "multiRowBufferSize is invalid!"));

    if (maxUbSize < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / sizeof(uint16_t);
        uint32_t elemSize = Utils::RoundDown(
            maxUbSize / (singleRowSizePerElem + multiRowSizePerElem), oneRepeatElemCount);
        tilingData.sliceNum = Utils::CeilDiv(numCol, elemSize);
        tilingData.sliceSize = elemSize;
        tilingData.tailSliceSize = numCol - (tilingData.sliceNum - 1) * elemSize;
        tilingData.rowLoop = tilingData.numRowDim;
        tilingData.rowSize = 1;
        tilingData.tailRowSize = 1;
    } else {
        tilingData.sliceNum = 1;
        tilingData.sliceSize = numCol;
        tilingData.tailSliceSize = numCol;
        tilingData.rowSize = std::min((maxUbSize - singleRowBufferSize) / multiRowBufferSize, tilingData.numRowDim);
        tilingData.rowLoop = Utils::CeilDiv(tilingData.numRowDim, tilingData.rowSize);
        tilingData.tailRowSize = tilingData.numRowDim - (tilingData.rowLoop - 1) * tilingData.rowSize;
    }

    return Status::OkStatus();
}

void GetTilingBasicInfo(const LaunchParam &launchParam, AdaLayerNormTilingData &tilingData,
                        uint32_t numCol, uint32_t dimsSize)
{
    tilingData.aveStr = float(1.0 / numCol);
    tilingData.numLastDim = numCol;
    uint32_t numRow = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[dimsSize - 2]);
    tilingData.numRowDim = numRow;
    tilingData.numFirstDim = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[0]);
    tilingData.gammaFirstDim = static_cast<uint32_t>(launchParam.GetInTensor(1).desc.dims[0]);
    tilingData.gammaRowDim = launchParam.GetInTensor(1).desc.dims.size() == SCALE_DIM_THREE ?
                            static_cast<uint32_t>(launchParam.GetInTensor(1).desc.dims[1]) : 1;
}

Status AdaLayerNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo,
                          KernelBufferInfoAdaLayerNorm &kernelBufferInfo)
{
    AdaLayerNormTilingData *tilingDataPtr =
        reinterpret_cast<AdaLayerNormTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    auto attrs = AnyCast<OpParam::Norm>(launchParam.GetParam());
    tilingDataPtr->epsStr = attrs.epsilon;

    uint32_t maxCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    MKI_CHECK(maxCoreNum > 0 && maxCoreNum <= UINT_MAX, "maxCoreNum is invalid!",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "maxCoreNum is invalid!"));
    uint32_t maxUbSize = PlatformInfo::Instance().GetUbSize(); // maxUb
    uint32_t dimsSize = launchParam.GetInTensor(0).desc.dims.size();
    uint32_t numCol = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[dimsSize - 1]);

    GetTilingBasicInfo(launchParam, *tilingDataPtr, numCol, dimsSize);

    GetTilingSliceInfo(*tilingDataPtr, maxUbSize, numCol, kernelBufferInfo);

    tilingDataPtr->coreLoop = tilingDataPtr->numFirstDim * tilingDataPtr->rowLoop;
    tilingDataPtr->numCore = std::min(tilingDataPtr->coreLoop, maxCoreNum);
    uint64_t tilingKey = LAYER_NORM_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? LAYER_NORM_TILING_KEY_DTYPE : 0;
    tilingKey += tilingDataPtr->sliceNum == 1 ? LAYER_NORM_TILING_KEY_FAST : 0;
    kernelInfo.SetTilingId(tilingKey); // 2000000000 + 100000000 + 10000000
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);
    AdaLayerNormPrintLog(*tilingDataPtr);

    return Status::OkStatus();
}
} // namespace AsdOps
