/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "post_layer_norm_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/norm.h"
#include "tiling_data.h"
#include "kernels/norm/common/common_tiling.h"

namespace AsdOps {
void PostLayerNormPrintLog(const PostLayerNormTilingData &tilingData)
{
    MKI_LOG(INFO) << "numCore = " << tilingData.numCore
                  << ", aveStr = " << tilingData.aveStr
                  << ", nlFirstdimPerCore = " << tilingData.nlFirstdimPerCore
                  << ", numFirstDim = " << tilingData.numFirstDim
                  << ", epsStr = " << tilingData.epsStr
                  << ", numLastDim = " << tilingData.numLastDim
                  << ", lFirstdimPerCore = " << tilingData.lFirstdimPerCore
                  << ", normMode = " << tilingData.normMode
                  << ", zoomScale = " << tilingData.zoomScale
                  << ", firstDimPerTimes= " << tilingData.firstDimPerTimes
                  << ", sliceNum = " << tilingData.sliceNum
                  << ", sliceSize = " << tilingData.sliceSize
                  << ", tailSliceSize = " << tilingData.tailSliceSize;
}

Status PostLayerNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const KernelBufferInfo &kernelBufferInfo)
{
    PostLayerNormTilingData *tilingDataPtr =
        reinterpret_cast<PostLayerNormTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));

    auto attrs = AnyCast<OpParam::Norm>(launchParam.GetParam());
    tilingDataPtr->normMode = attrs.opsMode;
    tilingDataPtr->zoomScale = attrs.zoomScaleValue;
    tilingDataPtr->epsStr = attrs.epsilon;
    NormTilingDataPtrCon postLayerNormPtrCon;
    Status ret = PostLayerNormPtrFunc<PostLayerNormTilingData>(tilingDataPtr, postLayerNormPtrCon, launchParam,
                                                               kernelInfo);
    OP_TILING_CHECK_STATUS_RETURN(ret);

    uint32_t maxUbSize = static_cast<uint32_t>(postLayerNormPtrCon.maxUbSize);
    uint32_t numCol = static_cast<uint32_t>(postLayerNormPtrCon.numCol);
    tilingDataPtr->aveStr =  float(1.0 / numCol);

    uint32_t singleRowSizePerElem = kernelBufferInfo.fp32BufNum * sizeof(uint32_t) +
                                    kernelBufferInfo.fp16BufNum * sizeof(uint16_t);
    uint32_t multiRowSizePerElem = kernelBufferInfo.fp16BufNumForMulRow * sizeof(uint16_t) +
                                   kernelBufferInfo.i8BufNumForMulRow * sizeof(uint8_t);
    uint64_t singleRowBufferSize = singleRowSizePerElem * numCol;
    uint64_t multiRowBufferSize = multiRowSizePerElem * numCol;
    uint64_t bf16TilingKey = launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? 2 : 0;    // 二进制：10 or 00
    if (maxUbSize < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / sizeof(uint16_t);
        uint32_t elemSize = Utils::RoundDown(maxUbSize / (singleRowSizePerElem + multiRowSizePerElem),
                                             oneRepeatElemCount);
        tilingDataPtr->sliceNum = Utils::CeilDiv(numCol, elemSize);
        tilingDataPtr->sliceSize = elemSize;
        tilingDataPtr->tailSliceSize = numCol - (tilingDataPtr->sliceNum - 1) * elemSize;
        tilingDataPtr->firstDimPerTimes = 1;
        kernelInfo.SetTilingId(bf16TilingKey + static_cast<uint64_t>(1));
    } else {
        tilingDataPtr->firstDimPerTimes = (maxUbSize - singleRowBufferSize) / multiRowBufferSize;
        tilingDataPtr->sliceNum = 1;
        tilingDataPtr->sliceSize = numCol;
        tilingDataPtr->tailSliceSize = numCol;
        kernelInfo.SetTilingId(bf16TilingKey + static_cast<uint64_t>(0));
    }
    PostLayerNormPrintLog(*tilingDataPtr);
    return Status::OkStatus();
}
} // namespace AsdOps
