/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cohere_layer_norm_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/norm.h"
#include "kernels/norm/common/common_tiling.h"
#include "kernels/norm/coherelayernorm/tiling/tiling_data.h"

namespace AsdOps {
static constexpr uint32_t LAYER_NORM_TILING_KEY_BASE = 2000000000;
static constexpr uint32_t LAYER_NORM_TILING_KEY_DTYPE = 100000000; // 0: fp16; 1: bf16
static constexpr uint32_t LAYER_NORM_TILING_KEY_FAST = 10000000;   // 0: fast; 1:slice
static constexpr uint32_t MISC_BUFFERS_SIZE = 32;

void CohereLayerNormPrintTilingInfo(const CohereLayerNormTilingData &tilingDataPtr)
{
    MKI_LOG(INFO) << "numCore = " << tilingDataPtr.numCore << ", "
                  << "numCoreRows = " << tilingDataPtr.numCoreRows << ", "
                  << "coreRowStrides = " << tilingDataPtr.coreRowStrides << ", "
                  << "coreRowRepeats = " << tilingDataPtr.coreRowRepeats << ", "
                  << "coreRowTailStrides = " << tilingDataPtr.coreRowTailStrides << ", "
                  << "coreRowTailRepeats = " << tilingDataPtr.coreRowTailRepeats << ", "
                  << "residualCoreRowStrides = " << tilingDataPtr.residualCoreRowStrides << ", "
                  << "residualCoreRowRepeats = " << tilingDataPtr.residualCoreRowRepeats << ", "
                  << "residualCoreRowTailStrides = " << tilingDataPtr.residualCoreRowTailStrides << ", "
                  << "residualCoreRowTailRepeats = " << tilingDataPtr.residualCoreRowTailRepeats << ", "
                  << "numColumns = " << tilingDataPtr.numColumns << ", "
                  << "columnStrides = " << tilingDataPtr.columnStrides << ", "
                  << "columnRepeats = " << tilingDataPtr.columnRepeats << ", "
                  << "residualColumnStrides = " << tilingDataPtr.residualColumnStrides << ", "
                  << "residualColumnRepeats = " << tilingDataPtr.residualColumnRepeats << ", "
                  << "numHeads = " << tilingDataPtr.numHeads << ", "
                  << "epsilon = " << tilingDataPtr.epsilon << ", "
                  << "averageFactor = " << tilingDataPtr.averageFactor;
}

Status MultipleRowMovedTiling(NormTilingDataPtrCon &layerNormPtrCon, CohereLayerNormTilingData &tilingDataPtr,
                              uint32_t singleRowMovedBufferSize, uint32_t multipleRowMovedBufferSize,
                              uint32_t miscBuffersSize)
{
    uint32_t numResidualCoreRows = layerNormPtrCon.numRow -
                                   tilingDataPtr.numCoreRows * (tilingDataPtr.numCore - 1);
    uint32_t calcCoreRowStrides = (layerNormPtrCon.maxUbSize - singleRowMovedBufferSize - miscBuffersSize) /
                                   multipleRowMovedBufferSize;
    tilingDataPtr.coreRowStrides = std::min(tilingDataPtr.numCoreRows, calcCoreRowStrides);
    MKI_CHECK(tilingDataPtr.coreRowStrides != 0, "coreRowStrides is equal to 0",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr.coreRowRepeats = tilingDataPtr.numCoreRows / tilingDataPtr.coreRowStrides;
    tilingDataPtr.coreRowTailStrides = tilingDataPtr.numCoreRows % tilingDataPtr.coreRowStrides;
    tilingDataPtr.coreRowTailRepeats = tilingDataPtr.coreRowTailStrides == 0 ? 0 : 1;
    tilingDataPtr.residualCoreRowStrides = std::min(numResidualCoreRows, calcCoreRowStrides);
    MKI_CHECK(tilingDataPtr.residualCoreRowStrides != 0, "residualCoreRowStrides is equal to 0",
                return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr.residualCoreRowRepeats = numResidualCoreRows /
                                            tilingDataPtr.residualCoreRowStrides;
    tilingDataPtr.residualCoreRowTailStrides = numResidualCoreRows %
                                                tilingDataPtr.residualCoreRowStrides;
    tilingDataPtr.residualCoreRowTailRepeats = tilingDataPtr.residualCoreRowTailStrides == 0 ? 0 : 1;
    tilingDataPtr.columnStrides = layerNormPtrCon.numCol;
    tilingDataPtr.columnRepeats = 1;
    tilingDataPtr.residualColumnStrides = 0;
    tilingDataPtr.residualColumnRepeats = 0;
    return Status::OkStatus();
}

Status SingleRowMovedTiling(NormTilingDataPtrCon &layerNormPtrCon, CohereLayerNormTilingData &tilingDataPtr,
                            uint32_t singleRowMovedElemSize, uint32_t multipleRowMovedElemSize,
                            uint32_t miscBuffersSize)
{
    uint32_t oneRepeatElemCount = 256U / sizeof(uint16_t);
    uint32_t calcColumnStrides = Utils::RoundDown((layerNormPtrCon.maxUbSize - miscBuffersSize) /
                                                  (singleRowMovedElemSize + multipleRowMovedElemSize),
                                                  oneRepeatElemCount);
    uint32_t numResidualCoreRows = layerNormPtrCon.numRow -
                                   tilingDataPtr.numCoreRows * (tilingDataPtr.numCore - 1);
    tilingDataPtr.columnStrides = std::min(tilingDataPtr.numColumns, calcColumnStrides);
    MKI_CHECK(tilingDataPtr.columnStrides != 0, "columnStrides is equal to 0",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr.columnRepeats = layerNormPtrCon.numCol / tilingDataPtr.columnStrides;
    tilingDataPtr.residualColumnStrides = layerNormPtrCon.numCol % tilingDataPtr.columnStrides;
    tilingDataPtr.residualColumnRepeats = tilingDataPtr.residualColumnStrides == 0 ? 0 : 1;
    tilingDataPtr.coreRowStrides = 1;
    tilingDataPtr.coreRowRepeats = tilingDataPtr.numCoreRows;
    tilingDataPtr.coreRowTailStrides = 0;
    tilingDataPtr.coreRowTailRepeats = 0;
    tilingDataPtr.residualCoreRowStrides = 1;
    tilingDataPtr.residualCoreRowRepeats = numResidualCoreRows;
    tilingDataPtr.residualCoreRowTailStrides = 0;
    tilingDataPtr.residualCoreRowTailRepeats = 0;
    return Status::OkStatus();
}

Status CohereLayerNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo, const
                            KernelBufferInfoCohereLayerNorm &kernelBufferInfo)
{
    // According to the operator design document,
    // the tiling strategy includes multiple rows moving and single row moving.
    //
    // The tiling strategy is adaptively determined based on the ratio of
    // the fixed Ub buffer size used for each calculation to the total Ub size.
    //
    // 1. Multiple rows moving: Move multiple rows of data to UB at once to reduce the number of DataCopy operations.
    // 2. Single row moving: Since the mean and variance need to be calculated for each token(row),
    //    if the data size of the row is too large, we can only move one row at a time,
    //    and the row needs to be moved partially.

    CohereLayerNormTilingData *tilingDataPtr =
        reinterpret_cast<CohereLayerNormTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    NormTilingDataPtrCon layerNormPtrCon;
    Status ret = LayerNormPtrFunc<CohereLayerNormTilingData>(layerNormPtrCon,
                                                             launchParam, kernelInfo);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    auto attrs = AnyCast<OpParam::Norm>(launchParam.GetParam());

    tilingDataPtr->numCore = layerNormPtrCon.numCore;
    tilingDataPtr->numCoreRows = layerNormPtrCon.nlFirstdimPerCoreNum;
    tilingDataPtr->numColumns = layerNormPtrCon.numCol;
    tilingDataPtr->numHeads = launchParam.GetInTensor(1).desc.dims[0];
    tilingDataPtr->epsilon = attrs.epsilon;
    tilingDataPtr->averageFactor = static_cast<float>(1.0 / layerNormPtrCon.numCol);
    uint32_t singleRowMovedElemSize = kernelBufferInfo.fp32BufNum * sizeof(uint32_t) +
                                      kernelBufferInfo.fp16BufNum * sizeof(uint16_t);
    uint32_t multipleRowMovedElemSize = kernelBufferInfo.fp16BufNumForMulRow * sizeof(uint16_t);
    // overflow check for calculating fixedUsedBufferSize
    MKI_CHECK(tilingDataPtr->numColumns <=
              (UINT_MAX / (singleRowMovedElemSize + multipleRowMovedElemSize) - MISC_BUFFERS_SIZE),
              "numColumns is too large",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t singleRowMovedBufferSize = singleRowMovedElemSize * layerNormPtrCon.numCol;
    uint32_t multipleRowMovedBufferSize = multipleRowMovedElemSize * layerNormPtrCon.numCol;
    uint32_t fixedUsedBufferSize = singleRowMovedBufferSize + multipleRowMovedBufferSize + MISC_BUFFERS_SIZE;
    uint64_t tilingKey = LAYER_NORM_TILING_KEY_BASE;

    if (fixedUsedBufferSize < layerNormPtrCon.maxUbSize) {  // multiple rows moved simultaneously
        MultipleRowMovedTiling(layerNormPtrCon, *tilingDataPtr,
                               singleRowMovedBufferSize, multipleRowMovedBufferSize, MISC_BUFFERS_SIZE);
    } else {  // single row moved
        SingleRowMovedTiling(layerNormPtrCon, *tilingDataPtr,
                             singleRowMovedElemSize, multipleRowMovedElemSize, MISC_BUFFERS_SIZE);
        tilingKey += LAYER_NORM_TILING_KEY_FAST;
    }
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? LAYER_NORM_TILING_KEY_DTYPE : 0;
    kernelInfo.SetTilingId(tilingKey); // 2000000000 + 100000000 + 10000000

    CohereLayerNormPrintTilingInfo(*tilingDataPtr);

    return Status::OkStatus();
}
} // namespace AsdOps