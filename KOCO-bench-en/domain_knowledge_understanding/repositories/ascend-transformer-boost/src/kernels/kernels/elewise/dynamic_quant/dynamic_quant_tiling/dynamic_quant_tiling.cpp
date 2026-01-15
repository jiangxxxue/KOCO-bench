/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dynamic_quant_tiling.h"
#include <cstring>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/elewise.h"
#include "kernels/norm/common/common_tiling.h"
#include "tiling_data.h"
#include "dynamic_quant_util.h"

namespace AsdOps {
using namespace Mki;

constexpr uint32_t DYNAMIC_QUANT_TILING_KEY_BASE = 10000;
constexpr uint32_t DYNAMIC_QUANT_TILING_KEY_UNALIGN_MODE = 1000;    // 0: 910b unalign, 1: 310p unalign
constexpr uint32_t DYNAMIC_QUANT_TILING_KEY_ALIGN = 100;            // 0:DynamicQuantUnAlign; 1:DynamicQuantAlign
constexpr uint32_t DYNAMIC_QUANT_TILING_KEY_DTYPE = 10;             // 0:fp16; 1:bf16
uint64_t ComputeTilingKey(uint32_t alignType, const LaunchParam &launchParam)
{
    uint64_t tilingKey = DYNAMIC_QUANT_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? DYNAMIC_QUANT_TILING_KEY_DTYPE : 0;
    if (alignType == 0) {
        tilingKey += DYNAMIC_QUANT_TILING_KEY_ALIGN;
    } else if (alignType == 1) {
        tilingKey += DYNAMIC_QUANT_TILING_KEY_UNALIGN_MODE;
    }
    return tilingKey;
}

Status ParseShape(const LaunchParam &launchParam, DynamicQuantTilingData &tilingDataPtr,
    uint64_t &rowNumTotal)
{
    const Mki::SVector<int64_t> &shape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(!shape.empty(), "shape should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "shape should not be empty"));

    size_t dims = shape.size();
    for (size_t i = 0; i < dims; ++i) {
        if (i < dims - 1) {
            MKI_CHECK(shape[i] > 0 && rowNumTotal < static_cast<uint64_t>(UINT32_MAX / shape[i]),
                "rowNumTotal or shape is invalid!",
                return Status::FailStatus(ERROR_INVALID_VALUE, "rowNumTotal or shape is invalid!"));
            rowNumTotal *= shape[i];
        } else {
            tilingDataPtr.sizeH = shape[i];
        }
    }
    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
        MKI_CHECK(tilingDataPtr.sizeH <= DYNAMIC_QUANT_BF16_LAST_DIM_LIMITATION,
            "Ascend910B BF16 input last dim is bigger than limitation!",
            return Status::FailStatus(ERROR_INVALID_VALUE,
            "Ascend910B BF16 input last dim is bigger than limitation!"));
    }
    if (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P) {
        MKI_CHECK(tilingDataPtr.sizeH <= DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P,
            "Ascend310P F16 input last dim is bigger than limitation!",
            return Status::FailStatus(ERROR_INVALID_VALUE,
            "Ascend310P F16 input last dim is bigger than limitation!"));
    }
    return Status::OkStatus();
}

/**
 * SetSuitNumCopyRow: get the most appropriate theoretical value for numCopyRow in one task
 * 1. numCopyRow = (ubSize - sizeX*2 - HEADSPACE) / (sizeX*5); 5:in*2*2(2pipline) + out*1; 2:tmp*2
 * 2. rowSuit = Utils::RoundUp(213 - sizeX * 2 / 7, 8);
 *    This formula is summarized after testing. from 512->64 and 64->192.
 * 3. sizeH > 512 -> numCopyRow: 64 ; 8 <= sizeH < 64 -> numCopyRow: 192
 *    64 <= sizeH <= 512 -> numCopyRow: Utils::RoundUp(213 - sizeX * 2 / 7, 8)
 */
void SetSuitNumCopyRow(DynamicQuantTilingData &tilingDataPtr)
{
    tilingDataPtr.sizeX = Utils::RoundUp(tilingDataPtr.sizeH, DYNAMIC_QUANT_ALIGN_NUM_X);
    tilingDataPtr.sizeZOut = Utils::RoundUp(tilingDataPtr.sizeH);

    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    tilingDataPtr.numCopyRow = (ubSize - tilingDataPtr.sizeX * DYNAMIC_QUANT_FP16_BUF_SCALE - \
        DYNAMIC_QUANT_HEADSPACE) / (tilingDataPtr.sizeX * DYNAMIC_QUANT_COPY_ROW_SCALE);
    MKI_LOG(INFO) << "numCopyRow = " << tilingDataPtr.numCopyRow;
    uint32_t rowSuit = DYNAMIC_QUANT_ROW_SUIT_ADD - tilingDataPtr.sizeX * \
        DYNAMIC_QUANT_ROW_SUIT_MUL / DYNAMIC_QUANT_ROW_SUIT_DIV;
    rowSuit = rowSuit - rowSuit % DYNAMIC_QUANT_ALIGN_NUM_SCALE;
    if (tilingDataPtr.numCopyRow > DYNAMIC_QUANT_COPY_ROW_LONG &&
        tilingDataPtr.sizeX >= DYNAMIC_QUANT_LEN_H_LONG) {
        tilingDataPtr.numCopyRow = DYNAMIC_QUANT_COPY_ROW_LONG;
    } else if (tilingDataPtr.numCopyRow > rowSuit && rowSuit > DYNAMIC_QUANT_ALIGN_NUM_SCALE &&
        tilingDataPtr.sizeX >= DYNAMIC_QUANT_LEN_H_SHORT) {
        tilingDataPtr.numCopyRow = rowSuit;
    } else if (tilingDataPtr.numCopyRow > DYNAMIC_QUANT_COPY_ROW_SHORT &&
        tilingDataPtr.sizeX < DYNAMIC_QUANT_LEN_H_SHORT &&
        tilingDataPtr.sizeX > DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        tilingDataPtr.numCopyRow = DYNAMIC_QUANT_COPY_ROW_SHORT;
    } else if (tilingDataPtr.numCopyRow > DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        tilingDataPtr.numCopyRow = tilingDataPtr.numCopyRow - tilingDataPtr.numCopyRow % \
            DYNAMIC_QUANT_ALIGN_NUM_SCALE;
    }
    MKI_LOG(INFO) << "numCopyRow = " << tilingDataPtr.numCopyRow;
}

/**
 * CorrectNumCopyRow: use real value to Correction numCopyRow
 * Ascend310P: numCopyRow >= 8, perRowNum <= 8 -> numCopyRow = 8
 *             numCopyRow > alignRowNum -> numCopyRow = alignRowNum
 * Ascend910B: perRowNum <= 0, numCopyRow > 0 -> numCopyRow = 1
 *             numCopyRow > alignRowNum, perRowNum > alignRowNum -> numCopyRow = alignRowNum
 *             numCopyRow > alignRowNum, perRowNum < alignRowNum -> numCopyRow = perRowNum
 *             numCopyRow > perRowNum, perRowNum < 8 -> numCopyRow = perRowNum
 */
Status CorrectNumCopyRow(DynamicQuantTilingData &tilingDataPtr, uint64_t rowNumTotal)
{
    uint32_t perRowNum = Utils::CeilDiv(static_cast<uint32_t>(rowNumTotal), tilingDataPtr.numCore);
    uint32_t alignRowNum = Utils::RoundUp(perRowNum, DYNAMIC_QUANT_ALIGN_NUM_SCALE);
    MKI_LOG(INFO) << "perRowNum = " << perRowNum;
    if (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_310P) {
        tilingDataPtr.alignType = DYNAMIC_QUANT_STATUS_UNALIGN_310P;
        if (tilingDataPtr.numCopyRow >= DYNAMIC_QUANT_ALIGN_NUM_SCALE &&
            perRowNum <= DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
            tilingDataPtr.numCopyRow = DYNAMIC_QUANT_ALIGN_NUM_SCALE;
        } else if (tilingDataPtr.numCopyRow >= alignRowNum) {
            tilingDataPtr.numCopyRow = alignRowNum;
        }
        if (tilingDataPtr.sizeH % DYNAMIC_QUANT_ALIGN_SIZE != 0 ||
            tilingDataPtr.numCopyRow < DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Ascend310P input last dim must 64Byte alignment");
        }
    } else {
        tilingDataPtr.alignType = DYNAMIC_QUANT_STATUS_UNALIGN_910B;
        if (perRowNum <= 0 && tilingDataPtr.numCopyRow > 0) {
            tilingDataPtr.numCopyRow = 1;
        } else if (tilingDataPtr.numCopyRow > alignRowNum && perRowNum > alignRowNum) {
            tilingDataPtr.numCopyRow = alignRowNum;
        } else if (tilingDataPtr.numCopyRow > alignRowNum && perRowNum < alignRowNum) {
            tilingDataPtr.numCopyRow = perRowNum;
        } else if (tilingDataPtr.numCopyRow > perRowNum && perRowNum < DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
            tilingDataPtr.numCopyRow = perRowNum;
        }
        if (tilingDataPtr.numCopyRow == 0) {
            return Status::FailStatus(ERROR_INVALID_VALUE, "Ascend910B input last dim is bigger than limitation");
        }
    }
    MKI_LOG(INFO) << "numCopyRow = " << tilingDataPtr.numCopyRow;
    tilingDataPtr.sizeCopyRow = Utils::RoundUp(tilingDataPtr.numCopyRow, DYNAMIC_QUANT_ALIGN_NUM_SCALE);
    return Status::OkStatus();
}

Status SetTilingData(DynamicQuantTilingData &tilingDataPtr, uint64_t rowNumTotal)
{
    SetSuitNumCopyRow(tilingDataPtr);

    Status status = CorrectNumCopyRow(tilingDataPtr, rowNumTotal);
    if (!status.Ok()) {
        return status;
    }

    uint32_t patchTotal = rowNumTotal / tilingDataPtr.numCopyRow;
    tilingDataPtr.numLastTailRow = rowNumTotal % tilingDataPtr.numCopyRow;
    tilingDataPtr.numTailTimes = patchTotal / tilingDataPtr.numCore;
    tilingDataPtr.numHeadCore = patchTotal % tilingDataPtr.numCore;
    tilingDataPtr.numTailCore = tilingDataPtr.numCore - tilingDataPtr.numHeadCore;
    tilingDataPtr.numHeadTimes = tilingDataPtr.numTailTimes + 1;

    if (tilingDataPtr.numLastTailRow == 0 &&
        tilingDataPtr.numCopyRow % DYNAMIC_QUANT_ALIGN_NUM_SCALE == 0 &&
        tilingDataPtr.sizeH % DYNAMIC_QUANT_ALIGN_SIZE == 0) {
        tilingDataPtr.alignType = DYNAMIC_QUANT_STATUS_ALIGN;
    }
    return Status::OkStatus();
}

Status DynamicQuantTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    DynamicQuantTilingData *tilingDataPtr =
        reinterpret_cast<DynamicQuantTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE, "tilingDataPtr should not be empty"));
    tilingDataPtr->numCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);

    const Mki::SVector<int64_t> &shape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(!shape.empty(), "shape should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "shape should not be empty"));

    auto attrs = AnyCast<OpParam::Elewise>(launchParam.GetParam());
    tilingDataPtr->asymmetric = *reinterpret_cast<uint32_t *>(&attrs.asymmetric);

    uint64_t rowNumTotal = 1;
    Status res = ParseShape(launchParam, *tilingDataPtr, rowNumTotal);
    OP_TILING_CHECK_STATUS_RETURN(res);

    Status ret = SetTilingData(*tilingDataPtr, rowNumTotal);
    OP_TILING_CHECK_STATUS_RETURN(ret);

    MKI_LOG(INFO) << "numCore = "           << tilingDataPtr->numCore
                  << ", sizeH = "           << tilingDataPtr->sizeH
                  << ", sizeX = "           << tilingDataPtr->sizeX
                  << ", sizeZOut = "        << tilingDataPtr->sizeZOut
                  << ", sizeCopyRow = "     << tilingDataPtr->sizeCopyRow
                  << ", numCopyRow = "      << tilingDataPtr->numCopyRow
                  << ", numHeadCore = "     << tilingDataPtr->numHeadCore
                  << ", numTailCore = "     << tilingDataPtr->numTailCore
                  << ", numHeadTimes = "    << tilingDataPtr->numHeadTimes
                  << ", numTailTimes = "    << tilingDataPtr->numTailTimes
                  << ", numLastTailRow = "  << tilingDataPtr->numLastTailRow
                  << ", alignType = "        << tilingDataPtr->alignType
                  << ", asymmetric = "      << *reinterpret_cast<bool *>(&tilingDataPtr->asymmetric);

    // 按数据类型区分TilingKey
    uint64_t tilingKey = ComputeTilingKey(tilingDataPtr->alignType, launchParam);
    MKI_LOG(INFO) << "Dynamic Quant tilingKey is : " << tilingKey;
    kernelInfo.SetTilingId(tilingKey);
    return Status::OkStatus();
}
} // namespace AsdOps
