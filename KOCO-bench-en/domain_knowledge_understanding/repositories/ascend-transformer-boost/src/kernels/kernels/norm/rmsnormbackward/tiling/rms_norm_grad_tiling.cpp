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
#include "rms_norm_grad_tiling.h"

namespace AsdOps {
static const uint32_t FP32_ALIGN_32 = 8;
static const uint32_t FP16_BF16_ALIGN_32 = 16;
static const uint64_t DTYPE_FP32 = 1;
static const uint64_t DTYPE_FP16 = 2;
static const uint64_t DTYPE_BF16 = 3;
static const uint64_t MULTI_CORE = 2;
static const uint64_t UB_SPLIT_N = 1;
static const uint64_t UB_SPLIT_D = 2;
static const uint32_t BF16_SPLIT_N_BUFFER_SIZE = 5760;
static const uint32_t BF16_SPLIT_D_BUFFER_SIZE = 4096;
static const uint32_t DY_INPUT = 0;
static const uint32_t X_INPUT = 1;
static const uint32_t RSTD_INPUT = 2;
static const uint32_t GAMMA_INPUT = 3;
static const uint32_t BLOCK_SIZE = 32;
static const uint32_t WORK_SPACE_INDEX = 6;
static const uint32_t DTYPE_WEIGHT = 100;
static const uint32_t BLOCK_WEIGHT = 10;
static const int32_t NEG_ONE = -1;
static const int32_t TWICE_WORKSPACE = 2;

static bool LargeNSmallD(KernelInfo &kernelInfo, RmsNormGradTilingData &tilingDataPointer, uint32_t bufferSize,
                         uint32_t rowVal, uint32_t colVal, uint32_t coreNum)
{
    if (coreNum == 0 || colVal == 0) {
        return false;
    }
    // block split
    uint32_t blockFactor = (rowVal + coreNum - 1) / coreNum;
    MKI_CHECK(rowVal <= UINT_MAX - coreNum, "rowVal + coreNum is invalid!", return false);
    uint32_t blockDim = (rowVal + blockFactor - 1) / blockFactor;
    MKI_CHECK(rowVal <= UINT_MAX - blockFactor, "rowVal + blockFactor is invalid!", return false);
    uint32_t coreCalcNum = blockFactor;
    uint32_t coreCalcTail = rowVal % blockFactor != 0 ? rowVal - (blockDim - 1) * blockFactor : 0;
    tilingDataPointer.blockFactor = blockFactor;
    tilingDataPointer.coreCalcNum = coreCalcNum;
    tilingDataPointer.coreCalcTail = coreCalcTail;
    tilingDataPointer.blockDim = blockDim;
    kernelInfo.SetBlockDim(blockDim);

    // ub split, assume colVal <= bufferSize
    uint32_t ubFactor = bufferSize / colVal < blockFactor ? bufferSize / colVal : blockFactor;
    uint32_t ubLoop = (blockFactor + ubFactor - 1) / ubFactor;
    MKI_CHECK(colVal <= UINT_MAX - ubFactor, "colVal + ubFactor is invalid!", return false);
    uint32_t ubCalcNum = ubFactor;
    uint32_t ubCalcTail = blockFactor % ubFactor != 0 ? blockFactor - (ubLoop - 1) * ubFactor : 0;
    tilingDataPointer.ubSplitDim = 0;
    tilingDataPointer.ubCalcNum = ubCalcNum;
    tilingDataPointer.ubFactor = ubFactor;
    tilingDataPointer.ubCalcTail = ubCalcTail;
    tilingDataPointer.ubCalcLoop = ubLoop;

    // calc ub factor for tail core
    if (coreCalcTail == 0) {
        tilingDataPointer.ubCalcTailNum = 0;
        tilingDataPointer.ubCalcTailTail = 0;
        tilingDataPointer.ubCalcTailLoop = 0;
    } else {
        uint32_t ubTailFactor = ubFactor <= coreCalcTail ? ubFactor : coreCalcTail;
        uint32_t ubTailLoop = (coreCalcTail + ubTailFactor - 1) / ubTailFactor;
        uint32_t ubTailNum = ubTailFactor;
        uint32_t ubTailTail =
                coreCalcTail % ubTailFactor != 0 ? coreCalcTail - (ubTailLoop - 1) * ubTailFactor : 0;
        tilingDataPointer.ubCalcTailNum = ubTailNum;
        tilingDataPointer.ubCalcTailTail = ubTailTail;
        tilingDataPointer.ubCalcTailLoop = ubTailLoop;
    }
    return true;
}

static bool LargeNLargeD(KernelInfo &kernelInfo, RmsNormGradTilingData &tilingDataPointer, uint32_t bufferSize,
                         uint32_t rowVal, uint32_t colVal, uint32_t coreNum)
{
    // block split
    if (coreNum == 0) {
        return false;
    }
    uint32_t blockFactor = (rowVal + coreNum - 1) / coreNum;
    MKI_CHECK(rowVal <= UINT_MAX - coreNum, "rowVal + coreNum is invalid!", return false);
    uint32_t blockDim = (rowVal + blockFactor - 1) / blockFactor;
    MKI_CHECK(rowVal <= UINT_MAX - blockFactor, "rowVal + blockFactor is invalid!", return false);
    uint32_t coreCalcNum = blockFactor;
    uint32_t coreCalcTail = rowVal % blockFactor != 0 ? rowVal - (blockDim - 1) * blockFactor : 0;
    tilingDataPointer.blockFactor = blockFactor;
    tilingDataPointer.coreCalcNum = coreCalcNum;
    tilingDataPointer.coreCalcTail = coreCalcTail;
    tilingDataPointer.blockDim = blockDim;
    kernelInfo.SetBlockDim(blockDim);
    // ub split
    uint32_t ubFactor = bufferSize;
    uint32_t ubLoop = (colVal + ubFactor - 1) / ubFactor;
    MKI_CHECK(colVal <= UINT_MAX - ubFactor, "colVal + ubFactor is invalid!", return false);
    uint32_t ubCalcNum = ubFactor;
    uint32_t ubCalcTail = colVal % ubFactor != 0 ? colVal - (ubLoop - 1) * ubFactor : 0;
    tilingDataPointer.ubSplitDim = 1;
    tilingDataPointer.ubFactor = ubFactor;
    tilingDataPointer.ubCalcNum = ubCalcNum;
    tilingDataPointer.ubCalcTail = ubCalcTail;
    tilingDataPointer.ubCalcLoop = ubLoop;
    if (coreCalcTail == 0) {
        tilingDataPointer.ubCalcTailNum = 0;
        tilingDataPointer.ubCalcTailTail = 0;
        tilingDataPointer.ubCalcTailLoop = 0;
    } else {
        tilingDataPointer.ubCalcTailNum = ubCalcNum;
        tilingDataPointer.ubCalcTailTail = ubCalcTail;
        tilingDataPointer.ubCalcTailLoop = ubLoop;
    }
    return true;
}

bool GetColVal(const size_t gammaDimNum, const Mki::SVector<int64_t>& gamma, int64_t &colVal)
{
    for (size_t i = 0; i < gammaDimNum; i++) {
        MKI_CHECK(gamma[i] > 0 && colVal <= UINT_MAX / gamma[i], "colVal is invalid!", return false);
        colVal *= gamma[i];
    }
    return true;
}

bool GetRowVal(const LaunchParam &launchParam, const size_t gammaDimNum,
               const Mki::SVector<int64_t>& dyShape, int64_t &rowVal)
{
    for (size_t i = 0; i < launchParam.GetInTensor(DY_INPUT).desc.dims.size() - gammaDimNum; i++) {
        MKI_CHECK(dyShape[i] > 0 && rowVal <= UINT_MAX / dyShape[i], "rowValTmp is invalid!", return false);
        rowVal *= dyShape[i];
    }
    return true;
}

bool GetRstdShape(const LaunchParam &launchParam, const Mki::SVector<int64_t>& rstd, int64_t &rstdShapeTmp)
{
    for (size_t i = 0; i < launchParam.GetInTensor(RSTD_INPUT).desc.dims.size(); i++) {
        MKI_CHECK(rstd[i] > 0 && rstdShapeTmp <= UINT_MAX / rstd[i], "rstdShapeTmp is invalid!", return false);
        rstdShapeTmp *= rstd[i];
    }
    return true;
}

uint64_t GetDTypeKey(const TensorDType dataType, uint32_t &bufferSize)
{
    uint64_t dtypeKey = DTYPE_FP32;
    if (dataType == TENSOR_DTYPE_FLOAT16) {
        dtypeKey = DTYPE_FP16;
    }
    if (dataType == TENSOR_DTYPE_BF16) {
        dtypeKey = DTYPE_BF16;
        bufferSize = BF16_SPLIT_N_BUFFER_SIZE;
    }
    return dtypeKey;
}

Status RmsNormGradTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto *tilingDataPointer =
        reinterpret_cast<RmsNormGradTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPointer != nullptr, "tilingdata should not be empty",
              return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata should not be empty"));
    Mki::SVector<int64_t> dyShape = launchParam.GetInTensor(DY_INPUT).desc.dims;
    Mki::SVector<int64_t> rstd = launchParam.GetInTensor(RSTD_INPUT).desc.dims;
    Mki::SVector<int64_t> gamma = launchParam.GetInTensor(GAMMA_INPUT).desc.dims;
    size_t gammaDimNum = launchParam.GetInTensor(GAMMA_INPUT).desc.dims.size();
    Mki::SVector<int64_t> xShape = launchParam.GetInTensor(X_INPUT).desc.dims;
    Mki::SVector<int64_t> gammaShape = launchParam.GetInTensor(GAMMA_INPUT).desc.dims;
    int32_t startDim = launchParam.GetInTensor(X_INPUT).desc.dims.size() - gammaDimNum;
    MKI_CHECK(startDim > NEG_ONE, "x's dim - gamma's dim should not less than 0!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "x's dim - gamma's dim should not less than 0!"));
    for (size_t i = 0; i < gammaDimNum; i++) {
        MKI_CHECK(xShape[startDim + i] == gammaShape[i],
                  "input gamma's shape invalid!",
                  return Status::FailStatus(ERROR_INVALID_VALUE, "input gamma's shape invalid!"));
    }
    int64_t colVal = 1;
    bool blColVal = GetColVal(gammaDimNum, gamma, colVal);
    MKI_CHECK(blColVal, "blColVal failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
    float avgVal = 1.0f / colVal;
    int64_t rowVal = 1;
    bool blRowVal = GetRowVal(launchParam, gammaDimNum, dyShape, rowVal);
    MKI_CHECK(blRowVal, "blRowVal failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
    int64_t rstdShapeTmp = 1;
    bool blRstdShape = GetRstdShape(launchParam, rstd, rstdShapeTmp);
    MKI_CHECK(blRstdShape, "blRstdShape failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK(static_cast<uint32_t>(rstdShapeTmp) == static_cast<uint32_t>(rowVal),
              "rstdShapeTmp shape invalid", return Status::FailStatus(ERROR_INVALID_VALUE, "rstdShapeTmp invalid"));
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    TensorDType dataType = launchParam.GetInTensor(DY_INPUT).desc.dtype;
    uint32_t bufferSize = 6400;
    uint64_t dtypeKey = GetDTypeKey(dataType, bufferSize);
    uint32_t alignFactor = dtypeKey == 1 ? FP32_ALIGN_32 : FP16_BF16_ALIGN_32;
    MKI_CHECK(colVal <= UINT_MAX - alignFactor + 1, "colVal + alignFactor is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE, "colVal + alignFactor is invalid!"));
    uint32_t colValAlign = (static_cast<uint32_t>(colVal) + alignFactor - 1) / alignFactor * alignFactor;
    tilingDataPointer->dataType = dtypeKey - 1;
    uint64_t blockKey = MULTI_CORE;
    uint64_t ubKey = UB_SPLIT_N;
    uint64_t tilingKey;
    if (colVal <= bufferSize) {
        tilingKey = dtypeKey * DTYPE_WEIGHT + blockKey * BLOCK_WEIGHT + ubKey;
        bool blSmallD = LargeNSmallD(kernelInfo, *tilingDataPointer, bufferSize, rowVal, colValAlign, coreNum);
        MKI_CHECK(blSmallD, "blSmallD failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
    } else {
        ubKey = UB_SPLIT_D;
        tilingKey = dtypeKey * DTYPE_WEIGHT + blockKey * BLOCK_WEIGHT + ubKey;
        if (dataType == TENSOR_DTYPE_BF16 || dataType == TENSOR_DTYPE_FLOAT16) {
            bufferSize = BF16_SPLIT_D_BUFFER_SIZE;
        }
        bool blLargeD = LargeNLargeD(kernelInfo, *tilingDataPointer, bufferSize, rowVal, colVal, coreNum);
        MKI_CHECK(blLargeD, "blLargeD failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
    }
    tilingDataPointer->row = static_cast<uint32_t>(rowVal);
    tilingDataPointer->col = static_cast<uint32_t>(colVal);
    tilingDataPointer->avg = avgVal;
    kernelInfo.SetTilingId(tilingKey);
    uint64_t sysWorkspaceSize =
        static_cast<uint64_t>((BLOCK_SIZE + tilingDataPointer->blockDim * BLOCK_SIZE * TWICE_WORKSPACE) * sizeof(int));
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);
    kernelInfo.SetMemsetInfo(WORK_SPACE_INDEX, sysWorkspaceSize);
    return Status::OkStatus();
}
} // namespace AsdOps