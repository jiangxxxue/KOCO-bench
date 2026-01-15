/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki_loader/op_register.h>
#include <iostream>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "swi_glu_quant_tiling_utils.h"
#include "swi_glu_quant_tiling.h"

static constexpr uint32_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t BLOCK_SIZE = 32;
static constexpr uint32_t L2_CACHE_LINE_SIZE = 512; // pack unit in cache 512B
static constexpr uint32_t SIZE_OF_FLOAT16 = 2;

static constexpr uint32_t SINGLE_UB_SIZE = 25;

static constexpr int TILING_KEY_BF16_QUANT_MODE = 206; // Tiling key for BF16 quantization mode
static constexpr int TILING_KEY_FP16_QUANT_MODE = 106; // Tiling key for FP16 quantization mode
static constexpr int TILING_KEY_FP32_QUANT_MODE = 306; // Tiling key for FP32 quantization mode

namespace AtbOps  {
using namespace Mki;
void SetTilingData(SwiGluQuantTilingData &tilingData)
{
    tilingData.basicRowLenHeadCore = tilingData.optBaseRowLenHeadCore;
    tilingData.basicRowLenTailCore = tilingData.optBaseRowLenTailCore;
    tilingData.basicColLen = tilingData.optBaseColLen;
    tilingData.realCoreNum = tilingData.coreNumUsed;
}
bool CalTilingData(SwiGluQuantTilingData& tilingData)
{
    uint32_t rowLen = tilingData.rowLen;
    tilingData.coreNumUsed = Max(Min(tilingData.totalCore, rowLen), ONE);
    tilingData.headCoreNum = rowLen % tilingData.coreNumUsed;
    tilingData.rowLenPerHeadCore = (rowLen + tilingData.coreNumUsed - 1) / tilingData.coreNumUsed;
    tilingData.rowLenPerTailCore = rowLen / tilingData.coreNumUsed;
    return CalculateMaxUbSizePerRow(tilingData);
}
void PrintSwiQuantTiling(const SwiGluQuantTilingData &tilingData)
{
    MKI_LOG(INFO) << "SwiGlu Tiling Data:"
                  << "\n"
                  << " groupLen " << tilingData.groupLen << "\n"
                  << " rowLen " << tilingData.rowLen << "\n"
                  << " colLen " << tilingData.colLen << "\n"
                  << " rowLenPerHeadCore " << tilingData.rowLenPerHeadCore << "\n"
                  << " rowLenPerTailCore " << tilingData.rowLenPerTailCore << "\n"
                  << " basicRowLenHeadCore " << tilingData.basicRowLenHeadCore << "\n"
                  << " basicRowLenTailCore " << tilingData.basicRowLenTailCore << "\n"
                  << " basicColLen  " << tilingData.basicColLen << "\n"
                  << " headCoreNum " << tilingData.headCoreNum << "\n"
                  << " realCoreNum  " << tilingData.realCoreNum << "\n"
                  << " totalCore  " << tilingData.totalCore;
}

void SwigluQuantTilingKeyChose(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    auto xDtype = launchParam.GetInTensor(0).desc.dtype;
    if ((xDtype == TENSOR_DTYPE_FLOAT16)) {
        kernelInfo.SetTilingId(TILING_KEY_FP16_QUANT_MODE);
    }
    if ((xDtype == TENSOR_DTYPE_BF16)) {
        kernelInfo.SetTilingId(TILING_KEY_BF16_QUANT_MODE);
    }
    MKI_LOG(INFO) << "swigluquant tilingKey is : " << kernelInfo.GetTilingId();
}

Status SwiGluQuantTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "Only supports 800I A2/A3",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_LOG(INFO) << "----- [ Enter SwiGluForwardTiling ] -----";
    SwiGluQuantTilingData *tilingData = reinterpret_cast<SwiGluQuantTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingData != nullptr, "tilingData should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t totalCore = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    uint32_t ubSize = PlatformInfo::Instance().GetUbSize();
    tilingData->totalCore = totalCore;
    tilingData->ubSize = ubSize;
    tilingData->groupLen = tilingData->groupLength;
    tilingData->dataNumSingleUb = tilingData->ubSize / SINGLE_UB_SIZE;
    tilingData->blockNum = BLOCK_SIZE / SIZE_OF_FLOAT16;
    tilingData->cacheLineLen = L2_CACHE_LINE_SIZE / SIZE_OF_FLOAT16;
    const Mki::SVector<int64_t> &xShape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK_NO_LOG(SetTotalShape(xShape, *tilingData), return Status::FailStatus(ERROR_INVALID_VALUE));
    MKI_CHECK_NO_LOG(CalTilingData(*tilingData), return Status::FailStatus(ERROR_INVALID_VALUE));
    SetTilingData(*tilingData);
    SwigluQuantTilingKeyChose(launchParam, kernelInfo);
    kernelInfo.SetBlockDim(tilingData->coreNumUsed);
    
    PrintSwiQuantTiling(*tilingData);
    return Status::OkStatus();
}
}