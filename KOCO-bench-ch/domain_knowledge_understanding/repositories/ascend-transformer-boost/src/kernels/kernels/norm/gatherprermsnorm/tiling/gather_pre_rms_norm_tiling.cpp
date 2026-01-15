/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/math/math.h>
#include "asdops/params/params.h"
#include "gather_pre_rms_norm_tiling_data.h"
#include "gather_pre_rms_norm_tiling.h"

static constexpr uint32_t DOUBLE_BUFFER_TBUF_NUM = 24;  // 2*fp16(x)+2*fp16(resIn)+fp16(g)+fp16(resOut)+fp32(y)+2*fp32
static constexpr uint32_t RESERVE_UB_BYTE = 768;

namespace AsdOps {

constexpr uint32_t GATHERPRENORM_TILING_KEY_BASE = 100;
constexpr uint32_t GATHERPRENORM_TILING_KEY_DTYPE = 10;    // 0: fp16; 1: bf16
constexpr uint32_t GATHERPRENORM_TILING_KEY_WITH_LARGE_INDICES = 1; // 0: small indices; 1: large indices

void GatherPreRmsNormPrintLog(const GatherPreRmsNormTilingData &tilingDataPtr)
{
    MKI_LOG(INFO) << "GatherPreRmsNorm Tiling Data:"
                  << " numCore " << tilingDataPtr.numCore << " numCol " << tilingDataPtr.numCol
                  << " numRow " << tilingDataPtr.numRow << " avgFactor " << tilingDataPtr.avgFactor
                  << " epsilon " << tilingDataPtr.epsilon << " ubResByte " << tilingDataPtr.ubResByte
                  << " numRowPerCore " << tilingDataPtr.numRowPerCore
                  << " numRowPerCoreAlign " << tilingDataPtr.numRowPerCoreAlign
                  << " numRowTailCore " << tilingDataPtr.numRowTailCore;
}

uint64_t CalcTilingKey(const LaunchParam &launchParam,
                       bool hasLargeIndices)
{
    uint64_t tilingKey = GATHERPRENORM_TILING_KEY_BASE;
    tilingKey += launchParam.GetInTensor(TENSOR_X_IDX).desc.dtype == TENSOR_DTYPE_BF16 ?
                                                                     GATHERPRENORM_TILING_KEY_DTYPE : 0;
    tilingKey += hasLargeIndices ? GATHERPRENORM_TILING_KEY_WITH_LARGE_INDICES : 0;
    return tilingKey;
}

Status GatherPreRmsNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    GatherPreRmsNormTilingData *tilingDataPtr =
        reinterpret_cast<GatherPreRmsNormTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr invalid", return Status::FailStatus(ERROR_INVALID_VALUE));

    uint32_t maxCoreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    MKI_CHECK(maxCoreNum > 0, "maxCoreNum is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));

    const Mki::SVector<int64_t> &inputXShape = launchParam.GetInTensor(TENSOR_X_IDX).desc.dims;
    const Mki::SVector<int64_t> &resInShape = launchParam.GetInTensor(TENSOR_RES_IN_IDX).desc.dims;
    MKI_CHECK(inputXShape[0] <= UINT_MAX && resInShape[0] <= UINT_MAX,
              "numRow invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->numRow = static_cast<uint32_t>(inputXShape[0]);

    MKI_CHECK(inputXShape[inputXShape.size() - 1] <= UINT_MAX, "numCol invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->numCol = static_cast<uint32_t>(inputXShape[inputXShape.size() - 1]);

    tilingDataPtr->numCore = static_cast<uint32_t>(Utils::CeilDiv(tilingDataPtr->numRow,
        Utils::CeilDiv(static_cast<uint32_t>(tilingDataPtr->numRow), maxCoreNum)));
    MKI_CHECK(tilingDataPtr->numCore <= UINT_MAX - tilingDataPtr->numRow, "numRow + numCore is invalid!",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);
    tilingDataPtr->numRowPerCore = (tilingDataPtr->numRow + tilingDataPtr->numCore - 1) / tilingDataPtr->numCore;
    tilingDataPtr->numRowPerCoreAlign = (tilingDataPtr->numRowPerCore + INT32_ALIGN_NUM - 1) /
                                        INT32_ALIGN_NUM * INT32_ALIGN_NUM;
    tilingDataPtr->numRowTailCore = tilingDataPtr->numRow - (tilingDataPtr->numCore - 1) * tilingDataPtr->numRowPerCore;

    tilingDataPtr->avgFactor = static_cast<float>(1.0 / tilingDataPtr->numCol);

    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    if (param.epsilon <= 0) {
        return Status::FailStatus(ERROR_INVALID_VALUE,
            "Invalid parameter: epsilon cannot be " + std::to_string(param.epsilon));
    }
    tilingDataPtr->epsilon = static_cast<float>(param.epsilon);

    uint32_t ubSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());   // A2: 192K
    tilingDataPtr->ubResByte = ubSize - static_cast<uint32_t>(tilingDataPtr->numCol * DOUBLE_BUFFER_TBUF_NUM) -
                               BRCB_BYTE - RESERVE_UB_BYTE;

    bool hasLargeIndices = tilingDataPtr->ubResByte < tilingDataPtr->numRowPerCoreAlign * sizeof(int32_t);

    uint64_t tilingKey = CalcTilingKey(launchParam, hasLargeIndices);
    MKI_LOG(INFO) << "GatherPreRmsNorm tilingKey is : " << tilingKey;
    kernelInfo.SetTilingId(tilingKey);
    GatherPreRmsNormPrintLog(*tilingDataPtr);
    return Status::OkStatus();
}
} // namespace AsdOps
