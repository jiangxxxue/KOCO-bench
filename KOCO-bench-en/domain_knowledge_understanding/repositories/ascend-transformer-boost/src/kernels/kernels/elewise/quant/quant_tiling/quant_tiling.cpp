/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "quant_tiling.h"

#include <cstring>

#include <mki/utils/env/env.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/elewise.h"
#include "tiling_data.h"
#include "kernels/norm/common/common_tiling.h"

namespace AsdOps {
Status QuantF16Tiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    QuantF16TilingData *tilingDataPtr = reinterpret_cast<QuantF16TilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingdata should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingdata should not be empty"));
    auto attrs = AnyCast<OpParam::Elewise>(launchParam.GetParam());
    tilingDataPtr->inputScale = *reinterpret_cast<uint32_t *>(&attrs.inputScale);
    tilingDataPtr->inputOffset = *reinterpret_cast<uint32_t *>(&attrs.inputOffset);
    NormTilingDataPtrCon quantPtrCon;
    Status ret = PostLayerNormPtrFunc(tilingDataPtr, quantPtrCon, launchParam, kernelInfo);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    int32_t scalarUsed = 256;
    MKI_CHECK(quantPtrCon.numCol % 32 == 0, "last dim is not 32 bytes align",
              return Status::FailStatus(ERROR_INVALID_VALUE, "last dim is not 32 bytes align")); // 32: 算子约束最后一维32字节对齐要求
    MKI_CHECK(quantPtrCon.nlFirstdimPerCoreNum <
                  (static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) / 2) / quantPtrCon.numCol,
              "numCol or nlFirstdimPerCoreNum is not invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE, "numCol or nlFirstdimPerCoreNum is not invalid"));
    int32_t totalMemNeed = static_cast<int32_t>(2 * quantPtrCon.nlFirstdimPerCoreNum * quantPtrCon.numCol);
    MKI_CHECK(quantPtrCon.maxEleFp16 < static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
        "maxEleFp16 is not invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE, "maxEleFp16 is not invalid"));
    int32_t sumData = static_cast<int32_t>(quantPtrCon.maxEleFp16) - scalarUsed;
    
    ret = CheckSplit(tilingDataPtr, totalMemNeed, sumData, quantPtrCon);
    OP_TILING_CHECK_STATUS_RETURN(ret);
    MKI_CHECK(tilingDataPtr->firstDimPerTimes != 0, "tilingData firstDimPerTimes is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingData firstDimPerTimes is invalid"));
    tilingDataPtr->quantMin = -128; // default int8 min is -128
    const char *quantMinSetPtr = Mki::GetEnv("ASDOPS_QUANT_MIN_NEG_127");
    if (quantMinSetPtr != nullptr && strcmp(quantMinSetPtr, "1") == 0) {
        tilingDataPtr->quantMin = -127; // set int8 min to -127
    }
    MKI_LOG(INFO) << "numCore = " << tilingDataPtr->numCore
                  << ", numFirstDim = " << tilingDataPtr->numFirstDim
                  << ", nlFirstdimPerCore = " << tilingDataPtr->nlFirstdimPerCore
                  << ", lFirstdimPerCore = " << tilingDataPtr->lFirstdimPerCore
                  << ", inputScale = " << *reinterpret_cast<float *>(&tilingDataPtr->inputScale)
                  << ", inputOffset = " << *reinterpret_cast<int *>(&tilingDataPtr->inputOffset)
                  << ", numLastDim = " << tilingDataPtr->numLastDim
                  << ", maxCoreNum = " << quantPtrCon.maxCoreNum
                  << ", maxUbSize = " << quantPtrCon.maxUbSize
                  << ", quantMin = " << tilingDataPtr->quantMin;
    return Status::OkStatus();
}
} // namespace AsdOps
