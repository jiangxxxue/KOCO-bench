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
#include <limits>
#include <climits>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include <mki/utils/checktensor/check_tensor.h>
#include <mki/utils/math/math.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"
#include "post_rms_norm_tiling_data.h"
#include "post_rms_norm_tiling.h"

static constexpr uint32_t SLICE_SIZE = 12288; // 用于切分过长的行
static constexpr uint32_t SLICE_SIZE_WITH_BIAS = 8192; // 用于切分过长的行
const constexpr uint32_t FP16_PER_REPEAT = 16;
static constexpr uint32_t SLICE_SIZE_DOUBLE_BUFFER = 7168; // 输入数据末维小于等于7k，开启double buffer(仅A2/A3)
static constexpr uint32_t TILING_KEY_DOUBLE_BUFFER_OFFSET = 1000; // double buffer场景需要在tilingKey上加该值

namespace AsdOps {

void PostRmsNormPrintLog(const PostRmsNormTilingData &tilingDataPtr)
{
    MKI_LOG(INFO) << "PosrRmsNorm Tiling Data:"
                  << " numCore " << tilingDataPtr.numCore << " numCol " << tilingDataPtr.numCol
                  << " numRow " << tilingDataPtr.numRow << " avgFactor " << tilingDataPtr.avgFactor
                  << " epsilon " << tilingDataPtr.epsilon << " sliceSize " << tilingDataPtr.sliceSize
                  << " precisionMode " << tilingDataPtr.precisionMode;
}

uint64_t CalcTilingKey(const LaunchParam &launchParam, const PostRmsNormTilingData &tilingDataPtr)
{
    auto numCol = tilingDataPtr.numCol;
    bool longSeq = numCol > tilingDataPtr.sliceSize ? true : false;
    bool isSmallNumCol = numCol <= SLICE_SIZE_DOUBLE_BUFFER ? true : false;
    bool isA2orA3 = PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B ? true : false;
    bool biasEmpty = CheckEmptyTensor(launchParam.GetInTensor(TENSOR_BIAS_IDX));
    bool isBf16 = launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16;
    uint64_t tilingKey = longSeq;
    tilingKey = (tilingKey << 1) + static_cast<uint64_t>(biasEmpty);
    tilingKey = (tilingKey << 1) + static_cast<uint64_t>(isBf16);
    if (isSmallNumCol && isA2orA3) {
        tilingKey += TILING_KEY_DOUBLE_BUFFER_OFFSET;
    }
    return tilingKey;
}
Status PostRmsNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    PostRmsNormTilingData *tilingDataPtr = reinterpret_cast<PostRmsNormTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    const Mki::SVector<int64_t> &shape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(!shape.empty(), "shape should not be empty", return Status::FailStatus(ERROR_INVALID_VALUE));
    int64_t tmpNumRow = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        MKI_CHECK(shape[i] > 0 && tmpNumRow <= UINT_MAX / shape[i],
                     "tmpNumRow is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE, "tmpNumRow is invalid!"));
        tmpNumRow *= shape[i];
    }
    tilingDataPtr->numRow = static_cast<uint32_t>(tmpNumRow);
    MKI_CHECK(shape[shape.size() - 1] <= UINT_MAX, "numCol invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->numCol = static_cast<uint32_t>(shape[shape.size() - 1]);
    bool isDoubleBuffer = tilingDataPtr->numCol <= SLICE_SIZE_DOUBLE_BUFFER &&
        PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B;
    tilingDataPtr->numCore = Utils::CeilDiv(tilingDataPtr->numRow, Utils::CeilDiv(tilingDataPtr->numRow, coreNum));
    MKI_CHECK(tilingDataPtr->numCore <= UINT_MAX - tilingDataPtr->numRow,
        "numRow + numCore is invalid!", return Status::FailStatus(ERROR_INVALID_VALUE));
    float factor = static_cast<float>(1.0 / tilingDataPtr->numCol);
    tilingDataPtr->avgFactor = *static_cast<uint32_t *>(static_cast<void *>(&factor));
    auto param = AnyCast<OpParam::Norm>(launchParam.GetParam());
    if (param.epsilon <= 0) {
        return Status::FailStatus(ERROR_INVALID_VALUE,
            "Invalid parameter: epsilon cannot be " + std::to_string(param.epsilon));
    }
    if (param.precisionMode != 0 && param.precisionMode != 1) {
        return Status::FailStatus(ERROR_INVALID_VALUE,
            "Invalid parameter: precisionMode cannot be " + std::to_string(param.precisionMode));
    }
    tilingDataPtr->precisionMode = param.precisionMode;
    tilingDataPtr->epsilon = *static_cast<uint32_t *>(static_cast<void *>(&param.epsilon));
    if (CheckEmptyTensor(launchParam.GetInTensor(TENSOR_BIAS_IDX))) { // 空Bias
        tilingDataPtr->sliceSize = isDoubleBuffer ? SLICE_SIZE_DOUBLE_BUFFER : SLICE_SIZE; // 7168/12288
    } else {
        tilingDataPtr->sliceSize = isDoubleBuffer ? SLICE_SIZE_DOUBLE_BUFFER : SLICE_SIZE_WITH_BIAS; // 7168/8192
    }
    MKI_CHECK(tilingDataPtr->numCol <= UINT_MAX - tilingDataPtr->sliceSize, "numSlice calculation invalid!",
        return Status::FailStatus(ERROR_INVALID_VALUE, "numSlice calculation invalid!"));
    MKI_CHECK(tilingDataPtr->numCol <= UINT_MAX - FP16_PER_REPEAT, "numSlice calculation invalid!",
        return Status::FailStatus(ERROR_INVALID_VALUE, "numSlice calculation invalid!"));
    uint64_t tilingKey = CalcTilingKey(launchParam, *tilingDataPtr);
    MKI_LOG(INFO) << "post rmsnorm tilingKey is : " << tilingKey;
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);
    kernelInfo.SetTilingId(tilingKey);
    PostRmsNormPrintLog(*tilingDataPtr);
    return Status::OkStatus();
}
} // namespace AsdOps
