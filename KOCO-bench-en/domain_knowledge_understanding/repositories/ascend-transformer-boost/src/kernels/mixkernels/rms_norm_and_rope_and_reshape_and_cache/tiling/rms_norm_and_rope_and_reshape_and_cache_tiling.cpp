/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <climits>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "rms_norm_and_rope_and_reshape_and_cache_tiling_data.h"
#include "rms_norm_and_rope_and_reshape_and_cache_tiling.h"
 
namespace {
constexpr uint32_t INPUT_X_IDX = 0;
constexpr uint32_t INPUT_KEYROPE_IDX = 2;
constexpr uint32_t INPUT_COS_IDX = 3;
constexpr uint32_t INPUT_KEYRAC_IDX = 5;
constexpr uint32_t INPUT_SLOTMAPPINF_IDX = 6;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t ATTR_0 = 0;
constexpr uint32_t ATTR_1 = 1;
constexpr uint32_t ATTR_2 = 2;
constexpr uint32_t ATTR_3 = 3;
static constexpr uint32_t TILING_FP16 = 101;
static constexpr uint32_t TILING_BF16 = 102;
constexpr uint32_t FP16DATASIZE = 2;
constexpr uint32_t FP32DATASIZE = 4;
 
constexpr uint32_t NUMHEAD = 1;
 
template <typename T>
T CeilDiv(T x, T y)
{
    return y == 0 ? x : (x + y - 1) / y;
}
} // namespace
 
namespace AtbOps {
    Status TilingKeyChose(const LaunchParam &launchParam, KernelInfo &kernelInfo, RmsNormAndRopeAndReshapeAndCacheTilingData *tilingDataPtr)
{
    auto platformType = PlatformInfo::Instance().GetPlatformType();
    if (platformType != PlatformType::ASCEND_910B) {
        MKI_LOG(ERROR) << "BF16 only supports 800I A2";
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16) {
        kernelInfo.SetTilingId(TILING_BF16);
    } else if (launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16) {
        kernelInfo.SetTilingId(TILING_FP16);
    }
    MKI_LOG(INFO) << "RmsNormAndRopeAndReshapeAndCache TilingKey: " << kernelInfo.GetTilingId();
    return Status::OkStatus();
}

void PrintRmsNormAndRopeAndReshapeAndCacheTilingParams(const RmsNormAndRopeAndReshapeAndCacheTilingData &tilingPtr)
{
    MKI_LOG(INFO) << "RmsNormAndRopeAndReshapeAndCache Tiling Data: "
                  << "numRow: " << tilingPtr.numRow << ", numCol: " << tilingPtr.numCol
                  << ", numCore: " << tilingPtr.numCore << ", avgFactor: " << tilingPtr.avgFactor
                  << ", precisionMode: " << tilingPtr.precisionMode << ", epsilon: " << tilingPtr.epsilon
                  << ", ropeNumTokens: " << tilingPtr.ropeNumTokens << ", ropeHeadNumK: " << tilingPtr.ropeHeadNumK
                  << ", ropeHeadDimK: " << tilingPtr.ropeHeadDimK << ", ropeHiddenSizeK: " << tilingPtr.ropeHiddenSizeK
                  << ", racNumTokens: " << tilingPtr.racNumTokens << ", racHeadSizeK: " << tilingPtr.racHeadSizeK
                  << ", slotMappingLen: " << tilingPtr.slotMappingLen << ", racNumHeads: " << tilingPtr.racNumHeads
                  << ", rotaryCoeff: " << tilingPtr.rotaryCoeff;
}
 
Status RmsNormAndRopeAndReshapeAndCacheTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    if (launchParam.GetParam().Type() != typeid(OpParam::RmsNormAndRopeAndReshapeAndCache)) {
        return Status::FailStatus(ERROR_ATTR_INVALID_TYPE,
            "Failed to check RmsNormAndRopeAndReshapeAndCache param");
    }
 
    RmsNormAndRopeAndReshapeAndCacheTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::RmsNormAndRopeAndReshapeAndCacheTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingHost should not be empty",
        return Status::FailStatus(ERROR_INVALID_VALUE, "tilingHost should not be empty"));
    auto ret = TilingKeyChose(launchParam, kernelInfo, tilingDataPtr);
    if (!ret.Ok()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    const Mki::SVector<int64_t> &xShape = launchParam.GetInTensor(INPUT_X_IDX).desc.dims;
    tilingDataPtr->numRow = 1;
    for (size_t i = 0; i < xShape.size() - 1; ++i) {
        tilingDataPtr->numRow *= xShape[i];
    }
    tilingDataPtr->numCol = static_cast<uint32_t>(xShape[xShape.size() - 1]);
    MKI_CHECK(tilingDataPtr->numCol > 0, "numCol should > 0",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->avgFactor = static_cast<float>(1.0 / tilingDataPtr->numCol);
 
    const auto coreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    tilingDataPtr->numCore = CeilDiv(tilingDataPtr->numRow, CeilDiv(tilingDataPtr->numRow, coreNum));
    MKI_CHECK(tilingDataPtr->numCore > 0, "numCore <= 0, which is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
 
    const auto param = AnyCast<OpParam::RmsNormAndRopeAndReshapeAndCache>(launchParam.GetParam());
    tilingDataPtr->epsilon = static_cast<float>(param.epsilon);
    tilingDataPtr->precisionMode = static_cast<uint32_t>(param.precisionMode);
    tilingDataPtr->rotaryCoeff = static_cast<uint32_t>(param.rotaryCoeff);
    MKI_CHECK(tilingDataPtr->rotaryCoeff > 0, "rotaryCoeff <= 0, which is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    
    const Mki::SVector<int64_t> &keyRopeShape = launchParam.GetInTensor(INPUT_KEYROPE_IDX).desc.dims;
    const Mki::SVector<int64_t> &cosShape = launchParam.GetInTensor(INPUT_COS_IDX).desc.dims;
    const Mki::SVector<int64_t> &slotMappingShape = launchParam.GetInTensor(INPUT_SLOTMAPPINF_IDX).desc.dims;
    tilingDataPtr->ropeNumTokens = keyRopeShape.at(DIM_0);
    tilingDataPtr->ropeHiddenSizeK = keyRopeShape.at(DIM_1);
    tilingDataPtr->ropeHeadDimK = static_cast<uint32_t>(cosShape.at(DIM_1));
    MKI_CHECK(tilingDataPtr->ropeHeadDimK > 0, "ropeHeadDimK <= 0, which is invalid",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->ropeHeadNumK = tilingDataPtr->ropeHiddenSizeK / tilingDataPtr->ropeHeadDimK;
    tilingDataPtr->racNumTokens = cosShape.at(DIM_0);
    tilingDataPtr->racNumHeads = NUMHEAD;
    tilingDataPtr->racHeadSizeK = tilingDataPtr->ropeHiddenSizeK + xShape.at(DIM_2);
    tilingDataPtr->slotMappingLen = slotMappingShape.at(DIM_0);

    auto maxUbSize = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    uint32_t allSize = tilingDataPtr->numCol * (3 * FP16DATASIZE + 5 * FP32DATASIZE) +
        tilingDataPtr->racHeadSizeK * FP16DATASIZE;
    uint32_t maxNPerLoopForUb = maxUbSize / allSize;  // ub每次能载入最大行数（包括所有计算数据）
    MKI_CHECK(maxNPerLoopForUb > 0, "The input vector is too large. It is not supported currently.",
              return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->maxRow = maxNPerLoopForUb;
 
    kernelInfo.SetBlockDim(tilingDataPtr->numCore);
 
    PrintRmsNormAndRopeAndReshapeAndCacheTilingParams(*tilingDataPtr);
    return Status::OkStatus();
}
} // namespace AtbOps