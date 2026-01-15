/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/assert/assert.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/group_topk.h"
#include "tiling_data.h"
#include "group_topk_tiling.h"

namespace {
constexpr uint32_t DATABLOCK_SIZE = 32;
constexpr uint32_t SORT_REPEAT_COUNT = 32;
constexpr uint32_t B16_SIZE = 2;
constexpr uint32_t B32_SIZE = 4;
constexpr uint32_t DATABLOCK_NUM_PER_REPEAT = 8;
constexpr uint32_t B16_PER_BLOCK = DATABLOCK_SIZE / B16_SIZE;
constexpr uint32_t MAX_TILE_SIZE = 16 * 1024;
constexpr uint64_t SINGLE_VALUE_GROUP_TILING_KEY = 900;
constexpr uint64_t TILING_KEY_COEFFICIENT = 10;
constexpr uint64_t TILING_KEY_VERSION = 20000000000;

} // namespace

namespace AsdOps {
void PrintGroupTopkTilingParams(const GroupTopkTilingData &tilingPtr)
{
    MKI_LOG(INFO) << "GroupTopk Tiling Data: "
                  << "groupNum: " << tilingPtr.groupNum << ", k: " << tilingPtr.k
                  << ", expertNumPerGroup: " << tilingPtr.expertNumPerGroup
                  << ", expertNumPerGroupPadded: " << tilingPtr.expertNumPerGroupPadded
                  << ", expertNumPerToken: " << tilingPtr.expertNumPerToken
                  << ", tokenNumPerCore: " << tilingPtr.tokenNumPerCore << ", tailTokenNum: " << tilingPtr.tailTokenNum;
}

Status GroupTopkTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    if (launchParam.GetParam().Type() != typeid(OpParam::GroupTopk)) {
        return Status::FailStatus(
            ERROR_ATTR_INVALID_TYPE,
            "Failed to check GroupTopk param, type of specificParam is not equals to OpParam::GroupTopk");
    }
    auto *tilingAddr = kernelInfo.GetTilingHostAddr();
    MKI_CHECK(tilingAddr, "tiling is nullptr", return Status::FailStatus(ERROR_INVALID_VALUE, "tiling is nullptr!"));
    auto tilingPtr = reinterpret_cast<GroupTopkTilingData *>(tilingAddr);
    const auto coreNum = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    const auto tokenNum = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims.at(0));
    const auto expertNum = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims.at(1));
    const auto param = AnyCast<OpParam::GroupTopk>(launchParam.GetParam());
    const auto dtype = launchParam.GetInTensor(0).desc.dtype;
    tilingPtr->groupNum = static_cast<uint32_t>(param.groupNum);
    tilingPtr->k = static_cast<uint32_t>(param.k);
    tilingPtr->kInner = static_cast<uint32_t>(param.kInner);
    tilingPtr->expertNumPerGroup = expertNum / static_cast<uint32_t>(param.groupNum);
    const uint32_t expertNumPerGroupAlign = param.kInner > 1 ? SORT_REPEAT_COUNT : B16_PER_BLOCK;
    tilingPtr->expertNumPerGroupPadded = Mki::Utils::RoundUp(tilingPtr->expertNumPerGroup, expertNumPerGroupAlign);
    tilingPtr->expertNumPerToken = expertNum;
    tilingPtr->tokenNumPerCore = tokenNum / coreNum; // 每个核平均分到的token数量
    tilingPtr->tailTokenNum = tokenNum % coreNum;
    PrintGroupTopkTilingParams(*tilingPtr);
    kernelInfo.SetBlockDim(std::min(tokenNum, coreNum));
    uint64_t tilingKey{0};
    if (tilingPtr->expertNumPerGroup != 1) {
        tilingKey = static_cast<uint64_t>(tilingPtr->kInner > 1);
        tilingKey = tilingKey * TILING_KEY_COEFFICIENT + static_cast<uint64_t>(dtype == TENSOR_DTYPE_BF16);
        tilingKey += TILING_KEY_VERSION;
    } else {
        tilingKey = SINGLE_VALUE_GROUP_TILING_KEY + static_cast<uint64_t>(dtype == TENSOR_DTYPE_BF16);
    }
    kernelInfo.SetTilingId(tilingKey);
    return Status::OkStatus();
}
} // namespace AsdOps
