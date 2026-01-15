/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/time.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "asdops/params/multinomial.h"
#include "tiling_data.h"
#include "multinomial_tiling.h"

static constexpr uint32_t BLK_SIZE = 32;
static constexpr uint32_t REMAIN_SPACE = 32768;
static constexpr uint64_t MULTINOMIAL_WKSP_TENSOR_IDX = 2;

namespace AsdOps {
void SetRandseed(uint32_t &randseed)
{
    if (randseed == 0xffffffff) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        randseed = static_cast<uint32_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000); // 1000 convert in us/ms/s
    }
    srand(randseed);
}

void FillTilingParam(const LaunchParam &launchParam, MultinomialTilingData &tilingDataPtr, uint64_t &firstDim)
{
    auto para = AnyCast<OpParam::Multinomial>(launchParam.GetParam());
    uint32_t randseed = para.randSeed;
    SetRandseed(randseed);
    uint32_t realLastDim = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[1]);
    uint32_t dimSize = launchParam.GetInTensor(0).desc.dims.size();
    for (size_t i = 0; i < dimSize - 1; i++) {
        firstDim *= static_cast<uint64_t>(launchParam.GetInTensor(0).desc.dims[i]);
    }
    uint32_t maxUbSizeMulti = static_cast<uint32_t>(PlatformInfo::Instance().GetUbSize());
    uint32_t tempUbEle = (maxUbSizeMulti - REMAIN_SPACE) / 2; // 2 for sizeof(half)
    uint32_t eleEachBlk = BLK_SIZE / 2; // curDataSize :2 for sizeof(half)
    uint32_t tempUbEleAligened = (tempUbEle / eleEachBlk) * eleEachBlk;
    uint32_t perCoreRunNum = 0;
    uint32_t nlElePerCorePerRun = 0;
    uint32_t lElePerCoreLastRun = 0;
    uint32_t expandLastDim = (realLastDim + eleEachBlk - 1) / eleEachBlk * eleEachBlk;
    if (tempUbEleAligened < realLastDim) {
        perCoreRunNum = (realLastDim + tempUbEleAligened - 1) / tempUbEleAligened;
        nlElePerCorePerRun = tempUbEleAligened; // 40960
        lElePerCoreLastRun = (realLastDim % tempUbEleAligened != 0)
                                 ? realLastDim - (perCoreRunNum - 1) * nlElePerCorePerRun
                                 : nlElePerCorePerRun;
        lElePerCoreLastRun = (lElePerCoreLastRun + eleEachBlk - 1) / eleEachBlk * eleEachBlk;
    } else {
        perCoreRunNum = 1;
        nlElePerCorePerRun = expandLastDim;
        lElePerCoreLastRun = expandLastDim;
    }
    for (uint32_t i = 0; i < MAX_RAND_NUM; i++) {
        tilingDataPtr.randValList[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    MKI_LOG(INFO) << "multinomial randseed " << randseed << ", randVal[0] " << tilingDataPtr.randValList[0];
    tilingDataPtr.numSamplesMax = MAX_RAND_NUM;
    tilingDataPtr.realLastDim = realLastDim;
    tilingDataPtr.expandLastDim = expandLastDim;
    tilingDataPtr.firstDim = firstDim;
    tilingDataPtr.numSamples = para.numSamples;
    tilingDataPtr.perCoreRunNum = perCoreRunNum;
    tilingDataPtr.nlElePerCorePerRun = nlElePerCorePerRun;
    tilingDataPtr.lElePerCoreLastRun = lElePerCoreLastRun;
    tilingDataPtr.tempUbEleAligened = tempUbEleAligened;
}

Status MultinomialTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    MultinomialTilingData *tilingDataPtr =
        reinterpret_cast<AsdOps::MultinomialTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));

    uint64_t firstDim = 1;
    FillTilingParam(launchParam, *tilingDataPtr, firstDim);
    kernelInfo.SetBlockDim(firstDim);
    uint64_t maxCore = static_cast<uint64_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    uint64_t sysWorkspaceSize = maxCore * static_cast<uint64_t>(BLK_SIZE) * static_cast<uint64_t>(firstDim);
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);
    kernelInfo.SetMemsetInfo(MULTINOMIAL_WKSP_TENSOR_IDX, sysWorkspaceSize);
    return Status::OkStatus();
}
} // namespace AsdOps