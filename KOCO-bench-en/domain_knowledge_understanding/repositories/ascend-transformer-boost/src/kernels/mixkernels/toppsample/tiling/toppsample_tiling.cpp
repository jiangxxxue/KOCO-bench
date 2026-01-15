/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include "atbops/params/params.h"
#include "tiling_data.h"
#include "toppsample_tiling.h"

static constexpr uint32_t BLK_SIZE = 32;
static constexpr uint32_t ONE_LOOP_ELE = 16384; // 2 fp16 , 1 fp32 ,1 int8 , 256 * half, 32B
static constexpr uint64_t MULTINOMIAL_WKSP_TENSOR_IDX = 4;

namespace AtbOps {
void RandSeedGenerate(const LaunchParam &launchParam, ToppsampleTilingData &tilingDataPtr, const uint32_t firstDim)
{
    auto para = AnyCast<OpParam::Toppsample>(launchParam.GetParam());
    if (para.randSeed.size() == 1) {
        Utils::SetRandseeds(para.randSeed[0]);
        for (uint32_t i = 0; i < firstDim; i++) {
                tilingDataPtr.randValList[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    } else if (para.randSeed.size() == 0) {
        uint32_t zeroSeed = 0;
        Utils::SetRandseeds(zeroSeed);
        for (uint32_t i = 0; i < firstDim; i++) {
            tilingDataPtr.randValList[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    } else {
        for (uint32_t i = 0; i < firstDim; i++) {
            if (i > para.randSeed.size() - 1) {
                uint32_t zeroSeedOne = 0;
                Utils::SetRandseeds(zeroSeedOne);
                tilingDataPtr.randValList[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            } else {
                Utils::SetRandseeds(para.randSeed[i]);
                tilingDataPtr.randValList[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
        }
    }
}

void FillTilingParam(const LaunchParam &launchParam, ToppsampleTilingData &tilingDataPtr)
{
    uint32_t realLastDim = static_cast<uint32_t>(launchParam.GetInTensor(0).desc.dims[1]);
    uint32_t eleEachBlk = 256; // 512Byte
    uint32_t tempUbEleAligened = ONE_LOOP_ELE; // 256 倍数
    uint32_t perCoreRunNum = 0;
    uint32_t nlElePerCorePerRun = 0; // 每个核非最后一次，单次跑的元素数量
    uint32_t lElePerCoreLastRun = 0; // 每个核最后一次，单次跑的元素数量
    uint32_t expandLastDim = Utils::CeilDiv(realLastDim, eleEachBlk) * eleEachBlk;
    if (tempUbEleAligened < realLastDim) {
        perCoreRunNum = Utils::CeilDiv(realLastDim, tempUbEleAligened); // 每个核做几次，向上取整
        nlElePerCorePerRun = tempUbEleAligened; // 19456
        lElePerCoreLastRun = (realLastDim % tempUbEleAligened != 0)
                                 ? realLastDim - (perCoreRunNum - 1) * nlElePerCorePerRun
                                 : nlElePerCorePerRun;
    } else {
        perCoreRunNum = 1;
        nlElePerCorePerRun = realLastDim;
        lElePerCoreLastRun = realLastDim;
    }
    tilingDataPtr.numSamplesMax = MAX_RAND_NUM;
    tilingDataPtr.realLastDim = realLastDim;
    tilingDataPtr.expandLastDim = expandLastDim;
    tilingDataPtr.perCoreRunNum = perCoreRunNum;
    tilingDataPtr.nlElePerCorePerRun = nlElePerCorePerRun;
    tilingDataPtr.lElePerCoreLastRun = lElePerCoreLastRun;
    tilingDataPtr.tempUbEleAligened = tempUbEleAligened;
}

Status ToppsampleTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    ToppsampleTilingData *tilingDataPtr =
        reinterpret_cast<AtbOps::ToppsampleTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    uint32_t dimSize = launchParam.GetInTensor(0).desc.dims.size();
    int64_t firstDim = 1;
    for (size_t i = 0; i < dimSize - 1; i++) {
        int64_t dim = launchParam.GetInTensor(0).desc.dims[i];
        MKI_CHECK(dim > 0, "dims should be positive",
            return Status::FailStatus(ERROR_INVALID_VALUE));
        MKI_CHECK(firstDim < INT64_MAX / dim, "Integer overflow detected in firstDim calculation",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        firstDim *= dim;
    }
    MKI_CHECK(firstDim > 0 && firstDim <= 512, "batch size should be less than 512",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->firstDim = static_cast<uint32_t>(firstDim);
    FillTilingParam(launchParam, *tilingDataPtr);
    RandSeedGenerate(launchParam, *tilingDataPtr, static_cast<uint32_t>(firstDim));
    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    uint32_t tempCore = (static_cast<uint32_t>(firstDim) + maxCore - 1) / maxCore;
    uint32_t realCore = (static_cast<uint32_t>(firstDim) + tempCore - 1) / tempCore;

    kernelInfo.SetBlockDim(realCore);
    uint64_t sysWorkspaceSize = static_cast<uint64_t>(firstDim) * MAX_RAND_NUM * BLK_SIZE;
    kernelInfo.GetScratchSizes().push_back(sysWorkspaceSize);
    kernelInfo.SetMemsetInfo(MULTINOMIAL_WKSP_TENSOR_IDX, sysWorkspaceSize);
    auto inTensor0 = launchParam.GetInTensor(0).desc; // batch, hiiden
    auto inTensor1 = launchParam.GetInTensor(1).desc; // batch / 1, num_samples
    uint64_t bf16TilingKey = launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16 ? 2 : 0;    // 二进制：10 or 00
    if (inTensor1.dims[0] == 1) {
        kernelInfo.SetTilingId(bf16TilingKey + static_cast<uint64_t>(1));
    } else if (inTensor1.dims[0] == inTensor0.dims[0]) {
        kernelInfo.SetTilingId(bf16TilingKey + static_cast<uint64_t>(0));
    }
    MKI_LOG(INFO) << "numSamplesMax " << tilingDataPtr->numSamplesMax
                  << " realLastDim " << tilingDataPtr->realLastDim
                  << " expandLastDim " << tilingDataPtr->expandLastDim
                  << " firstDim " << tilingDataPtr->firstDim
                  << " perCoreRunNum "<< tilingDataPtr->perCoreRunNum
                  << " nlElePerCorePerRun " << tilingDataPtr->nlElePerCorePerRun
                  << " lElePerCoreLastRun " << tilingDataPtr->lElePerCoreLastRun
                  << " tempUbEleAligened " << tilingDataPtr->tempUbEleAligened
                  << "maxCore" << maxCore
                  << " realCore " << realCore;
    return Status::OkStatus();
}
} // namespace AtbOps