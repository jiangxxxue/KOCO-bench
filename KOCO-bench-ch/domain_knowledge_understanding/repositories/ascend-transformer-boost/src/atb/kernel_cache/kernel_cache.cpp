/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/kernel_cache/kernel_cache.h"
#include <functional>
#include <securec.h>
#include <map>
#include <asdops/params/params.h>
#include <mki/utils/time/timer.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_compare.h"
#include "atb/utils/statistic.h"

namespace atb {
constexpr uint64_t DEFAULT_TILING_SIZE = 10240;
void CacheSlot::Init(uint32_t cacheItemCount)
{
    cachedItems.resize(cacheItemCount);
    for (auto &cacheItem : cachedItems) {
        cacheItem.tilingBuffer.reserve(DEFAULT_TILING_SIZE);
    }
}

void CacheSlot::AddTiling(uint8_t *tilingData, uint64_t tilingSize, const Mki::LaunchParam &launchParam,
                          const Mki::Kernel *kernel)
{
    if (replacePos >= cachedItems.size()) {
        replacePos = 0;
    }
    auto &cachedItem = cachedItems.at(replacePos);
    cachedItem.launchParam = launchParam;
    if (kernel != nullptr) {
        cachedItem.kernel.reset(kernel->Clone());
    }
    cachedItem.tilingBuffer.resize(tilingSize);
    int ret = memcpy_s(cachedItem.tilingBuffer.data(), tilingSize, tilingData, tilingSize);
    ATB_LOG_IF(ret != EOK, ERROR) << "memcpy_s Error! Error Code: " << ret;

    replacePos++;
    validSize = replacePos > validSize ? replacePos : validSize;
}

TilingBufferPtr CacheSlot::GetTilingByIndex(const size_t index, const Mki::LaunchParam &launchParam,
                                            Mki::Kernel *&kernel)
{
    auto &cachedItem = cachedItems.at(index);
    Mki::Timer timer;

    bool equal = IsLaunchParamEqual(cachedItem.launchParam, launchParam);
    GetOpSetupStatistic().kernelCacheCompareRunInfoTime += timer.ElapsedMicroSecond();
    if (equal) {
        Mki::Timer kernelCacheTimer;
        if (cachedItem.kernel != nullptr) {
            kernel = cachedItem.kernel->Clone();
        }
        GetOpSetupStatistic().kernelCacheGetRunInfoTime += kernelCacheTimer.ElapsedMicroSecond();
        hitPos = index;
        return &cachedItem.tilingBuffer;
    }
    return nullptr;
}

TilingBufferPtr CacheSlot::GetTiling(const Mki::LaunchParam &launchParam, Mki::Kernel *&kernel)
{
    TilingBufferPtr tilingBuffeerAddr = nullptr;
    for (size_t i = hitPos; i < validSize; i++) {
        tilingBuffeerAddr = GetTilingByIndex(i, launchParam, kernel);
        if (tilingBuffeerAddr != nullptr) {
            return tilingBuffeerAddr;
        }
    }
    for (size_t i = 0; i < hitPos; i++) {
        tilingBuffeerAddr = GetTilingByIndex(i, launchParam, kernel);
        if (tilingBuffeerAddr != nullptr) {
            return tilingBuffeerAddr;
        }
    }
    return tilingBuffeerAddr;
}

KernelCache::KernelCache() noexcept {}

KernelCache::~KernelCache() {}

void KernelCache::Init(uint64_t kernelCount, uint32_t cacheItemCount)
{
    if (cachedSlots_.empty()) {
        cachedSlots_.resize(kernelCount);
        for (auto &cachedSlot : cachedSlots_) {
            cachedSlot.Init(cacheItemCount);
        }
    }
}

void KernelCache::AddTiling(size_t kernelIndex, uint8_t *tilingData, uint64_t tilingSize,
                            const Mki::LaunchParam &launchParam, const Mki::Kernel *kernel)
{
    if (IsValid(kernelIndex)) {
        auto &cacheSlot = cachedSlots_.at(kernelIndex);
        cacheSlot.AddTiling(tilingData, tilingSize, launchParam, kernel);
    }
}

TilingBufferPtr KernelCache::GetTiling(size_t kernelIndex, const Mki::LaunchParam &launchParam, Mki::Kernel *&kernel)
{
    if (IsValid(kernelIndex)) {
        auto &cacheSlot = cachedSlots_.at(kernelIndex);
        return cacheSlot.GetTiling(launchParam, kernel);
    }
    return nullptr;
}

bool KernelCache::IsValid(size_t kernelIndex) const
{
    return static_cast<uint64_t>(kernelIndex) < cachedSlots_.size();
}
} // namespace atb