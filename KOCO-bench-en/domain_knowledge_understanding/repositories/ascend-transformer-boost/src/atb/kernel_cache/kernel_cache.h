/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_KERNEL_CACHE_H
#define ATB_KERNEL_CACHE_H
#include <vector>
#include <memory>
#include <mki/launch_param.h>
#include <mki/kernel.h>

namespace atb {
using TilingBuffer = std::vector<uint8_t>;
using TilingBufferPtr = TilingBuffer const *;
struct CacheItem {
    Mki::LaunchParam launchParam;
    std::shared_ptr<Mki::Kernel> kernel = nullptr;
    TilingBuffer tilingBuffer;
};

struct CacheSlot {
    std::vector<CacheItem> cachedItems;
    size_t replacePos = 0;
    size_t hitPos = 0;
    size_t validSize = 0;
    void Init(uint32_t cacheItemCount);
    void AddTiling(uint8_t *tilingData, uint64_t tilingSize, const Mki::LaunchParam &launchParam,
                   const Mki::Kernel *kernel);
    TilingBufferPtr GetTilingByIndex(const size_t index, const Mki::LaunchParam &launchParam, Mki::Kernel *&kernel);
    TilingBufferPtr GetTiling(const Mki::LaunchParam &launchParam, Mki::Kernel *&kernel);
};

class KernelCache {
public:
    KernelCache() noexcept;
    ~KernelCache();
    void Init(uint64_t kernelCount, uint32_t cacheItemCount = 1);
    void AddTiling(size_t kernelIndex, uint8_t *tilingData, uint64_t tilingSize, const Mki::LaunchParam &launchParam,
                   const Mki::Kernel *kernel);
    TilingBufferPtr GetTiling(size_t kernelIndex, const Mki::LaunchParam &launchParam, Mki::Kernel *&kernel);

private:
    bool IsValid(size_t kernelIndex) const;

private:
    std::vector<CacheSlot> cachedSlots_;
};
} // namespace atb
#endif