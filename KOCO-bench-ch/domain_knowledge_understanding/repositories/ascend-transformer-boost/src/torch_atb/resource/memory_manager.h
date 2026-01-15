/*
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_MEMORY_MANAGER_H
#define TORCH_ATB_MEMORY_MANAGER_H
#include <cstdint>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include "buffer_device.h"

namespace TorchAtb {
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();
    static MemoryManager &GetMemoryManager();
    static void SetBufferRing(uint64_t ring);
    static void SetBufferSize(uint64_t size);
    void *GetWorkspaceBuffer(uint64_t bufferSize);

private:
    static uint64_t bufferRing_;
    static uint64_t bufferSize_;
    static std::mutex mutex_;
    std::vector<std::unique_ptr<BufferDevice>> workspaceBuffers_;
    size_t workspaceBufferOffset_ = 0;
};
} // namespace TorchAtb
#endif