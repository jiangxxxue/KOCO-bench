/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "memory_manager.h"
#include <atb/utils/log.h>

namespace TorchAtb {
std::mutex MemoryManager::mutex_;

uint64_t MemoryManager::bufferRing_ = 1;                          // 设置默认值为1
uint64_t MemoryManager::bufferSize_ = 100ULL * 1024ULL * 1024ULL; // 设置默认值为100MB

MemoryManager::MemoryManager()
{
    uint64_t bufferRing = bufferRing_;
    uint64_t bufferSize = bufferSize_;
    ATB_LOG(INFO) << "MemoryManager workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

void MemoryManager::SetBufferRing(uint64_t ring)
{
    std::lock_guard<std::mutex> lock(mutex_);
    bufferRing_ = ring;
}

void MemoryManager::SetBufferSize(uint64_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    bufferSize_ = size;
}

MemoryManager::~MemoryManager() {}

MemoryManager &MemoryManager::GetMemoryManager()
{
    thread_local MemoryManager instance;
    return instance;
}

void *MemoryManager::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers_.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBuffers_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

} // namespace TorchAtb