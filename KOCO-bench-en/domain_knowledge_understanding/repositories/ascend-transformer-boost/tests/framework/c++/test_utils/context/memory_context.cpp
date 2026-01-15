/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "memory_context.h"
#include <atb/utils/log.h>
#include "atb/utils/config.h"
#include "buffer_device.h"
#include "buffer_host.h"

namespace atb {
MemoryContext::MemoryContext()
{
    uint64_t bufferRing = GetHostTilingBufferRing();
    uint64_t bufferSize = GetHostTilingBufferSize();
    ATB_LOG(FATAL) << "MemoryContext hosttiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    hostTilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        hostTilingBuffers_.at(i).reset(new BufferHost(bufferSize));
    }

    bufferRing = GetTilingBufferRing();
    bufferSize = GetTilingBufferSize();
    ATB_LOG(FATAL) << "MemoryContext tiling bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    tilingBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        tilingBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetWorkspaceBufferRing();
    bufferSize = GetWorkspaceBufferSize();
    ATB_LOG(FATAL) << "MemoryContext workspace bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    workspaceBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        workspaceBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }

    bufferRing = GetIntermediateBufferRing();
    bufferSize = GetIntermediateBufferSize();
    ATB_LOG(FATAL) << "MemoryContext intermediate bufferRing:" << bufferRing << ", bufferSize:" << bufferSize;
    intermediateBuffers_.resize(bufferRing);
    for (size_t i = 0; i < bufferRing; ++i) {
        intermediateBuffers_.at(i).reset(new BufferDevice(bufferSize));
    }
}

MemoryContext::~MemoryContext() {}

void *MemoryContext::GetHostTilingBuffer(uint64_t bufferSize)
{
    if (hostTilingBufferOffset_ == hostTilingBuffers_.size()) {
        hostTilingBufferOffset_ = 0;
    }
    return hostTilingBuffers_.at(hostTilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *MemoryContext::GetTilingBuffer(uint64_t bufferSize)
{
    if (tilingBufferOffset_ == tilingBuffers_.size()) {
        tilingBufferOffset_ = 0;
    }
    return tilingBuffers_.at(tilingBufferOffset_++)->GetBuffer(bufferSize);
}

void *MemoryContext::GetWorkspaceBuffer(uint64_t bufferSize)
{
    if (workspaceBufferOffset_ == workspaceBuffers_.size()) {
        workspaceBufferOffset_ = 0;
    }
    return workspaceBuffers_.at(workspaceBufferOffset_++)->GetBuffer(bufferSize);
}

void *MemoryContext::GetIntermediateBuffer(uint64_t bufferSize)
{
    if (intermediateBufferOffset_ == intermediateBuffers_.size()) {
        intermediateBufferOffset_ = 0;
    }
    return intermediateBuffers_.at(intermediateBufferOffset_++)->GetBuffer(bufferSize);
}

uint64_t MemoryContext::GetHostTilingBufferRing() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_HOSTTILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetHostTilingBufferSize() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_HOSTTILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetTilingBufferRing() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_TILING_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetTilingBufferSize() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_TILING_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetWorkspaceBufferRing() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetWorkspaceBufferSize() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_WORKSPACE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetIntermediateBufferRing() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_INTERMEDIATE_RING");
    if (envStr == nullptr) {
        return 1;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}

uint64_t MemoryContext::GetIntermediateBufferSize() const
{
    const char *envStr = std::getenv("ATB_CONTEXT_INTERMEDIATE_SIZE");
    if (envStr == nullptr) {
        return 0;
    }
    return strtoll(envStr, nullptr, 10); // 10:Decimal
}
} // namespace atb