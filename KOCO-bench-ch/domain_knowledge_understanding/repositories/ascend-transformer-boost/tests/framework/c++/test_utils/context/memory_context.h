/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_CONTEXT_CONTEXT_H
#define ATB_CONTEXT_CONTEXT_H
#include <cstdint>
#include <memory>
#include <vector>

namespace atb {
class BufferBase;

class MemoryContext {
public:
    MemoryContext();
    ~MemoryContext();
    void *GetHostTilingBuffer(uint64_t bufferSize);
    void *GetTilingBuffer(uint64_t bufferSize);
    void *GetWorkspaceBuffer(uint64_t bufferSize);
    void *GetIntermediateBuffer(uint64_t bufferSize);

private:
    uint64_t GetHostTilingBufferRing() const;
    uint64_t GetHostTilingBufferSize() const;

    uint64_t GetTilingBufferRing() const;
    uint64_t GetTilingBufferSize() const;

    uint64_t GetWorkspaceBufferRing() const;
    uint64_t GetWorkspaceBufferSize() const;

    uint64_t GetIntermediateBufferRing() const;
    uint64_t GetIntermediateBufferSize() const;

private:
    std::vector<std::unique_ptr<BufferBase>> hostTilingBuffers_;
    size_t hostTilingBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> tilingBuffers_;
    size_t tilingBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> workspaceBuffers_;
    size_t workspaceBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> intermediateBuffers_;
    size_t intermediateBufferOffset_ = 0;
};
} // namespace atb
#endif