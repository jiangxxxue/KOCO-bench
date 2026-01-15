/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "host_tiling_buffer_pool.h"
#include "atb/utils/log.h"

namespace atb {
HostTilingBufferPool::HostTilingBufferPool(uint64_t blockNum, uint64_t blockSize)
    : TilingBufferPool(blockNum, blockSize)
{
}

HostTilingBufferPool::~HostTilingBufferPool() {}

uint8_t *HostTilingBufferPool::MallocTotalBuffer(uint64_t bufferSize)
{
    ATB_LOG(INFO) << "malloc bufferSize:" << bufferSize;
    return static_cast<uint8_t *>(malloc(bufferSize));
}

void HostTilingBufferPool::FreeTotalBuffer(uint8_t *buffer)
{
    free(buffer);
    buffer = nullptr;
}

bool HostTilingBufferPool::IsDeviceBufferPool()
{
    return false;
}
} // namespace atb