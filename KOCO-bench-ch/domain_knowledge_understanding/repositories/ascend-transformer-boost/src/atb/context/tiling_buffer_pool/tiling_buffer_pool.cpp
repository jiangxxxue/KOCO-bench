/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/context/tiling_buffer_pool/tiling_buffer_pool.h"
#include "atb/utils/log.h"

namespace atb {
TilingBufferPool::TilingBufferPool(uint64_t blockNum, uint64_t blockSize) : blockNum_(blockNum), blockSize_(blockSize)
{
}

TilingBufferPool::~TilingBufferPool() {}

Status TilingBufferPool::Init()
{
    blockIndex_ = 0;

    if (totalBuffer_) {
        FreeTotalBuffer(totalBuffer_);
        totalBuffer_ = nullptr;
    }

    uint64_t totalSize = blockNum_ * blockSize_;
    ATB_LOG(INFO) << "TilingBufferPool malloc buffer, blockNum:" << blockNum_ << ", blockSize:" << blockSize_
                  << ", totalSize:" << totalSize;
    totalBuffer_ = MallocTotalBuffer(totalSize);
    if (totalBuffer_) {
        totalSize_ = totalSize;
        return NO_ERROR;
    } else {
        ATB_LOG(ERROR) << "malloc total buffer fail";
        return IsDeviceBufferPool() ? ERROR_OUT_OF_DEVICE_MEMORY : ERROR_OUT_OF_HOST_MEMORY;
    }
}

void TilingBufferPool::Destroy()
{
    if (totalBuffer_ != nullptr) {
        FreeTotalBuffer(totalBuffer_);
        totalBuffer_ = nullptr;
    }
    blockIndex_ = 0;
}

uint8_t *TilingBufferPool::GetBuffer()
{
    uint8_t *nextBuffer = totalBuffer_ + blockSize_ * blockIndex_;

    blockIndex_++;
    if (blockIndex_ == blockNum_) {
        blockIndex_ = 0;
    }

    return nextBuffer;
}

uint64_t TilingBufferPool::GetBlockNum() const
{
    return blockNum_;
}

uint64_t TilingBufferPool::GetBlockSize() const
{
    return blockSize_;
}
} // namespace atb