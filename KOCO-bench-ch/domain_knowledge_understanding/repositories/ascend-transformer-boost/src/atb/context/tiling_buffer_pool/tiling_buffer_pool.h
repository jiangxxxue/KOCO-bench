/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_TILING_BUFFER_POOL_H
#define ATB_TILING_BUFFER_POOL_H
#include <cstdint>
#include <atb/types.h>

namespace atb {
class TilingBufferPool {
public:
    TilingBufferPool(uint64_t blockNum, uint64_t blockSize);
    virtual ~TilingBufferPool();
    TilingBufferPool(const TilingBufferPool &other) = delete;
    TilingBufferPool &operator=(const TilingBufferPool &other) = delete;
    Status Init();
    void Destroy();
    uint8_t *GetBuffer();
    uint64_t GetBlockNum() const;
    uint64_t GetBlockSize() const;

protected:
    virtual uint8_t *MallocTotalBuffer(uint64_t bufferSize) = 0;
    virtual void FreeTotalBuffer(uint8_t *buffer) = 0;
    virtual bool IsDeviceBufferPool() = 0;

private:
    uint64_t blockNum_ = 0;
    uint64_t blockSize_ = 0;
    uint8_t *totalBuffer_ = nullptr;
    uint64_t totalSize_ = 0;
    uint64_t blockIndex_ = 0;
};
} // namespace atb
#endif