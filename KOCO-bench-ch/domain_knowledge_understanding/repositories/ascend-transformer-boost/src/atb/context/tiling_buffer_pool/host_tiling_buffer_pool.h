/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_HOST_TILING_BUFFER_POOL_H
#define ATB_HOST_TILING_BUFFER_POOL_H
#include "atb/context/tiling_buffer_pool/tiling_buffer_pool.h"

namespace atb {
class HostTilingBufferPool : public TilingBufferPool {
public:
    HostTilingBufferPool(uint64_t blockNum, uint64_t blockSize);
    ~HostTilingBufferPool() override;

protected:
    uint8_t *MallocTotalBuffer(uint64_t bufferSize) override;
    void FreeTotalBuffer(uint8_t *buffer) override;
    bool IsDeviceBufferPool() override;
};
} // namespace atb
#endif