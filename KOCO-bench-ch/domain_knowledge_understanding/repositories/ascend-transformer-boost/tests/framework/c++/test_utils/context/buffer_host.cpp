/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "buffer_host.h"
#include <atb/utils/log.h>

namespace atb {
BufferHost::BufferHost(uint64_t bufferSize)
{
    ATB_LOG(INFO) << "BufferHost::BufferHost called, bufferSize:" << bufferSize;
    if (bufferSize > 0) {
        buffer_.resize(bufferSize);
    }
}

BufferHost::~BufferHost() {}

void *BufferHost::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= buffer_.size()) {
        ATB_LOG(INFO) << "BufferHost::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << buffer_.size() <<
            ", not new host mem";
        return buffer_.data();
    }

    ATB_LOG(INFO) << "BufferHost::GetBuffer success, bufferSize:" << bufferSize;
    buffer_.resize(bufferSize);
    ATB_LOG(INFO) << "BufferHost::GetBuffer success, buffer:" << buffer_.data();
    return buffer_.data();
}
} // namespace atb