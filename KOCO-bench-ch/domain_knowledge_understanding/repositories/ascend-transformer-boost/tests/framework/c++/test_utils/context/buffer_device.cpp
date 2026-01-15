/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "buffer_device.h"
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include <mki/utils/time/timer.h>
#include "atb/utils/statistic.h"

namespace atb {
BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ATB_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    if (bufferSize_ > 0) {
        ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc bufferSize:" << bufferSize_;
        int st = aclrtMalloc(static_cast<void **>(&buffer_), bufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (st != ACL_RT_SUCCESS) {
            ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc fail, ret:" << st;
        }
    }
}

BufferDevice::~BufferDevice()
{
    Free();
}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << bufferSize_ <<
            ", not new device mem";
        return buffer_;
    }

    if (buffer_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer aclrtFree buffer:" << buffer_ << ", bufferSize:" <<
            bufferSize_;
        Free();
    }

    ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc bufferSize:" << bufferSize;
    int st = aclrtMalloc(static_cast<void **>(&buffer_), bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (st != ACL_RT_SUCCESS) {
        ATB_LOG(ERROR) << "BufferDevice::GetBuffer aclrtMalloc fail, ret:" << st;
        return nullptr;
    }
    ATB_LOG(INFO) << "BufferDevice::GetBuffer aclrtMalloc success, buffer:" << buffer_;
    bufferSize_ = bufferSize;
    return buffer_;
}

void BufferDevice::Free()
{
    if (buffer_) {
        aclrtFree(buffer_);
        buffer_ = nullptr;
        bufferSize_ = 0;
    }
}
} // namespace atb