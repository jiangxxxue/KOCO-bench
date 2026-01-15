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
#include <stdexcept>
#include <acl/acl_rt.h>
#include <atb/utils/log.h>
#include "utils.h"

namespace TorchAtb {
constexpr uint64_t KB = 1024;
constexpr uint64_t DIM_NUM_2 = 2;

BufferDevice::BufferDevice(uint64_t bufferSize)
{
    ATB_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    if (bufferSize > 0) {
        CreateTorchTensorWithSize(bufferSize);
    }
}

BufferDevice::~BufferDevice()
{
    Free();
}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << bufferSize_
                      << ", not new device mem";
        return buffer_;
    }
    CreateTorchTensorWithSize(bufferSize);
    return buffer_;
}

void BufferDevice::CreateTorchTensorWithSize(const uint64_t bufferSize)
{
    ATB_LOG(INFO) << "CreateTorchTensorWithSize bufferSize:" << bufferSize;
    atb::TensorDesc tensorDesc;
    tensorDesc.dtype = ACL_UINT8;
    tensorDesc.format = ACL_FORMAT_ND;
    tensorDesc.shape.dimNum = 1;
    tensorDesc.shape.dims[0] = (bufferSize + KB - 1) / KB * KB; // 1024对齐
    torchTensor_ = Utils::CreateTorchTensorFromTensorDesc(tensorDesc);
    buffer_ = torchTensor_.data_ptr();
    if (buffer_ == nullptr) {
        ATB_LOG(ERROR) << "malloc buffer with torch failed.";
        bufferSize_ = 0;
    }
    bufferSize_ = static_cast<uint64_t>(tensorDesc.shape.dims[0]);
    ATB_LOG(INFO) << "new bufferSize:" << bufferSize_;
}

void BufferDevice::Free()
{
    if (buffer_) {
        torchTensor_.reset();
        buffer_ = nullptr;
        bufferSize_ = 0;
        ATB_LOG(DEBUG) << "Buffer freed";
    }
}
} // namespace TorchAtb