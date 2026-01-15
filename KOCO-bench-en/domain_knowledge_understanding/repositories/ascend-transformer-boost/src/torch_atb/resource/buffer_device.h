/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_BUFFER_DEVICE_H
#define TORCH_BUFFER_DEVICE_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include <cstdint>

namespace TorchAtb {
class BufferDevice {
public:
    explicit BufferDevice(uint64_t bufferSize);
    ~BufferDevice();
    void *GetBuffer(uint64_t bufferSize);

private:
    void Free();
    void CreateTorchTensorWithSize(const uint64_t bufferSize);

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
    torch::Tensor torchTensor_;
};
} // namespace TorchAtb
#endif