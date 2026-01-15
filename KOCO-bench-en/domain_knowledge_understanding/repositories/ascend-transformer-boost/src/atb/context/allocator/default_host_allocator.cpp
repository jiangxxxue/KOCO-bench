/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "default_host_allocator.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
namespace atb {
const int ALIGN_INT = 32;
DefaultHostAllocator::DefaultHostAllocator() {}
DefaultHostAllocator::~DefaultHostAllocator()
{
    // 释放所有管理的host侧地址
    for (auto it = memMap.begin(); it != memMap.end(); ++it) {
        Status st = aclrtFreeHost(it->first);
        if (st != 0) {
            ATB_LOG(ERROR) << "aclrtFree device buffer failed!";
        }
        currentAllocateSize_ -= it->second;
#ifdef _DEBUG
        ATB_LOG(INFO) << "DefaultHostAllocator::~DefaultHostAllocator aclrtFreeHost free host buffer: " << it->first
                      << ", which the host bufferSize is " << it->second << ", currentAllocateSize_: "
                      << currentAllocateSize_;
#else
        ATB_LOG(INFO) << "DefaultHostAllocator::~DefaultHostAllocator aclrtFreeHost free host bufferSize: " << it->second
                      << ", and currentAllocateSize_: " << currentAllocateSize_;
#endif
    }
}

void *DefaultHostAllocator::Allocate(size_t bufferSize)
{
    if (bufferSize == 0) {
        ATB_LOG(WARN) << "bufferSize can not be 0, please check the bufferSize";
        return nullptr;
    }
    void *addr = nullptr;
    bufferSize = static_cast<size_t>(TensorUtil::AlignInt(bufferSize, ALIGN_INT));
    ATB_LOG(INFO) << "bufferSize should be 32-bit alignment, automate align upwards to " << bufferSize;
    // aclrtMallocHost会自动对于bufferSize+32，不论bufferSize是否是32的整数倍
    Status st = aclrtMallocHost(&addr, bufferSize);
    if (st != 0) {
        ATB_LOG(ERROR) << "aclrtMallocHost host buffer failed!";
        return nullptr;
    }
    currentAllocateSize_ += bufferSize;
    memMap.insert(std::make_pair(addr, bufferSize));
#ifdef _DEBUG
    ATB_LOG(INFO) << "DefaultHostAllocator::Allocate host buffer success, host buffer is " << addr
                  << ", which bufferSize is " << bufferSize << " and the currentAllocateSize_: " << currentAllocateSize_;
#else
    ATB_LOG(INFO) << "DefaultHostAllocator::Allocate host buffer success, bufferSize is " << bufferSize << " currentAllocateSize_: "
                  << currentAllocateSize_;
#endif
    return addr;
}

Status DefaultHostAllocator::Deallocate(void *addr)
{
    if (addr == nullptr) {
        ATB_LOG(INFO) << "the addr is nullptr, do not need to deallocate";
        return NO_ERROR;
    }
    auto it = memMap.find(addr);
    if (it == memMap.end()) {
        ATB_LOG(ERROR) << "free fail, can not find the address please check the address is made by allocator";
        return ERROR_CANN_ERROR;
    }
    // Free
    Status st = aclrtFreeHost(addr);
    if (st != 0) {
        ATB_LOG(ERROR) << "aclrtFreeHost host buffer failed!";
        return ERROR_CANN_ERROR;
    }
    currentAllocateSize_ -= it->second;
#ifdef _DEBUG
    ATB_LOG(INFO) << "DefaultHostAllocator::Deallocate host buffer success, free host buffer: " << addr
                  << ", which bufferSize is "<< it->second << ", currentAllocateSize_: " << currentAllocateSize_;
#else
    ATB_LOG(INFO) << "DefaultHostAllocator::Deallocate host buffer success, free bufferSize: "<< it->second
                    << ", currentAllocateSize_: " << currentAllocateSize_;
#endif
    memMap.erase(addr);
    return NO_ERROR;
}
} // namespace atb