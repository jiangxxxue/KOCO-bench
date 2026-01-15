/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <acl/acl.h>
#include "atb/context/allocator/default_host_allocator.h"
#include "atb/context/allocator/default_device_allocator.h"

using namespace atb;
TEST(TestDefaultAllocator, TestAllocateAndDeAllocate)
{
    /*
        测试场景：使用DefualtAllocator管理device和host侧地址
        结果：申请成功，释放成功
    */

    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    DefaultDeviceAllocator *deviceAllocator = new DefaultDeviceAllocator();
    DefaultHostAllocator *hostAllocator = new DefaultHostAllocator();

    // 通过Allocator创建device和host buffer
    uint32_t bufferSize = 1022;
    void *deviceBuffer = deviceAllocator->Allocate(bufferSize);
    void *hostBuffer = hostAllocator->Allocate(bufferSize);
    ASSERT_NE(deviceBuffer, nullptr);
    ASSERT_NE(hostBuffer, nullptr);

    // 释放通过Allocator创建的device和host buffer
    Status st = deviceAllocator->Deallocate(deviceBuffer);
    ASSERT_EQ(st, atb::NO_ERROR);
    st = hostAllocator->Deallocate(hostBuffer);
    ASSERT_EQ(st, atb::NO_ERROR);
    delete deviceAllocator;
    delete hostAllocator;
}

TEST(TestDefaultAllocator, TestDeAllocateError)
{
    /*
        测试场景：使用DefualtAllocator释放非自己创建的device和host侧地址
        结果：释放失败
    */

    uint32_t deviceId = 1;
    aclrtSetDevice(deviceId);
    DefaultDeviceAllocator *deviceAllocator = new DefaultDeviceAllocator();
    DefaultHostAllocator *hostAllocator = new DefaultHostAllocator();
    // 使用acl接口创建device和host buffer
    void *deviceBufferOut = nullptr;
    void *hostBufferOut = nullptr;
    uint32_t bufferSize = 1022;
    Status st = aclrtMalloc(&deviceBufferOut, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(st, 0);
    st = aclrtMallocHost(&hostBufferOut, bufferSize);
    ASSERT_EQ(st, 0);

    // 进行释放，释放结果不成功
    st = deviceAllocator->Deallocate(deviceBufferOut);
    ASSERT_NE(st, atb::NO_ERROR);
    st = hostAllocator->Deallocate(hostBufferOut);
    ASSERT_NE(st, atb::NO_ERROR);
    delete deviceAllocator;
    delete hostAllocator;
}