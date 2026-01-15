/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_CONTEXT_BASE_H
#define ATB_CONTEXT_BASE_H
#include <memory>
#include "atb/context.h"
#include "atb/svector.h"
#include "atb/context/allocator/allocator.h"
#include "atb/context/tiling_buffer_pool/tiling_buffer_pool.h"
#include "atb/runner/runner_type.h"
#include "atb/context/runner_pool.h"

namespace atb {
class ContextBase : public Context {
public:
    explicit ContextBase();
    ~ContextBase() override;
    ContextBase(const ContextBase &other) = delete;
    ContextBase &operator=(const ContextBase &other) = delete;
    Status Init(const std::function<void*(size_t)>& alloc = nullptr, const std::function<void(void*)>& dealloc = nullptr);
    void Destroy();
    Status SetExecuteStream(aclrtStream stream) override;
    aclrtStream GetExecuteStream() const override;
    Status SetAsyncTilingCopyStatus(bool status) override;
    bool GetAsyncTilingCopyStatus() const override;
    Status SetExecuteStreams(const std::vector<aclrtStream> &streams) override;
    std::vector<aclrtStream> GetExecuteStreams() override;

    aclrtStream GetAsyncTilingCopyStream() const;
    aclrtEvent GetAsyncTilingCopyEvent();

    uint8_t *GetHostTilingBuffer();
    uint8_t *GetDeviceTilingBuffer();
    uint64_t GetTilingBufferBlockSize() const;
    RunnerPool &GetRunnerPool(RunnerType runnerType);
    const Tensor &GetOverflowKernelOutTensor();
    Status SetExecuteType(ExecuteType type) override;
    ExecuteType GetExecuteType() override;
    Status SetLaunchMode(LaunchMode mode) override;
    LaunchMode GetLaunchMode() override;
    void *GetArgsDeviceBuffer(size_t bufferSize);
    void *GetArgsHostBuffer(size_t bufferSize);
    Status FreeArgsDeviceBuffer(void *addr);
    Status FreeArgsHostBuffer(void *addr);
    bool GetLaunchWithTilingStatus() const;

private:
    Status CreateCopyStreamAndEvents();
    Status DestoryCopyStreamAndEvents();
    Status CreateOverflowOutTensor();
    Status FreeOverflowTensor();

private:
    std::vector<aclrtStream> executeStreams_;

    aclrtStream asyncTilingCopyStream_ = nullptr;
    SVector<aclrtEvent> asyncTilingCopyEvents_;
    size_t asyncTilingCopyEventsIndex_ = 0;

    std::unique_ptr<TilingBufferPool> hostTilingBufferPool_;
    std::unique_ptr<TilingBufferPool> deviceTilingBufferPool_;
    std::vector<RunnerPool> runnerPools_;
    Tensor overflowOutTensor_;
    static thread_local ExecuteType executeType_;
    LaunchMode mode_ = KERNEL_LAUNCH_MODE;
    std::unique_ptr<Allocator> deviceAllocator_;  // 一开始就赋值为defaultDeviceAllocator
    std::unique_ptr<Allocator> hostAllocator_;  // 一开始就赋值为defaultHostAllocator
    std::function<void*(size_t size)> allocateFunc_;  // 默认使用defaultDeviceAllocator中的Allocate方法
    std::function<void(void*)> deallocateFunc_;       // 默认使用defaultDeviceAllocator中的Deallocate方法
};
} // namespace atb
#endif