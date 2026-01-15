/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_KERNEL_IMPLEMENT_H
#define ATB_KERNEL_IMPLEMENT_H

#include <memory>
#include <mki/operation.h>
#include <mki/kernel.h>
#include <mki/op_desc.h>
#include "atb/runner/atb_kernel_method.h"

namespace atb {
class MkiNodeImplement : public AtbKernelMethod {
public:
    MkiNodeImplement() = delete;
    ~MkiNodeImplement() override;
    MkiNodeImplement(Mki::Operation *op, MkiInferShapePreFunc func);
    std::string GetName() const override;
    void Reset() override;
    bool BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors, SVector<ViewFunc> &inTensorViewFuncs,
                          const Mki::OpDesc &opDesc, size_t outTensorNum) override;
    Status BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors, const SVector<Mki::Tensor *> &outTensors,
                            const Mki::OpDesc &opDesc) override;
    bool PlanKernelInferShape() override;
    size_t GetTilingSize() const override;
    bool UpdateBestKernel() override;
    int64_t GetWorkspaceSize() const override;
    Status InitKernelInfo(uint8_t *hostTilingBuffer, uint64_t tilingSize, bool launchWithTiling) override;
    void SetWorkspaceDeviceAddr(uint8_t *deviceWorkspaceBuffer) override;
    void SetTilingDeviceAddr(uint8_t *deviceTilingBuffer) override;
    Status Run(aclrtStream stream) override;
    bool GetCachedTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *kernelHostTilingBuffer,
                         uint64_t maxTilingSize, uint64_t &tilingSizeFetched, bool launchWithTiling) override;
    void AddTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *hostTilingBuffer,
                   size_t tilingSize) const override;
    void SetArgsDeviceBuffer(void *deviceBuffer) override;
    void SetArgsHostBuffer(void *hostBuffer) override;
    void *GetArgsDeviceBuffer() override;
    void *GetArgsHostBuffer() override;
    Status BuildArgs() override;
    uint64_t GetArgsSize() override;

    // utils
    Mki::SVector<Mki::Tensor> &GetInTensors() override;
    Mki::SVector<Mki::Tensor> &GetOutTensors() override;
    void SaveLaunchParam(aclrtStream stream, const std::string &dirPath) const override;
    void *GetMsprofInfoKey() const override;
    void GetReportTensors(Mki::SVector<std::pair<bool, Mki::Tensor>> &allTensors) const override;
    uint32_t GetOpType() const override;
    uint32_t GetBlockDim() const override;
    void ResetLogPrefix(const std::string &prefix, size_t kernelId) override;
    bool GetTilingFilledFlag() const override;
    void SetTilingFilledFlag(bool flag) override;

private:
    bool OperationGetBestKernel();
    std::string GetLogPrefix() const;

private:
    Mki::Operation *operation_ = nullptr;
    std::shared_ptr<Mki::Kernel> kernel_ = nullptr;
    Mki::LaunchParam launchParam_;
    Mki::RunInfo runInfo_;
    MkiInferShapePreFunc mkiInferShapePreFunc_ = nullptr;
    std::string logPrefix_;
    bool kernelCacheValid_ = false;   // kernel cache是否命中
    bool tilingBufferFilled_ = false; // tiling 是否已从缓存中读取
    void *argsDeviceBuffer_ = nullptr;
    void *argsHostBuffer_ = nullptr;
};

static const std::unordered_map<const Mki::ErrorType, atb::ErrorType> InitAtbMkiErrorHash() noexcept
{
    return {{Mki::ErrorType::NO_ERROR, atb::ErrorType::NO_ERROR},
            {Mki::ErrorType::ERROR_INVALID_VALUE, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_OPERATION_NOT_EXIST, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_TACTIC_NOT_EXIST, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_KERNEL_NOT_EXIST, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_ATTR_NOT_EXIST, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_ATTR_INVALID_TYPE, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_LAUNCH_KERNEL_ERROR, atb::ErrorType::ERROR_RT_FAIL},
            {Mki::ErrorType::ERROR_SYNC_STREAM_ERROR, atb::ErrorType::ERROR_RT_FAIL},
            {Mki::ErrorType::ERROR_INFERSHAPE_ERROR, atb::ErrorType::ERROR_RT_FAIL},
            {Mki::ErrorType::ERROR_NOT_CONSISTANT, atb::ErrorType::ERROR_INVALID_PARAM},
            {Mki::ErrorType::ERROR_ALLOC_HOST, atb::ErrorType::ERROR_OUT_OF_HOST_MEMORY},
            {Mki::ErrorType::ERROR_MEMERY_COPY_ERROR, atb::ErrorType::ERROR_COPY_HOST_MEMORY_FAIL},
            {Mki::ErrorType::ERROR_RUN_TIME_ERROR, atb::ErrorType::ERROR_RT_FAIL}};
}

static const std::unordered_map<const Mki::ErrorType, atb::ErrorType> ATB_MKI_ERROR_HASH = InitAtbMkiErrorHash();
} // namespace atb
#endif