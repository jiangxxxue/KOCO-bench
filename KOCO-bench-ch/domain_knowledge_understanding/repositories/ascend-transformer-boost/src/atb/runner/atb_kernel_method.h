/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_IMPLEMENT_H
#define ATB_IMPLEMENT_H
#include <mki/utils/any/any.h>
#include <mki/op_desc.h>
#include <mki/launch_param.h>
#include "atb/types.h"
#include "atb/svector.h"
#include "atb/kernel_cache/kernel_cache.h"

namespace atb {
enum TensorType : int {
    UNDEFINED_TENSOR = 0,
    IN_TENSOR,
    OUT_TENSOR,
    INTERMEDIATE_TENSOR,
};

using ViewFunc = std::function<void(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)>;
using AsdOpsInferShapePreFunc = std::function<void(Mki::LaunchParam &)>;
using MkiInferShapePreFunc = std::function<void(Mki::LaunchParam &)>;

class AtbKernelMethod {
public:
    AtbKernelMethod() = default;
    virtual ~AtbKernelMethod() = default;
    virtual std::string GetName() const = 0;
    virtual void Reset() = 0;
    virtual bool BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors, SVector<ViewFunc> &inTensorViewFuncs,
                                  const Mki::OpDesc &opDesc, size_t outTensorNum) = 0;
    virtual Status BuildLaunchParam(const SVector<Mki::Tensor *> &inTensors, const SVector<Mki::Tensor *> &outTensors,
                                    const Mki::OpDesc &opDesc) = 0;
    virtual bool PlanKernelInferShape() = 0;
    virtual size_t GetTilingSize() const = 0;
    virtual bool UpdateBestKernel() = 0;
    virtual int64_t GetWorkspaceSize() const = 0;
    virtual Status InitKernelInfo(uint8_t *hostTilingBuffer, uint64_t tilingSize, bool launchWithTiling) = 0;
    virtual void SetWorkspaceDeviceAddr(uint8_t *deviceWorkspaceBuffer) = 0;
    virtual void SetTilingDeviceAddr(uint8_t *deviceTilingBuffer) = 0;
    virtual Status Run(aclrtStream stream) = 0;
    virtual bool GetCachedTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *kernelHostTilingBuffer,
                                 uint64_t maxTilingSize, uint64_t &tilingSizeFetched, bool launchWithTiling) = 0;
    virtual void AddTiling(KernelCache &kernelCache, size_t kernelIndex, uint8_t *hostTilingBuffer,
                           size_t tilingSize) const = 0;
    virtual void SetArgsDeviceBuffer(void *deviceBuffer) = 0;
    virtual void SetArgsHostBuffer(void *hostBuffer) = 0;
    virtual void *GetArgsDeviceBuffer() = 0;
    virtual void *GetArgsHostBuffer() = 0;
    virtual uint64_t GetArgsSize() = 0;
    virtual Status BuildArgs() = 0;

    // utils
    virtual Mki::SVector<Mki::Tensor> &GetInTensors() = 0;
    virtual Mki::SVector<Mki::Tensor> &GetOutTensors() = 0;
    virtual void SaveLaunchParam(aclrtStream stream, const std::string &dirPath) const = 0;
    virtual void *GetMsprofInfoKey() const = 0;
    virtual void GetReportTensors(Mki::SVector<std::pair<bool, Mki::Tensor>> &allTensors) const = 0;
    virtual uint32_t GetOpType() const = 0;
    virtual uint32_t GetBlockDim() const = 0;
    virtual void ResetLogPrefix(const std::string &prefix, size_t kernelId) = 0;
    virtual bool GetTilingFilledFlag() const = 0;
    virtual void SetTilingFilledFlag(bool flag) = 0;
};
} // namespace atb
#endif