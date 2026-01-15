/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/elewise/tiling/elewise_tiling.h"

namespace AsdOps {
using namespace Mki;
class LogicalOrKernel : public KernelBase {
public:
    explicit LogicalOrKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_LOGICAL_OR, "logicor: param type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BroadcastCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// LogicalOrInt8Kernel
class LogicalOrInt8Kernel : public LogicalOrKernel {
public:
    explicit LogicalOrInt8Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LogicalOrKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LogicalOrKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(LogicalOrInt8Kernel);

} // namespace AsdOps
