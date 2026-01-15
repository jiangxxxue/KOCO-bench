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
class LessKernel : public KernelBase {
public:
    explicit LessKernel(const std::string &kernelName, const BinHandle *handle) noexcept
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
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_LESS, "less: param type invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "output dtype invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BroadcastCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// LessInt64Kernel
class LessInt64Kernel : public LessKernel {
public:
    explicit LessInt64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LessKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LessKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(LessInt64Kernel);

// LessF16Kernel
class LessF16Kernel : public LessKernel {
public:
    explicit LessF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LessKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LessKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(LessF16Kernel);

// LessF32Kernel
class LessF32Kernel : public LessKernel {
public:
    explicit LessF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LessKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LessKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(LessF32Kernel);

// LessBF16Kernel
class LessBF16Kernel : public LessKernel {
public:
    explicit LessBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LessKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LessKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(LessBF16Kernel);
} // namespace AsdOps
