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
#include "kernels/activation/tiling/activation_tiling.h"

namespace AsdOps {
using namespace Mki;
class GeluKernel : public KernelBase {
public:
    explicit GeluKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation),
            "gelu: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Activation>(launchParam.GetParam());
        OpParam::Activation::ActivationType type = opParam.activationType;
        MKI_CHECK(type == OpParam::Activation::ACTIVATION_GELU, "activation type invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == launchParam.GetOutTensor(0).desc.dtype,
            "activation type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ActivationCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// GeluF32
class GeluF32Kernel : public GeluKernel {
public:
    explicit GeluF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluF32Kernel);

// GeluF16
class GeluF16Kernel : public GeluKernel {
public:
    explicit GeluF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluF16Kernel);

// GeluBF16
class GeluBF16Kernel : public GeluKernel {
public:
    explicit GeluBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluBF16Kernel);

// GeluApproxF32
class GeluApproxF32Kernel : public GeluKernel {
public:
    explicit GeluApproxF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluApproxF32Kernel);

// GeluApproxF16
class GeluApproxF16Kernel : public GeluKernel {
public:
    explicit GeluApproxF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluApproxF16Kernel);

// GeluApproxBF16
class GeluApproxBF16Kernel : public GeluKernel {
public:
    explicit GeluApproxBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : GeluKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(GeluApproxBF16Kernel);
} // namespace AsdOps