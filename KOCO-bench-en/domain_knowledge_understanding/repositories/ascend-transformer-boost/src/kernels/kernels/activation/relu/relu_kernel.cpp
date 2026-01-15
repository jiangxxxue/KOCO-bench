/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/launch_param.h>
#include <mki/base/kernel_base.h>
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/activation/tiling/activation_tiling.h"

namespace AsdOps {
using namespace Mki;
// Relu
class ReluKernel : public KernelBase {
public:
    explicit ReluKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation),
            "relu: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Activation>(launchParam.GetParam());
        OpParam::Activation::ActivationType type = opParam.activationType;
        MKI_CHECK(type == OpParam::Activation::ACTIVATION_RELU, "activation type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ActivationCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// ReluF32
class ReluF32Kernel : public ReluKernel {
public:
    explicit ReluF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReluKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReluKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "in tensor 0 dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "out tensor 0 dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(ReluF32Kernel);
 
// ReluBF16
class ReluBF16Kernel : public ReluKernel {
public:
    explicit ReluBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReluKernel(kernelName, handle)
        {
        }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReluKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor 0 dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "out tensor 0 dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(ReluBF16Kernel);
} // namespace AsdOps