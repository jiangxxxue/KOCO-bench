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
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/norm/layernorm/tiling/layernorm_tiling.h"

namespace {
static const float THRESHOLD = 2e-38;
} // namespace

namespace AsdOps {
using namespace Mki;
class LayernormKernel : public KernelBase {
public:
    explicit LayernormKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 3, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_LOG(DEBUG) << "norm type: " << type;
        MKI_CHECK(type == OpParam::Norm::LAYER_NORM, "layernorm: param type invalid", return false);
        MKI_CHECK(param.epsilon >= THRESHOLD, "epsilon invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND, "intensor0 format invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.format == TENSOR_FORMAT_ND, "intensor1 format invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(2).desc.format == TENSOR_FORMAT_ND, "intensor2 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.format == TENSOR_FORMAT_ND, "out0 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dims == launchParam.GetInTensor(0).desc.dims,
                  "outtensor0 shape not same as intensor0", return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
            "intensor1 dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(2).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
            "intensor2 dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == launchParam.GetInTensor(0).desc.dtype,
            "outtensor0 dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return LayerNormTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class LayernormF16Kernel : public LayernormKernel {
public:
    explicit LayernormF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LayernormKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LayernormKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(LayernormF16Kernel);

class LayernormBF16Kernel : public LayernormKernel {
public:
    explicit LayernormBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LayernormKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LayernormKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(LayernormBF16Kernel);

class LayernormF32Kernel : public LayernormKernel {
public:
    explicit LayernormF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : LayernormKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(LayernormKernel::CanSupport(launchParam), "failed to check support", return false);
        return true;
    }
};
REG_KERNEL_BASE(LayernormF32Kernel);
} // namespace AsdOps