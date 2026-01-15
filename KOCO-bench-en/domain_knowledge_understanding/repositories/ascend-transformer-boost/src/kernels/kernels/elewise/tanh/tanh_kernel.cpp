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
class TanhKernel : public KernelBase {
public:
    explicit TanhKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_TANH, "tanh: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == launchParam.GetOutTensor(0).desc.dtype,
            "elementwise type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ElewiseCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// TanhF16
class TanhF16Kernel : public TanhKernel {
public:
    explicit TanhF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : TanhKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(TanhF16Kernel);

// TanhBF16
class TanhBF16Kernel : public TanhKernel {
public:
    explicit TanhBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : TanhKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(TanhBF16Kernel);

} // namespace AsdOps