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
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include "asdops/params/softmax.h"
#include "kernels/softmax/tiling/softmax_tiling.h"

namespace AsdOps {
// Softmax
class SoftmaxKernel : public KernelBase {
public:
    explicit SoftmaxKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Softmax),
            "softmax: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return SoftmaxCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// SoftmaxF16Kernel
class SoftmaxF16Kernel : public SoftmaxKernel {
public:
    explicit SoftmaxF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SoftmaxKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(SoftmaxF16Kernel);

// SoftmaxF32Kernel
class SoftmaxF32Kernel : public SoftmaxKernel {
public:
    explicit SoftmaxF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SoftmaxKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(SoftmaxF32Kernel);

// SoftmaxBF16Kernel
class SoftmaxBF16Kernel : public SoftmaxKernel {
public:
    explicit SoftmaxBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SoftmaxKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(SoftmaxBF16Kernel);
} // namespace AsdOps
