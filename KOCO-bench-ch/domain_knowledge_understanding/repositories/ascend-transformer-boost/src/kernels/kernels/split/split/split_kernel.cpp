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
#include "asdops/params/params.h"
#include "kernels/split/split/tiling/split_tiling.h"

namespace AsdOps {
// split
class SplitKernel : public KernelBase {
public:
    explicit SplitKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Split),
            "split: param type invalid", return false);
        OpParam::Split param = AnyCast<OpParam::Split>(launchParam.GetParam());
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == static_cast<size_t>(param.splitNum),
            "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        size_t outputNum = launchParam.GetOutTensorCount();
        if (outputNum == 2) { // split 2 Outputs
            return Split2OutputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        } else if (outputNum == 3) { // split 3 Outputs
            return Split3OutputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        }
        return Status::FailStatus(1);
    }
};

// SplitF16Output2Kernel
class SplitF16Output2Kernel : public SplitKernel {
public:
    explicit SplitF16Output2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                  launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitF16Output2Kernel);

// SplitF16Output3Kernel
class SplitF16Output3Kernel : public SplitKernel {
public:
    explicit SplitF16Output3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                  launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                  "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitF16Output3Kernel);

// SplitInt64Output2Kernel
class SplitInt64Output2Kernel : public SplitKernel {
public:
    explicit SplitInt64Output2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
                  "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitInt64Output2Kernel);
} // namespace AsdOps
