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
class AddKernel : public KernelBase {
public:
    explicit AddKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_ADD, "add: param type invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BroadcastCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// AddI32
class AddI32Kernel : public AddKernel {
public:
    explicit AddI32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(AddI32Kernel);

// AddI64
class AddI64Kernel : public AddKernel {
public:
    explicit AddI64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(AddI64Kernel);

// AddF16
class AddF16Kernel : public AddKernel {
public:
    explicit AddF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(AddF16Kernel);

// AddF32
class AddF32Kernel : public AddKernel {
public:
    explicit AddF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(AddF32Kernel);

// AddBF16
class AddBF16Kernel : public AddKernel {
public:
    explicit AddBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AddKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AddKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "in tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "out tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(AddBF16Kernel);
} // namespace AsdOps