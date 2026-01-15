/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <mki/base/kernel_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki_loader/op_register.h>
#include "asdops/params/reverse.h"
#include "kernels/reverse/tiling/reverse_tiling.h"

namespace AsdOps {
constexpr uint64_t TENSOR_AXIS_IDX = 1;
class ReverseKernel : public KernelBase {
public:
    explicit ReverseKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Reverse), "reverse: param type invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Reverse), "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Reverse>(launchParam.GetParam());
        size_t constTensor0Size = Utils::RoundUp(Utils::GetConstTensorSize<int32_t>(param.axis));
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = ReverseTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Reverse>(launchParam.GetParam());
        auto ret = kernelInfo_.AddConstTensorData<int32_t>(TENSOR_AXIS_IDX, param.axis);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(1));

        return Status::OkStatus();
    }
};

// ReverseF16Kernel
class ReverseF16Kernel : public ReverseKernel {
public:
    explicit ReverseF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReverseKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReverseKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype unsupported",
                  return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(ReverseF16Kernel);

// ReverseBF16Kernel
class ReverseBF16Kernel : public ReverseKernel {
public:
    explicit ReverseBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReverseKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReverseKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(ReverseBF16Kernel);
 
// ReverseF32Kernel
class ReverseF32Kernel : public ReverseKernel {
public:
    explicit ReverseF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReverseKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(ReverseKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(ReverseF32Kernel);
} // namespace AsdOps