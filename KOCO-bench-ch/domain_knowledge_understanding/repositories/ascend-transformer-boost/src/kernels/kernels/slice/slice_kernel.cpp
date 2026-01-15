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
#include "asdops/params/slice.h"
#include "kernels/slice/tiling/slice_tiling.h"

namespace AsdOps {
constexpr uint64_t TENSOR_OFFSET_IDX = 1;
constexpr uint64_t TENSOR_SIZE_IDX = 2;
// Slice
class SliceKernel : public KernelBase {
public:
    explicit SliceKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Slice),
            "slice: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Slice),
            "transpose: param type invalid", return 0);
        auto param = AnyCast<OpParam::Slice>(launchParam.GetParam());
        size_t constTensor0Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.offsets));
        size_t constTensor1Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.size));
        return launchBufferSize_ + constTensor0Size + constTensor1Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = SliceTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Slice>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_OFFSET_IDX, param.offsets);
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_SIZE_IDX, param.size);

        return Status::OkStatus();
    }
};

// SliceF16Int64Kernel
class SliceF16Int64Kernel : public SliceKernel {
public:
    explicit SliceF16Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SliceKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SliceKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                     launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                     "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SliceF16Int64Kernel);

// SliceInt8Int64Kernel
class SliceInt8Int64Kernel : public SliceKernel {
public:
    explicit SliceInt8Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SliceKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SliceKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT8 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_UINT8 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BOOL, "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SliceInt8Int64Kernel);

// SliceInt32Int64Kernel
class SliceInt32Int64Kernel : public SliceKernel {
public:
    explicit SliceInt32Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SliceKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SliceKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_UINT32 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SliceInt32Int64Kernel);
} // namespace AsdOps