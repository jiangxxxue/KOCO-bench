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
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki_loader/op_register.h>
#include "asdops/params/asstrided.h"
#include "kernels/asstrided/tiling/asstrided_tiling.h"

namespace AsdOps {
using namespace Mki;
constexpr uint64_t TENSOR_SIZE_IDX = 1;
constexpr uint64_t TENSOR_STRIDE_IDX = 2;
constexpr uint64_t TENSOR_OFFSET_IDX = 3;
// AsStridedKernel
class AsStridedKernel : public KernelBase {
public:
    explicit AsStridedKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::AsStrided), "asstrided: param type invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::AsStrided), "asstrided: param type invalid",
                  return 0);
        auto param = AnyCast<OpParam::AsStrided>(launchParam.GetParam());
        size_t constTensor0Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.size));
        size_t constTensor1Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.stride));
        size_t constTensor2Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.offset));
        return launchBufferSize_ + constTensor0Size + constTensor1Size + constTensor2Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = AsStridedTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::AsStrided>(launchParam.GetParam());
        auto ret = kernelInfo_.AddConstTensorData<int64_t>(TENSOR_SIZE_IDX, param.size);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(ERROR_INVALID_VALUE));
        ret = kernelInfo_.AddConstTensorData<int64_t>(TENSOR_STRIDE_IDX, param.stride);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(ERROR_INVALID_VALUE));
        ret = kernelInfo_.AddConstTensorData<int64_t>(TENSOR_OFFSET_IDX, param.offset);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(ERROR_INVALID_VALUE));

        return Status::OkStatus();
    }
};

// AsStridedF16Int64Kernel
class AsStridedF16Int64Kernel : public AsStridedKernel {
public:
    explicit AsStridedF16Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AsStridedKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AsStridedKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(AsStridedF16Int64Kernel);

class AsStridedF32Int64Kernel : public AsStridedKernel {
public:
    explicit AsStridedF32Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AsStridedKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AsStridedKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(AsStridedF32Int64Kernel);

// AsStridedInt64Int64Kernel
class AsStridedInt64Int64Kernel : public AsStridedKernel {
public:
    explicit AsStridedInt64Int64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : AsStridedKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(AsStridedKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64, "tensor dtype unsupported",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(AsStridedInt64Int64Kernel);
} // namespace AsdOps