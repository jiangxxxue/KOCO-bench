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
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include "asdops/params/params.h"
#include "tiling/copy_tiling.h"

namespace AsdOps {
constexpr uint64_t TENSOR_DST_SIZE_IDX = 1;
constexpr uint64_t TENSOR_DST_STRIDE_IDX = 2;
constexpr uint64_t TENSOR_DST_OFFSET_IDX = 3;
constexpr uint64_t TENSOR_SRC_SIZE_IDX = 5;
constexpr uint64_t TENSOR_SRC_STRIDE_IDX = 6;
constexpr uint64_t TENSOR_SRC_OFFSET_IDX = 7;
// CopyKernel
class CopyKernel : public KernelBase {
public:
    explicit CopyKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Copy), "check param type failed!",
                     return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed!", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed!", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == launchParam.GetInTensor(1).desc.dtype,
                     "check dtype failed", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Copy), "OpParam is invalid",
                     return 0);
        auto param = AnyCast<OpParam::Copy>(launchParam.GetParam());
        return launchBufferSize_ + Utils::GetConstTensorSize<int64_t>(param.dstSize) +
               Utils::GetConstTensorSize<int64_t>(param.dstStride) +
               Utils::GetConstTensorSize<int64_t>(param.dstOffset) +
               Utils::GetConstTensorSize<int64_t>(param.dstSize) +
               Utils::GetConstTensorSize<int64_t>(param.dstStride) +
               Utils::GetConstTensorSize<int64_t>(param.dstOffset);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        /* Attention: now, src size/stride/offset is useless in tiling, placeholder by dst. */
        auto status = ViewCopyTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Copy>(launchParam.GetParam());
        // copy dstSize
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_DST_SIZE_IDX, param.dstSize);
        // copy dstStride
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_DST_STRIDE_IDX, param.dstStride);
        // copy dstOffset
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_DST_OFFSET_IDX, param.dstOffset);
        // copy srcSize
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_SRC_SIZE_IDX, param.dstSize);
        // copy srcStride
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_SRC_STRIDE_IDX, param.dstStride);
        // copy srcOffset
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_SRC_OFFSET_IDX, param.dstOffset);
        return Status::OkStatus();
    }
};

// ViewCopyInt64Kernel
class ViewCopyInt64Kernel : public CopyKernel {
public:
    explicit ViewCopyInt64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CopyKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "inTensor dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "outTensor dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(ViewCopyInt64Kernel);

// ViewCopyInt32Kernel
class ViewCopyInt32Kernel : public CopyKernel {
public:
    explicit ViewCopyInt32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CopyKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "inTensor dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT32,
            "outTensor dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(ViewCopyInt32Kernel);

// ViewCopyF32Kernel
class ViewCopyF32Kernel : public CopyKernel {
public:
    explicit ViewCopyF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CopyKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "inTensor dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT,
            "outTensor dtype invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(ViewCopyF32Kernel);

// ViewCopyF16Kernel
class ViewCopyF16Kernel : public CopyKernel {
public:
    explicit ViewCopyF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CopyKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16,
            "inTensor dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(ViewCopyF16Kernel);
 
// ViewCopyBF16Kernel
class ViewCopyBF16Kernel : public CopyKernel {
public:
    explicit ViewCopyBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CopyKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CopyKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "inTensor dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "outTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(ViewCopyBF16Kernel);
} // namespace AsdOps