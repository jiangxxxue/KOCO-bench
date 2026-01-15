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
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/math/math.h>
#include "asdops/params/params.h"
#include "kernels/split/split_v/tiling/split_v_tiling.h"

namespace AsdOps {
// split
class SplitVKernel : public KernelBase {
public:
    explicit SplitVKernel(const std::string &kernelName, const BinHandle *handle) noexcept
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
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        int64_t sum = 0;
        for (int64_t element : param.splitSize) {
            MKI_CHECK((element >= 1), "the elements in splitSizes is less than 1", return false);
            sum += element;
        }
        MKI_CHECK((sum == dims[param.splitVDim[0]]),
            "The elements in SplitSize not sum to the size of dimension SplitVDim", return false);
        MKI_CHECK((param.splitVDim[0] < static_cast<int>(dims.size()) && param.splitVDim[0] >= 0),
            "SplitVDim Oversize.", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Split), "OpParam is invalid",
                     return 0);
        auto param = AnyCast<OpParam::Split>(launchParam.GetParam());
        return launchBufferSize_ + Utils::GetConstTensorSize<int32_t>(param.splitSize) +
               Utils::GetConstTensorSize<int32_t>(param.splitVDim);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        size_t outputNum = launchParam.GetOutTensorCount();
        if (outputNum == 2) { // splitV 2 Outputs
            auto status = SplitV2OutputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
            MKI_CHECK(status.Ok(), "SplitV tiling failed", return status);
            MKI_CHECK_NO_LOG(status.Ok(), return status);
        } else if (outputNum == 3) { // splitV 3 Outputs
            auto status = SplitV3OutputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
            MKI_CHECK(status.Ok(), "SplitV tiling failed", return status);
            MKI_CHECK_NO_LOG(status.Ok(), return status);
        };
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Split>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int32_t>(DIM_1, param.splitSize);
        kernelInfo_.AddConstTensorData<int32_t>(DIM_2, param.splitVDim);
        return Status::OkStatus();
    }
};

// SplitVF16Output2Kernel
class SplitVF16Output2Kernel : public SplitVKernel {
public:
    explicit SplitVF16Output2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitVKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitVKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                     launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                     "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitVF16Output2Kernel);

// SplitVF16Output3Kernel
class SplitVF16Output3Kernel : public SplitVKernel {
public:
    explicit SplitVF16Output3Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitVKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitVKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                     launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
                     "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitVF16Output3Kernel);

// SplitVInt64Output2Kernel
class SplitVInt64Output2Kernel : public SplitVKernel {
public:
    explicit SplitVInt64Output2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : SplitVKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(SplitVKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT64,
            "tensor dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(SplitVInt64Output2Kernel);
} // namespace AsdOps