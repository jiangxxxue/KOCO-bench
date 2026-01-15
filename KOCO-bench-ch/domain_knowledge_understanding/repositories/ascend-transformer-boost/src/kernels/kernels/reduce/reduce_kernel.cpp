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
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/tensor_utils.h>
#include "asdops/params/reduce.h"
#include "kernels/reduce/tiling/reduce_tiling.h"

namespace AsdOps {
constexpr uint64_t TENSOR_REDUCE_AXIS_IDX = 1;
constexpr uint64_t TENSOR_REDUCE_WORKSPACE_IDX = 4;

class ReduceKernel : public KernelBase {
public:
    explicit ReduceKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Reduce),
            "reduce max: param type invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        auto param = AnyCast<OpParam::Reduce>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int64_t>(param.axis);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Reduce),
            "reduce max: param type invalid", return Status::FailStatus(1));

        auto status = ReduceCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Reduce>(launchParam.GetParam());
        if (PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B) {
            kernelInfo_.SetHwsyncIdx(0);
            kernelInfo_.AddConstTensorData<int64_t>(TENSOR_REDUCE_AXIS_IDX + 1, param.axis);
            kernelInfo_.SetMemsetInfo(TENSOR_REDUCE_WORKSPACE_IDX + 1, kernelInfo_.GetScratchSizes().at(1));
        } else {
            kernelInfo_.AddConstTensorData<int64_t>(TENSOR_REDUCE_AXIS_IDX, param.axis);
            kernelInfo_.SetMemsetInfo(TENSOR_REDUCE_WORKSPACE_IDX, kernelInfo_.GetScratchSizes().at(1));
        }
        return Status::OkStatus();
    }
};

// MaxInt32
class ReduceMaxInt32Kernel : public ReduceKernel {
public:
    explicit ReduceMaxInt32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceMaxInt32Kernel);

// MaxBF16
class ReduceMaxBF16Kernel : public ReduceKernel {
public:
    explicit ReduceMaxBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceMaxBF16Kernel);

// MinInt32
class ReduceMinInt32Kernel : public ReduceKernel {
public:
    explicit ReduceMinInt32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceMinInt32Kernel);

// MinBF16
class ReduceMinBF16Kernel : public ReduceKernel {
public:
    explicit ReduceMinBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceMinBF16Kernel);

// SumF16
class ReduceSumF16Kernel : public ReduceKernel {
public:
    explicit ReduceSumF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceSumF16Kernel);

// SumBF16
class ReduceSumBF16Kernel : public ReduceKernel {
public:
    explicit ReduceSumBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ReduceKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ReduceSumBF16Kernel);
} // namespace AsdOps
