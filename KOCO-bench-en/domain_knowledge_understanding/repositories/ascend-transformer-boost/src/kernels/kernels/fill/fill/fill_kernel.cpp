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
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/bf16/bf16_t.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include "asdops/params/params.h"
#include "kernels/fill/tiling/fill_tiling.h"

namespace AsdOps {
using namespace Mki;
constexpr uint64_t TENSOR_FILL_OUTDIM_IDX = 0;
constexpr uint64_t TENSOR_FILL_VALUE_IDX = 1;
// FillKernel
template <typename T>
class FillKernel : public KernelBase {
public:
    explicit FillKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Fill), "check param type failed!",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 0, "check inTensor count failed",
                  return false); // Fill has 0 inputs
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed",
                  return false); // Fill has 1 output
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Fill),
            "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int64_t>(param.outDim);
        size_t constTensor1Size = Utils::GetConstTensorSize<float, T>(param.value);
        return launchBufferSize_ + constTensor0Size + constTensor1Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status status = FillTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK(status.Ok(), "failed to check support", return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        // copy outDim
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_FILL_OUTDIM_IDX, param.outDim);
        // copy value
        kernelInfo_.AddConstTensorData<float, T>(TENSOR_FILL_VALUE_IDX, param.value);
        return Status::OkStatus();
    }
};

class FillF16Kernel : public FillKernel<fp16_t> {
public:
    explicit FillF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FillKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(FillKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid",
            return false);
        return true;
    }
};
REG_KERNEL_BASE(FillF16Kernel);

class FillBF16Kernel : public FillKernel<bf16_t> {
public:
    explicit FillBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : FillKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(FillKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "outTensor dtype invalid",
            return false);
        return true;
    }
};
REG_KERNEL_BASE(FillBF16Kernel);
} // namespace AsdOps