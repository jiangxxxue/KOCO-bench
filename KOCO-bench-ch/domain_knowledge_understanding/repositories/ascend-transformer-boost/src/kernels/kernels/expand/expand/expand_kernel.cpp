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
#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include "asdops/params/expand.h"
#include "kernels/expand/tiling/expand_tiling.h"

namespace AsdOps {
using namespace Mki;
constexpr uint64_t TENSOR_EXPAND_SHAPE_IDX = 1;

class ExpandKernel : public KernelBase {
public:
    explicit ExpandKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Expand),
                  "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "check inTensor count failed",
                  return false); // Expand has 1 inputs
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed",
                  return false); // Expand has 1 output
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Expand),
                  "OpParam is invalid", return 0);
        auto param = AnyCast<OpParam::Expand>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int64_t>(param.shape);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = ExpandTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Expand>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_EXPAND_SHAPE_IDX, param.shape);
        return Status::OkStatus();
    }
};

// ExpandF16Kernel
class ExpandF16Kernel : public ExpandKernel {
public:
    explicit ExpandF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ExpandKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(ExpandKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor dtype invalid",
                     return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(ExpandF16Kernel);

// ExpandBF16Kernel
class ExpandBF16Kernel : public ExpandKernel {
public:
    explicit ExpandBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ExpandKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(ExpandKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "inTensor dtype invalid",
                     return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "outTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(ExpandBF16Kernel);
} // namespace AsdOps