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
#include "asdops/params/cumsum.h"
#include "kernels/cumsum/tiling/cumsum_tiling.h"

namespace AsdOps {
using namespace Mki;
constexpr uint64_t TENSOR_CUMSUM_AXIS_IDX = 1;
constexpr uint64_t TENSOR_CUMSUM_OUTPUT_IDX = 2;
// Cumsum
class CumsumKernel : public KernelBase {
public:
    explicit CumsumKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cumsum),
                     "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check ouTensor count failed", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Cumsum), "OpParam is invalid",
                     return 0);
        auto param = AnyCast<OpParam::Cumsum>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int64_t>(param.axis);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = CumsumTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        kernelInfo_.SetMemsetInfo(TENSOR_CUMSUM_OUTPUT_IDX, launchParam.GetOutTensor(0).dataSize);
        auto param = AnyCast<OpParam::Cumsum>(launchParam.GetParam());
        kernelInfo_.AddConstTensorData<int64_t>(TENSOR_CUMSUM_AXIS_IDX, param.axis);
        return Status::OkStatus();
    }
};

// CumsumF16Kernel
class CumsumF16Kernel : public CumsumKernel {
public:
    explicit CumsumF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CumsumKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CumsumKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(CumsumF16Kernel);

class CumsumBF16Kernel : public CumsumKernel {
public:
    explicit CumsumBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CumsumKernel(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(CumsumKernel::CanSupport(launchParam), return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "inTensor dtype invalid",
                     return false);
        return true;
    }
};
REG_KERNEL_BASE(CumsumBF16Kernel);
 

// CumsumF16DtmKernel
class CumsumF16DtmKernel : public CumsumF16Kernel {
public:
    explicit CumsumF16DtmKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CumsumF16Kernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(CumsumF16DtmKernel);
 
// CumsumBF16DtmKernel
class CumsumBF16DtmKernel : public CumsumBF16Kernel {
public:
    explicit CumsumBF16DtmKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : CumsumBF16Kernel(kernelName, handle)
        {
        }
};
REG_KERNEL_BASE(CumsumBF16DtmKernel);
} // namespace AsdOps