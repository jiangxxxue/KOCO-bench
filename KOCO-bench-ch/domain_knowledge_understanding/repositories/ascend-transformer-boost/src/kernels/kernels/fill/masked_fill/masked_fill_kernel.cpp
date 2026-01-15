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
constexpr uint64_t TENSOR_MASKFILL_VALUE_IDX = 2;
// MaskedFillKernel
template <typename T>
class MaskedFillKernel : public KernelBase {
public:
    explicit MaskedFillKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Fill), "check param type failed!",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed",
                  return false); // MaskedFill has 2 inputs
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed",
                  return false); // MaskedFill has 1 output
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        MKI_CHECK(param.withMask, "mask param invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Fill), "OpParam is invalid",
                  return 0);
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<float, T>(param.value);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status status = MaskedFillTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
        // copy value
        kernelInfo_.AddConstTensorData<float, T>(TENSOR_MASKFILL_VALUE_IDX, param.value);
        return Status::OkStatus();
    }
};

// MaskedFillF16Kernel
class MaskedFillF16Kernel : public MaskedFillKernel<fp16_t> {
public:
    explicit MaskedFillF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MaskedFillKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(MaskedFillKernel::CanSupport(launchParam), "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor0 dtype invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT8 ||
                      launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_BOOL,
                  "inTensor1 dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(MaskedFillF16Kernel);

// MaskedFillBF16Kernel
class MaskedFillBF16Kernel : public MaskedFillKernel<bf16_t> {
public:
    explicit MaskedFillBF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MaskedFillKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(MaskedFillKernel::CanSupport(launchParam), "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "inTensor0 dtype invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT8 ||
                      launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_BOOL,
                  "inTensor1 dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_BF16, "outTensor dtype invalid",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(MaskedFillBF16Kernel);

class MaskedFillInt32Kernel : public MaskedFillKernel<int32_t> {
public:
    explicit MaskedFillInt32Kernel(const std::string &kernelName, const BinHandle *handle)
        : MaskedFillKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(MaskedFillKernel::CanSupport(launchParam), "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_INT32, "inTensor0 dtype invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_INT8 ||
                      launchParam.GetInTensor(1).desc.dtype == TENSOR_DTYPE_BOOL,
                  "inTensor1 dtype invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT32, "outTensor dtype invalid",
                  return false);
        return true;
    }
};
REG_KERNEL_BASE(MaskedFillInt32Kernel);
} // namespace AsdOps