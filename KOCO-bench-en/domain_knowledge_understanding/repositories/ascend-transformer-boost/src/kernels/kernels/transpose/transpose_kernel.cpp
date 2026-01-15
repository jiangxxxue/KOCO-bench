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
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/math.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki_loader/op_register.h>

#include "asdops/params/params.h"
#include "kernels/transpose/tiling/transpose_tiling.h"

namespace AsdOps {
constexpr uint64_t TENSOR_PERM_IDX = 1;
static constexpr size_t BYTES_64BITS = 8;
static constexpr size_t BYTES_32BITS = 4;
static constexpr size_t BYTES_16BITS = 2;
static constexpr size_t BYTES_8BITS = 1;

template <size_t ELEM_SIZE>
class TransposeKernel : public KernelBase {
public:
    explicit TransposeKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Transpose), "transpose: param type invalid",
                  return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        size_t elemSize = GetTensorElementSize(launchParam.GetInTensor(0).desc.dtype);
        MKI_CHECK(elemSize == ELEM_SIZE, "tensor element size not supported", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Transpose), "transpose: param type invalid",
                  return 0);
        auto param = AnyCast<OpParam::Transpose>(launchParam.GetParam());
        size_t constTensor0Size = Utils::GetConstTensorSize<int32_t>(param.perm);
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = TransposeTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Transpose>(launchParam.GetParam());
        auto ret = kernelInfo_.AddConstTensorData<int32_t>(TENSOR_PERM_IDX, param.perm);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(1));

        return Status::OkStatus();
    }
};
using Transpose8Kernel = TransposeKernel<BYTES_8BITS>;
REG_KERNEL_BASE(Transpose8Kernel);
using Transpose16Kernel = TransposeKernel<BYTES_16BITS>;
REG_KERNEL_BASE(Transpose16Kernel);
using Transpose32Kernel = TransposeKernel<BYTES_32BITS>;
REG_KERNEL_BASE(Transpose32Kernel);
using Transpose64Kernel = TransposeKernel<BYTES_64BITS>;
REG_KERNEL_BASE(Transpose64Kernel);
} // namespace AsdOps