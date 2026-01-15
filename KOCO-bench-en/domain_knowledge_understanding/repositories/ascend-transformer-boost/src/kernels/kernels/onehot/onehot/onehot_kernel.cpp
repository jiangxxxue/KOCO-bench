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
#include "asdops/params/onehot.h"
#include "kernels/onehot/tiling/onehot_tiling.h"

namespace AsdOps {
using namespace Mki;
constexpr uint64_t TENSOR_AXIS_IDX = 1;
class OnehotKernel : public KernelBase {
public:
    explicit OnehotKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Onehot),
            "onehot: param type invalid", return false);
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorFormat format0 = launchParam.GetInTensor(0).desc.format;
        TensorFormat format1 = launchParam.GetInTensor(1).desc.format;
        TensorFormat format2 = launchParam.GetInTensor(2).desc.format;
        MKI_CHECK(dtype0 == TENSOR_DTYPE_INT32 || dtype0 == TENSOR_DTYPE_INT64, "input0 dtype invalid", return false);
        MKI_CHECK(format0 == TENSOR_FORMAT_ND && format1 == format0 && format2 == format0,
                    "input format invalid", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Onehot),
            "OpParam is invalid", return 0);
        auto &param = AnyCast<OpParam::Onehot>(launchParam.GetParam());
        size_t constTensor0Size = Utils::RoundUp(Utils::GetConstTensorSize<int64_t>(param.depth));
        return launchBufferSize_ + constTensor0Size;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        auto status = OnehotTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
        MKI_CHECK_NO_LOG(status.Ok(), return status);

        kernelInfo_.SetConstTensorOffset(launchBufferSize_);

        auto &param = AnyCast<OpParam::Onehot>(launchParam.GetParam());
        auto ret = kernelInfo_.AddConstTensorData<int64_t>(TENSOR_AXIS_IDX, param.depth);
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(1));
        return Status::OkStatus();
    }
};

// OnehotInt64Kernel
class OnehotInt64Kernel : public OnehotKernel {
public:
    explicit OnehotInt64Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : OnehotKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(OnehotKernel::CanSupport(launchParam), "failed to check support", return false);
        auto inTensor2 = launchParam.GetInTensor(1).desc;
        auto inTensor3 = launchParam.GetInTensor(2).desc;
        MKI_CHECK(inTensor2.dtype == TENSOR_DTYPE_INT64,
            "InTensor(2) dtype unsupported TENSOR_DTYPE_INT64", return false);
        MKI_CHECK(inTensor3.dtype == TENSOR_DTYPE_INT64,
            "InTensor(3) dtype unsupported TENSOR_DTYPE_INT64", return false);
        return true;
    }
};
REG_KERNEL_BASE(OnehotInt64Kernel);

// OnehotInt32Kernel
class OnehotInt32Kernel : public OnehotKernel {
public:
    explicit OnehotInt32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : OnehotKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(OnehotKernel::CanSupport(launchParam), "failed to check support", return false);
        auto inTensor2 = launchParam.GetInTensor(1).desc;
        auto inTensor3 = launchParam.GetInTensor(2).desc;
        MKI_CHECK(inTensor2.dtype == TENSOR_DTYPE_INT32,
            "InTensor(2) dtype unsupported", return false);
        MKI_CHECK(inTensor3.dtype == TENSOR_DTYPE_INT32,
            "InTensor(3) dtype unsupported", return false);
        return true;
    }
};
REG_KERNEL_BASE(OnehotInt32Kernel);
} // namespace AsdOps