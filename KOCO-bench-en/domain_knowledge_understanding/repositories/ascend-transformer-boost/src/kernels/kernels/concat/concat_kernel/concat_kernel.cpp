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
#include "asdops/params/concat.h"
#include "kernels/concat/tiling/concat_tiling.h"

namespace AsdOps {
using namespace Mki;
class ConcatKernel : public KernelBase {
public:
    explicit ConcatKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Concat),
            "transpose: param type invalid", return false);
        OpParam::Concat param = AnyCast<OpParam::Concat>(launchParam.GetParam());
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        TensorDType dtype = launchParam.GetInTensor(0).desc.dtype;
        MKI_CHECK(dtype == TENSOR_DTYPE_FLOAT16 || dtype == TENSOR_DTYPE_FLOAT || dtype == TENSOR_DTYPE_BF16,
                  "input data type error", return false);
        SVector<int64_t> dims = launchParam.GetInTensor(0).desc.dims;
        if (param.concatDim < 0) {
            int realConcatDim = static_cast<int>(dims.size()) + param.concatDim;
            MKI_CHECK((realConcatDim >= 0), "Incorrect concatDim.", return false);
        } else {
            MKI_CHECK((param.concatDim < static_cast<int>(dims.size())), "ConcatDim Oversize.", return false);
        }

        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return Concat2InputsTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class ConcatF16Input2Kernel : public ConcatKernel {
public:
    explicit ConcatF16Input2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ConcatF16Input2Kernel);

class ConcatF32Input2Kernel : public ConcatKernel {
public:
    explicit ConcatF32Input2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : ConcatKernel(kernelName, handle)
    {
    }
};
REG_KERNEL_BASE(ConcatF32Input2Kernel);
} // namespace AsdOps