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
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/matmul_nz_tiling.h"

namespace AsdOps {
class BatchMatMulNzKernel : public KernelBase {
public:
    explicit BatchMatMulNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);
        auto batchMMInTensor0 = launchParam.GetInTensor(0);
        auto batchMMInTensor1 = launchParam.GetInTensor(1);
        auto batchMMIOutTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(batchMMInTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
        MKI_CHECK(batchMMInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMInTensor0.desc.dims.size() == 4, "tensor dims invalid", return false);
        MKI_CHECK(batchMMInTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
        MKI_CHECK(batchMMInTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMInTensor1.desc.dims.size() == 4, "tensor dims invalid", return false);
        MKI_CHECK(batchMMIOutTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
        MKI_CHECK(batchMMIOutTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMIOutTensor.desc.dims.size() == 4, "tensor dims invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul),
                     "check param type failed!", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BatchMatMulNzTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class BatchMatMulNzF16TAKernel : public BatchMatMulNzKernel {
public:
    explicit BatchMatMulNzF16TAKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNzKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNzKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK(opParam.transposeA && (!opParam.transposeB), "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNzF16TAKernel);

class BatchMatMulNzF16TATBKernel : public BatchMatMulNzKernel {
public:
    explicit BatchMatMulNzF16TATBKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNzKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNzKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK(opParam.transposeA && opParam.transposeB, "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNzF16TATBKernel);

class BatchMatMulNzF16Kernel : public BatchMatMulNzKernel {
public:
    explicit BatchMatMulNzF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNzKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNzKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && (!opParam.transposeB), "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNzF16Kernel);

class BatchMatMulNzF16TBKernel : public BatchMatMulNzKernel {
public:
    explicit BatchMatMulNzF16TBKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNzKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNzKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && opParam.transposeB, "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNzF16TBKernel);
} // namespace AsdOps