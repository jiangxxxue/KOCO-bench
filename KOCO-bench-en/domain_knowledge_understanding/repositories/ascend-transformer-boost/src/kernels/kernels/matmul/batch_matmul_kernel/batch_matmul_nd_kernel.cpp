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
#include "kernels/matmul/tiling/matmul_nd_tiling.h"
#include "kernels/matmul/tiling/matmul_nz_tiling.h"

namespace AsdOps {
using namespace Mki;
class BatchMatMulNdKernel : public KernelBase {
public:
    explicit BatchMatMulNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);

        auto batchMMNdInTensor0 = launchParam.GetInTensor(0);
        auto batchMMNdInTensor1 = launchParam.GetInTensor(1);
        auto batchMMNdOutTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(batchMMNdInTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(batchMMNdInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                  batchMMNdInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMNdInTensor0.desc.dims.size() == 3, "tensor dims invalid", return false);
        MKI_CHECK(batchMMNdInTensor1.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(batchMMNdInTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                  batchMMNdInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMNdInTensor1.desc.dims.size() == 3, "tensor dims invalid", return false);
        MKI_CHECK(batchMMNdOutTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(batchMMNdOutTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 ||
                  batchMMNdInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(batchMMNdOutTensor.desc.dims.size() == 3, "tensor dims invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul),
                     "check param type failed!", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return BatchMatMulNdTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class BatchMatMulNdF16Kernel : public BatchMatMulNdKernel {
public:
    explicit BatchMatMulNdF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNdKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNdKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && (!opParam.transposeB), "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNdF16Kernel);

class BatchMatMulNdF32Kernel : public BatchMatMulNdKernel {
public:
    explicit BatchMatMulNdF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNdKernel(kernelName, handle) {}
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNdKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && !opParam.transposeB, "transpose invalid", return false);

        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNdF32Kernel);

class BatchMatMulNdF16TbKernel : public BatchMatMulNdKernel {
public:
    explicit BatchMatMulNdF16TbKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BatchMatMulNdKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(BatchMatMulNdKernel::CanSupport(launchParam), return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && opParam.transposeB, "transpose invalid", return false);

        return true;
    }
};
REG_KERNEL_BASE(BatchMatMulNdF16TbKernel);
} // namespace AsdOps