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
#include <mki/utils/log/log.h>
#include <mki_loader/op_register.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/matmul_nd_tiling.h"

namespace AsdOps {
using namespace Mki;
class MatMulNdKernel : public KernelBase {
public:
    explicit MatMulNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);
        auto mmInTensor0 = launchParam.GetInTensor(0);
        auto mmInTensor1 = launchParam.GetInTensor(1);
        auto mmOutTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(mmInTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(mmInTensor1.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(mmOutTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return MatMulNdTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

class MatMulNdF16Kernel : public MatMulNdKernel {
public:
    explicit MatMulNdF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MatMulNdKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(MatMulNdKernel::CanSupport(launchParam), return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == 2, "inTensor0 dims invalid", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == 2, "inTensor1 dims invalid", return false);
        MKI_CHECK(outTensor.desc.dims.size() == 2, "outTensor dims invalid", return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && (!opParam.transposeB), "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(MatMulNdF16Kernel);

class MatMulNdF32Kernel : public MatMulNdKernel {
public:
    explicit MatMulNdF32Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MatMulNdKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(MatMulNdKernel::CanSupport(launchParam), return false);
        auto mmInTensor0 = launchParam.GetInTensor(0);
        auto mmInTensor1 = launchParam.GetInTensor(1);
        auto mmOutTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(mmInTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(mmInTensor1.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK(mmOutTensor.desc.dtype == TENSOR_DTYPE_FLOAT, "tensor dtype invalid", return false);
        MKI_CHECK((mmInTensor0.desc.dims.size() == 2 || mmInTensor0.desc.dims.size() == 3), "inTensor0 dims invalid",
                   return false);
        MKI_CHECK((mmInTensor1.desc.dims.size() == 2 || mmInTensor1.desc.dims.size() == 3), "inTensor1 dims invalid",
                   return false);
        MKI_CHECK((mmOutTensor.desc.dims.size() == 2 || mmOutTensor.desc.dims.size() == 3), "outTensor dims invalid",
                   return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && (!opParam.transposeB), "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(MatMulNdF32Kernel);

class MatMulNdF16TbKernel : public MatMulNdKernel {
public:
    explicit MatMulNdF16TbKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : MatMulNdKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(MatMulNdKernel::CanSupport(launchParam), return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);
        MKI_CHECK((inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3), "inTensor0 dims invalid",
                  return false);
        MKI_CHECK((inTensor1.desc.dims.size() == 2 || inTensor1.desc.dims.size() == 3), "inTensor1 dims invalid",
                  return false);
        MKI_CHECK((outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3), "outTensor dims invalid",
                  return false);
        auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && opParam.transposeB, "transpose invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(MatMulNdF16TbKernel);
} // namespace AsdOps
