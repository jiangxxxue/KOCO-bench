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
#include "kernels/matmul/tiling/pp_matmul_nz_tiling.h"

namespace AsdOps {
using namespace Mki;
class PpMatMulNzKernel : public KernelBase {
public:
    explicit PpMatMulNzKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "inTensor0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor0 dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4, "inTensor0 dims invalid",
                     return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "inTensor1 format invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor1 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4, "inTensor1 dims invalid",
                     return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "outTensor format invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid", return false);
        MKI_CHECK(outTensor.desc.dims.size() == 3 || inTensor0.desc.dims.size() == 4, "outTensor dims invalid",
                     return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul),
                     "check param type failed!", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PpTilingDataNz);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpTilingNz(launchParam, kernelInfo_);
    }
};

class PpMatMulNzF16Kernel : public PpMatMulNzKernel {
public:
    explicit PpMatMulNzF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulNzKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK_NO_LOG(PpMatMulNzKernel::CanSupport(launchParam), return false);
        return true;
    }
};
REG_KERNEL_BASE(PpMatMulNzF16Kernel);
} // namespace AsdOps