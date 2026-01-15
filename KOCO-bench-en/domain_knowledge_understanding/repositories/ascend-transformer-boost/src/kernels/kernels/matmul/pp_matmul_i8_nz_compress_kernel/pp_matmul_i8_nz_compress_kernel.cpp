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
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "kernels/matmul/tiling/pp_matmul_i8_nz_tiling.h"
constexpr uint32_t ALIGN32 = 32;
constexpr uint32_t ALIGN16 = 16;

namespace AsdOps {
using namespace Mki;
class PpMatMulI8NzCompressKernel : public KernelBase {
public:
    explicit PpMatMulI8NzCompressKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul),
                     "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == DIM_5, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == DIM_1, "check outTensor count failed", return false);

        auto inTensor0 = launchParam.GetInTensor(DIM_0);
        auto inTensor1 = launchParam.GetInTensor(DIM_1);
        auto inTensor2 = launchParam.GetInTensor(DIM_2);
        auto inTensor3 = launchParam.GetInTensor(DIM_3);
        auto inTensor4 = launchParam.GetInTensor(DIM_4);
        auto outTensor = launchParam.GetOutTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_INT8, "inTensor0 dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "inTensor0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == DIM_4, "inTensor0 dims invalid", return false);
        MKI_CHECK((inTensor0.desc.dims.at(DIM_3) == ALIGN32) && ((inTensor0.desc.dims.at(DIM_2) % ALIGN16) == 0),
                     "inTensor0 dims value invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_INT8, "inTensor1 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "inTensor1 format invalid", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == DIM_4 || inTensor1.desc.dims.size() == DIM_2,
                     "inTensor1 dims invalid", return false);
        MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_INT32, "inTensor2 dtype invalid", return false);
        MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor2 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_UINT64 || inTensor3.desc.dtype == TENSOR_DTYPE_INT64,
                     "inTensor3 dtype invalid", return false);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "inTensor3 format invalid", return false);
        MKI_CHECK(inTensor4.desc.dtype == TENSOR_DTYPE_INT8, "inTensor4 dtype invalid", return false);
        MKI_CHECK(inTensor4.desc.format == TENSOR_FORMAT_ND, "inTensor4 format invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "outTensor format invalid", return false);
        MKI_CHECK(outTensor.desc.dims.size() == 4, "outTensor dims invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PpTilingData310P);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpTiling310P(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(PpMatMulI8NzCompressKernel);
} // namespace AsdOps