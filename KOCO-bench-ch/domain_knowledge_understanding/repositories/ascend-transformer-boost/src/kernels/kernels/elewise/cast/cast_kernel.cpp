/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/kernel_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>

#include "asdops/params/params.h"
#include "kernels/elewise/tiling/elewise_tiling.h"

namespace AsdOps {
template <TensorDType IN_TYPE, TensorDType OUT_TYPE>
class CastKernel : public KernelBase {
public:
    explicit CastKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise), "elewise: param type invalid",
                  return false);

        Mki::PlatformType platformType = PlatformInfo::Instance().GetPlatformType();
        if constexpr (IN_TYPE == TENSOR_DTYPE_BF16 || OUT_TYPE == TENSOR_DTYPE_BF16) {
            MKI_CHECK(platformType == PlatformType::ASCEND_910B, "Only Ascend910B supports bfloat16 datatype.",
                      return false);
        }

        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_CAST, "cast: param type invalid", return false);
        TensorDType paramOutputDType = opParam.outTensorType;
        if (paramOutputDType != TENSOR_DTYPE_UNDEFINED) {
            MKI_CHECK(paramOutputDType == OUT_TYPE, "cast: param output dtype invalid", return false);
        }
        auto inTensor = launchParam.GetInTensor(0);
        auto outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor.desc.format == TENSOR_FORMAT_ND, "input format invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "output format invalid", return false);
        MKI_CHECK(inTensor.desc.dtype == IN_TYPE, "input dtype invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == OUT_TYPE, "output dtype invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return ElewiseCommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

using CastF16F32Kernel = CastKernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_FLOAT>;
using CastF32F16Kernel = CastKernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_FLOAT16>;
using CastI32I64Kernel = CastKernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT64>;
using CastI64I32Kernel = CastKernel<TENSOR_DTYPE_INT64, TENSOR_DTYPE_INT32>;
using CastF16I32Kernel = CastKernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT32>;
using CastI32F16Kernel = CastKernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_FLOAT16>;
using CastBF16F32Kernel = CastKernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_FLOAT>;
using CastF32BF16Kernel = CastKernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_BF16>;
using CastI32F32Kernel = CastKernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_FLOAT>;
using CastF32I32Kernel = CastKernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT32>;
using CastI8F16Kernel = CastKernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_FLOAT16>;
using CastF16I8Kernel = CastKernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT8>;

REG_KERNEL_BASE(CastF16F32Kernel);
REG_KERNEL_BASE(CastF32F16Kernel);
REG_KERNEL_BASE(CastI32I64Kernel);
REG_KERNEL_BASE(CastI64I32Kernel);
REG_KERNEL_BASE(CastF16I32Kernel);
REG_KERNEL_BASE(CastI32F16Kernel);
REG_KERNEL_BASE(CastBF16F32Kernel);
REG_KERNEL_BASE(CastF32BF16Kernel);
REG_KERNEL_BASE(CastI32F32Kernel);
REG_KERNEL_BASE(CastF32I32Kernel);
REG_KERNEL_BASE(CastI8F16Kernel);
REG_KERNEL_BASE(CastF16I8Kernel);
} // namespace AsdOps