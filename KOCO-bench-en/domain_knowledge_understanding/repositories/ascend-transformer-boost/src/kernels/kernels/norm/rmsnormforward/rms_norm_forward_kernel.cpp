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
#include "tiling/rms_norm_forward_tiling.h"
#include "tiling/tiling_data.h"

namespace {
constexpr size_t RMSNORMFORWARD_TENSOR_IN_COUNT = 2;
constexpr size_t RMSNORMFORWARD_TENSOR_OUT_COUNT = 2;
} // namespace

namespace AsdOps {
using namespace Mki;
class RmsNormForwardKernel : public KernelBase {
public:
    explicit RmsNormForwardKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormForwardTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RmsNormForwardTiling(launchParam, kernelInfo_);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::RMS_NORM_FORWARD,
                  "rmsnormforward: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == RMSNORMFORWARD_TENSOR_IN_COUNT,
                  "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == RMSNORMFORWARD_TENSOR_OUT_COUNT,
                  "output num invalid", return false);
        TensorDType dtype0 = launchParam.GetInTensor(0).desc.dtype;
        TensorDType dtype1 = launchParam.GetInTensor(1).desc.dtype;
        TensorFormat format0 = launchParam.GetInTensor(0).desc.format;
        TensorFormat format1 = launchParam.GetInTensor(1).desc.format;
        MKI_CHECK((dtype0 == TENSOR_DTYPE_FLOAT16 || dtype0 == TENSOR_DTYPE_BF16 ||
            dtype0 == TENSOR_DTYPE_FLOAT) && dtype1 == dtype0, "input dtype invalid", return false);
        MKI_CHECK(format0 == TENSOR_FORMAT_ND && format1 == format0, "input format invalid",
            return false);
        return true;
    }
};
REG_KERNEL_BASE(RmsNormForwardKernel);
} // namespace AsdOps