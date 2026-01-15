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
#include "tiling/rms_norm_grad_tiling.h"
#include "tiling/tiling_data.h"

namespace {
constexpr size_t RMSNORM_BACKWARD_IN_COUNT = 4;
constexpr size_t RMSNORM_BACKWARD_OUT_COUNT = 2;
constexpr size_t DY_INPUT = 0;
constexpr size_t X_INPUT = 1;
constexpr size_t RSTD_INPUT = 2;
constexpr size_t GAMMA_INPUT = 3;
} // namespace

namespace AsdOps {
using namespace Mki;
class RmsNormBackwardKernel : public KernelBase {
public:
    explicit RmsNormBackwardKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormGradTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RmsNormGradTiling(launchParam, kernelInfo_);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::RMS_NORM_BACKWARD,
                     "rmsnormbackward: param type invalid", return false);
        TensorDType dtypeDy = launchParam.GetInTensor(DY_INPUT).desc.dtype;
        TensorDType dtypeX = launchParam.GetInTensor(X_INPUT).desc.dtype;
        TensorDType dtypeRstd = launchParam.GetInTensor(RSTD_INPUT).desc.dtype;
        TensorDType dtypeGamma = launchParam.GetInTensor(GAMMA_INPUT).desc.dtype;
        TensorFormat formatDy = launchParam.GetInTensor(DY_INPUT).desc.format;
        TensorFormat formatX = launchParam.GetInTensor(X_INPUT).desc.format;
        TensorFormat formatRstd = launchParam.GetInTensor(RSTD_INPUT).desc.format;
        TensorFormat formatGamma = launchParam.GetInTensor(GAMMA_INPUT).desc.format;
        MKI_CHECK((dtypeDy == TENSOR_DTYPE_FLOAT16 || dtypeDy == TENSOR_DTYPE_BF16 ||
            dtypeDy == TENSOR_DTYPE_FLOAT) && dtypeDy == dtypeX && dtypeDy == dtypeGamma,
            "input dtype invalid", return false);
        MKI_CHECK((dtypeRstd == TENSOR_DTYPE_FLOAT),
                     "input rstd dtype invalid", return false);
        MKI_CHECK(formatDy == TENSOR_FORMAT_ND && formatDy == formatX &&
                     formatDy == formatGamma && formatDy == formatRstd,
                     "input format invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(RmsNormBackwardKernel);
}  // namespace AsdOps