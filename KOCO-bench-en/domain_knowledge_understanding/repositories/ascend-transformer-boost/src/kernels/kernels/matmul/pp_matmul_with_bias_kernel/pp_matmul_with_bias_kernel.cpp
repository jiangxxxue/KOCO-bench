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
#include <mki/utils/platform/platform_info.h>
#include "asdops/params/params.h"
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/tiling_data.h"

namespace AsdOps {
class PpMatmulWithBiasKernel : public KernelBase {
public:
    explicit PpMatmulWithBiasKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "Platform not support.",
                     return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "Check inTensor count failed.", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "Check outTensor count failed.", return false);

        const auto &descA = launchParam.GetInTensor(0).desc;
        const auto &descB = launchParam.GetInTensor(1).desc;
        const auto &descBias = launchParam.GetInTensor(2).desc;
        const auto &descC = launchParam.GetOutTensor(0).desc;

        MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
        MKI_CHECK(descA.dtype == TENSOR_DTYPE_BF16 || descA.dtype == TENSOR_DTYPE_FLOAT16,
                     "Tensor dtype is invalid.", return false);
        MKI_CHECK(descB.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
        MKI_CHECK(descB.dtype == TENSOR_DTYPE_BF16 || descB.dtype == TENSOR_DTYPE_FLOAT16,
                     "Tensor dtype is invalid.", return false);
        MKI_CHECK(descBias.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
        MKI_CHECK(descBias.dtype == TENSOR_DTYPE_FLOAT, "Tensor dtype is invalid.", return false);

        MKI_CHECK(descA.dtype == descB.dtype, "Input dtype must be the same.", return false);
        MKI_CHECK(descA.dtype == descC.dtype, "Output dtype must be the same as input dtype.", return false);
        MKI_CHECK(descC.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        return GetMatmulTilingSizeCommon<PpMatmulTilingData>(launchParam);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpMatmulTiling(launchParam, kernelInfo_);
    }
};

REG_KERNEL_BASE(PpMatmulWithBiasKernel);
} // namespace AsdOps