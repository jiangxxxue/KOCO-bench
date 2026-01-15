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
#include "kernels/matmul/tiling/tiling_data.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"

namespace AsdOps {
using namespace Mki;
class PpMatMulBf16NdNzNdKernel : public KernelBase {
public:
    explicit PpMatMulBf16NdNzNdKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "platformtype is invalid",
                     return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);

        const auto &descA = launchParam.GetInTensor(0).desc;
        const auto &descB = launchParam.GetInTensor(1).desc;

        MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "in tensor0 format invalid", return false);
        MKI_CHECK(descA.dtype == TENSOR_DTYPE_BF16, "in tensor0 dtype invalid", return false);

        MKI_CHECK(descB.format == TENSOR_FORMAT_FRACTAL_NZ, "in tensor1 format invalid", return false);
        MKI_CHECK(descB.dtype == TENSOR_DTYPE_BF16, "in tensor1 dtype invalid", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        constexpr uint32_t CONST_256 = 256;
        return (sizeof(PpMatmulTilingData) + CONST_256 - 1) / CONST_256 * CONST_256;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpMatmulTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(PpMatMulBf16NdNzNdKernel);
} // namespace AsdOps