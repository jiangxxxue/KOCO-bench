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
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/tiling_data.h"

namespace AsdOps {
using namespace Mki;
class PpMatmulEinSumKernel : public KernelBase {
public:
    explicit PpMatmulEinSumKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto attr = launchParam.GetParam();
        MKI_CHECK(attr.Type() == typeid(OpParam::MatMul), "Invalid launch param type.", return false);
        auto mmType = AnyCast<OpParam::MatMul>(attr).matmulType;
        MKI_CHECK(mmType == OpParam::MatMul::MatMulType::MATMUL_EIN_SUM, "Invalid matmul type.", return false);
        const auto &inTensor1 = launchParam.GetInTensor(1);
        if (inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ) {
            return CheckAsdOpsWeightNZ(launchParam, 2); // 输入参数数量为2
        } else {
            return CheckAsdOpsND(launchParam, 2); // 输入参数数量为2
        }
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        constexpr uint32_t CONST_256 = 256;
        return (sizeof(PpMatmulTilingData) + CONST_256 - 1) / CONST_256 * CONST_256;
    }

    Status InitImpl(const LaunchParam &launchParam) override { return PpMatmulTiling(launchParam, kernelInfo_); }
};
REG_KERNEL_BASE(PpMatmulEinSumKernel);
} // namespace AsdOps