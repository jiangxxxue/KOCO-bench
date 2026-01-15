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
#include "kernels/matmul/tiling/pp_matmul_mix_tiling.h"
#include "kernels/matmul/tiling/pp_matmul_info.h"
#include "kernels/matmul/common/common.h"

namespace AsdOps {

class PpMatMulF16MixKernel : public KernelBase {
public:
    explicit PpMatMulF16MixKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return CheckAsdOpsND(launchParam, 3); // 输入参数数量为3
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PpMixTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpMatmulMixTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(PpMatMulF16MixKernel);
} // namespace AsdOps
