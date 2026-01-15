/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "kernels/moe_gate_corr/tiling/moe_gate_corr_tiling.h"
#include "kernels/moe_gate_corr/tiling/tiling_data.h"

namespace AsdOps {
using namespace Mki;

class MoeGateCorrKernel : public KernelBase {
public:
    explicit MoeGateCorrKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MoeGateCorr),
            "check param type failed!", return false);

        auto opParam = AnyCast<OpParam::MoeGateCorr>(launchParam.GetParam());
        MKI_CHECK((!opParam.transposeA) && (opParam.transposeB), "transpose invalid", return false);

        const auto &inTensor0 = launchParam.GetInTensor(0);
        const auto &inTensor1 = launchParam.GetInTensor(1);
        const auto &outTensor = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "inTensor0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT, "inTensor0 dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.dims.size() == 2, "inTensor0 dims invalid", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "inTensor1 format invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT, "inTensor1 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == 2, "inTensor1 dims invalid", return false);
        MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "outTensor format invalid", return false);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT, "outTensor dtype invalid", return false);
        MKI_CHECK(outTensor.desc.dims.size() == 2, "outTensor dims invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(MoeGateCorrTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return MoeGateCorrTiling(launchParam, kernelInfo_);
    }
};

class MoeGateCorrM16N16K2048Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM16N32K1024Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM32N32K1024Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM32N48K640Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM48N48K640Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM48N64K512Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

class MoeGateCorrM64N64K512Kernel : public MoeGateCorrKernel {
    using MoeGateCorrKernel::MoeGateCorrKernel;
};

REG_KERNEL_BASE(MoeGateCorrM16N16K2048Kernel);
REG_KERNEL_BASE(MoeGateCorrM16N32K1024Kernel);
REG_KERNEL_BASE(MoeGateCorrM32N32K1024Kernel);
REG_KERNEL_BASE(MoeGateCorrM32N48K640Kernel);
REG_KERNEL_BASE(MoeGateCorrM48N48K640Kernel);
REG_KERNEL_BASE(MoeGateCorrM48N64K512Kernel);
REG_KERNEL_BASE(MoeGateCorrM64N64K512Kernel);
} // namespace AsdOps
