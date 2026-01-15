/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "atbops/params/params.h"

#include "mm_deq_swiglu_quant_mm_deq_common.h"
#include "tiling/mm_deq_swiglu_quant_mm_deq_tiling.h"
#include "tiling/mm_deq_swiglu_quant_mm_deq_tiling_data.h"

namespace AtbOps {
using namespace Mki;

class MmDeqSwigluQuantMmDeqKernel : public KernelBase {
public:
    explicit MmDeqSwigluQuantMmDeqKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == INPUT_NUM, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == OUTPUT_NUM, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq),
            "param type invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MmDeqSwigluQuantMmDeq),
            "param type invalid", return 0);
        return sizeof(MmDeqSwigluQuantMmDeqTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetHwsyncIdx(0);
        return MmDeqSwigluQuantMmDeqTiling(launchParam, kernelInfo_);
    }
};

class MmDeqSwigluQuantMmDeqN256Kernel : public MmDeqSwigluQuantMmDeqKernel {
    using MmDeqSwigluQuantMmDeqKernel::MmDeqSwigluQuantMmDeqKernel;
};

class MmDeqSwigluQuantMmDeqN128Kernel : public MmDeqSwigluQuantMmDeqKernel {
    using MmDeqSwigluQuantMmDeqKernel::MmDeqSwigluQuantMmDeqKernel;
};

REG_KERNEL_BASE(MmDeqSwigluQuantMmDeqN256Kernel);
REG_KERNEL_BASE(MmDeqSwigluQuantMmDeqN128Kernel);
}  // namespace AtbOps