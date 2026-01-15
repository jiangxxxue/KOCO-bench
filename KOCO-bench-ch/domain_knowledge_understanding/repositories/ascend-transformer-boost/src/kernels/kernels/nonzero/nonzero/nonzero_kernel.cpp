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
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "tiling/nonzero_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 1;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
namespace AsdOps {
class NonzeroKernel : public KernelBase {
public:
    explicit NonzeroKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        auto inTensor0 = launchParam.GetInTensor(0);
        auto outTensor0 = launchParam.GetOutTensor(0);
        auto outTensor1 = launchParam.GetOutTensor(1);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "inTensor0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "outTensor0 format invalid", return false);
        MKI_CHECK(outTensor1.desc.format == TENSOR_FORMAT_ND, "outTensor1 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_INT64, "inTensor0 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT64, "outTensor0 dtype invalid", return false);
        MKI_CHECK(outTensor1.desc.dtype == TENSOR_DTYPE_INT64, "outTensor1 dtype invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "check outTensor count failed", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(NonzeroTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return NonzeroTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(NonzeroKernel);
} // namespace AsdOps