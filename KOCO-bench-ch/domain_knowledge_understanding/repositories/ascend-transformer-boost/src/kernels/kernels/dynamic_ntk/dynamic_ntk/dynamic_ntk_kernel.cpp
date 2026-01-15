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
#include "asdops/params/dynamic_ntk.h"
#include "tiling/dynamic_ntk_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 3;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
namespace AsdOps {
class DynamicNTKKernel : public KernelBase {
public:
    explicit DynamicNTKKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "check outTensor count failed", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::DynamicNTK),
            "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 1, "check inTensor dims failed", return false);
        auto para = AnyCast<OpParam::DynamicNTK>(launchParam.GetParam());
        MKI_CHECK(para.outType == 0 || para.outType == 1 || para.outType == 2, "outType invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(DynamicNTKTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return DynamicNTKTilingFunc(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(DynamicNTKKernel);
} // namespace AsdOps