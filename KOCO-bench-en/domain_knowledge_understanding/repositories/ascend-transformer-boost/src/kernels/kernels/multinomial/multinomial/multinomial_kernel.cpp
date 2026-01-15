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
#include "asdops/params/multinomial.h"
#include "tiling/multinomial_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 1;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AsdOps {
using namespace Mki;
class MultinomialKernel : public KernelBase {
public:
    explicit MultinomialKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "check outTensor count failed", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Multinomial),
            "check param type failed!", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dims.size() == 2, "check inTensor dims failed", return false);
        uint32_t inTensor0Row = launchParam.GetInTensor(0).desc.dims.size();
        uint32_t inTensor0Col = launchParam.GetInTensor(0).desc.dims[inTensor0Row - 1];
        auto para = AnyCast<OpParam::Multinomial>(launchParam.GetParam());
        MKI_CHECK(para.numSamples < MAX_RAND_NUM, "numSamples should be smaller than " << MAX_RAND_NUM, return false);
        MKI_CHECK(para.numSamples < inTensor0Col, "numSamples should be smaller than lastdim of input0", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(MultinomialTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return MultinomialTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(MultinomialKernel);
} // namespace AsdOps