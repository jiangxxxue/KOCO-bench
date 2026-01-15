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
#include "asdops/params/params.h"
#include "dynamic_quant_tiling/dynamic_quant_tiling.h"
#include "dynamic_quant_tiling/tiling_data.h"

namespace AsdOps {
class DynamicQuantKernel : public KernelBase {
public:
    explicit DynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        const int64_t outTensorIdxTwo = 2;
        const int64_t outTensorNumThree = 3;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Elewise),
            "elewise: param type invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 1, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == outTensorNumThree, "output num invalid", return false);
        auto opParam = AnyCast<OpParam::Elewise>(launchParam.GetParam());
        OpParam::Elewise::ElewiseType type = opParam.elewiseType;
        MKI_CHECK(type == OpParam::Elewise::ELEWISE_DYNAMIC_QUANT, "dynamic quant: param type invalid",
            return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.format == TENSOR_FORMAT_ND,
            "input format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.format == TENSOR_FORMAT_ND,
            "output1 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(1).desc.format == TENSOR_FORMAT_ND,
            "output2 format invalid", return false);
        MKI_CHECK(launchParam.GetOutTensor(outTensorIdxTwo).desc.format == TENSOR_FORMAT_ND,
            "output3 format invalid", return false);
        MKI_CHECK(launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_FLOAT16 ||
            launchParam.GetInTensor(0).desc.dtype == TENSOR_DTYPE_BF16,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == TENSOR_DTYPE_INT8,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(1).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(outTensorIdxTwo).desc.dtype == TENSOR_DTYPE_FLOAT,
            "tensor dtype unsupported", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return DynamicQuantTiling(launchParam, kernelInfo_);
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(DynamicQuantTilingData);
    }
};
REG_KERNEL_BASE(DynamicQuantKernel);
} // namespace AsdOps