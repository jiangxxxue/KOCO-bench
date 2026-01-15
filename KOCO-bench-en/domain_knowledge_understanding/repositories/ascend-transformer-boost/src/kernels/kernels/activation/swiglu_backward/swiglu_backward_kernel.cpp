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
#include "kernels/activation/swiglu_backward/tiling/swiglu_backward_tiling.h"
#include "kernels/activation/swiglu_backward/tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;

namespace AsdOps {
using namespace Mki;
class SwiGluBackwardKernel : public KernelBase {
public:
    explicit SwiGluBackwardKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_LOG(DEBUG) << "SwiGluBackwardCanSupport Start";
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Activation),
                     "param type is invalid", return false);

        TensorDType xdtype = launchParam.GetInTensor(1).desc.dtype;
        MKI_CHECK(xdtype == TENSOR_DTYPE_FLOAT || xdtype == TENSOR_DTYPE_FLOAT16 ||
                     xdtype == TENSOR_DTYPE_BF16,
                     "Input dtype invalid, should be float or float16 or bf16", return false);
        MKI_LOG(DEBUG) << "SwiGluBackwardCanSupport Passed";
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(SwiGluBackwardTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return SwiGluBackwardTiling(launchParam, kernelInfo_);
    }
};

REG_KERNEL_BASE(SwiGluBackwardKernel);
} // namespace AsdOps