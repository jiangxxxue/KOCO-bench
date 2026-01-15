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
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tiling/norm_dynamic_quant_tiling_data.h"
#include "tiling/norm_dynamic_quant_tiling.h"

namespace {
static const float THRESHOLD = 2e-38;
} // namespace

namespace AsdOps {
using namespace Mki;
class NormDynamicQuantKernel : public KernelBase {
public:
    explicit NormDynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        MKI_CHECK(param.epsilon >= THRESHOLD, "epsilon invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 3, "output num invalid", return false);

        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto inTensor2 = launchParam.GetInTensor(2);
        auto outTensor0 = launchParam.GetOutTensor(0);
        auto outTensor1 = launchParam.GetOutTensor(1);
        auto outTensor2 = launchParam.GetOutTensor(2);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "input0 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "input1 dtype invalid", return false);
        MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_FLOAT16, "input2 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT8, "output0 dtype invalid", return false);
        MKI_CHECK(outTensor1.desc.dtype == TENSOR_DTYPE_FLOAT, "output1 dtype invalid", return false);
        MKI_CHECK(outTensor2.desc.dtype == TENSOR_DTYPE_FLOAT, "output2 dtype invalid", return false);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input0 format invalid", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "input1 format invalid", return false);
        MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "input2 format invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "output0 format invalid", return false);
        MKI_CHECK(outTensor1.desc.format == TENSOR_FORMAT_ND, "output1 format invalid", return false);
        MKI_CHECK(outTensor2.desc.format == TENSOR_FORMAT_ND, "output2 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dims == inTensor0.desc.dims, "outtensor0 shape not same as intensor0", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(NormDynamicQuantTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return NormDynamicQuantTiling(launchParam, kernelInfo_);
    }
};

// RmsNormDynamicQuant
class RmsNormDynamicQuantKernel : public NormDynamicQuantKernel {
public:
    explicit RmsNormDynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : NormDynamicQuantKernel(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        MKI_CHECK(NormDynamicQuantKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(param.normType == AsdOps::OpParam::Norm::RMS_NORM,
                  "rmsnorm: param type invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(RmsNormDynamicQuantKernel);

// LayerNormDynamicQuant
class LayerNormDynamicQuantKernel : public NormDynamicQuantKernel {
public:
    explicit LayerNormDynamicQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : NormDynamicQuantKernel(kernelName, handle)
    {
    }
    
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        MKI_CHECK(NormDynamicQuantKernel::CanSupport(launchParam), "failed to check support", return false);
        MKI_CHECK(param.normType == AsdOps::OpParam::Norm::LAYER_NORM,
                  "layernorm: param type invalid", return false);
        return true;
    }
};
REG_KERNEL_BASE(LayerNormDynamicQuantKernel);

} // namespace AsdOps