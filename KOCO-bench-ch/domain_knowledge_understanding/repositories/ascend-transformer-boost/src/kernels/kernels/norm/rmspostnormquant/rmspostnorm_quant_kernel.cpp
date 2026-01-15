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
#include "tiling/rms_post_norm_quant_tiling.h"
#include "kernels/norm/common/common_tiling_data.h"
 
namespace AsdOps {
class RmsPostNormQuantKernel : public KernelBase {
public:
    explicit RmsPostNormQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }
 
    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm), "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::RMS_NORM, "rmsnorm: param type invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto inTensor2 = launchParam.GetInTensor(2);
        auto inTensor3 = launchParam.GetInTensor(3);
        auto inTensor4 = launchParam.GetInTensor(4);
        auto outTensor0 = launchParam.GetOutTensor(0);
        auto outTensor1 = launchParam.GetOutTensor(1);
        TensorFormat format0 = inTensor0.desc.format;
        TensorDType dtype0 = inTensor0.desc.dtype;
        MKI_CHECK(format0 == TENSOR_FORMAT_ND, "input0 format invalid", return false);
        MKI_CHECK(dtype0 == TENSOR_DTYPE_FLOAT16 || dtype0 == TENSOR_DTYPE_BF16, "input0 dtype invalid",
                  return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "input1 format invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == dtype0, "input1 dtype invalid", return false);
        MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "input2 format invalid", return false);
        MKI_CHECK(inTensor2.desc.dtype == dtype0, "input2 dtype invalid", return false);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "input3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == dtype0, "input3 dtype invalid", return false);
        MKI_CHECK(inTensor4.desc.dtype == TENSOR_DTYPE_INT8, "input4 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "output0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT8, "output0 dtype invalid", return false);
        MKI_CHECK(outTensor1.desc.format == TENSOR_FORMAT_ND, "output1 format invalid", return false);
        MKI_CHECK(outTensor1.desc.dtype == dtype0, "output1 dtype invalid", return false);
        return true;
    }
 
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(RmsNormQuantCommonTilingData);
    }
 
    Status InitImpl(const LaunchParam &launchParam) override {return RmsPostNormQuantTiling(launchParam, kernelInfo_);}
};
REG_KERNEL_BASE(RmsPostNormQuantKernel);
} // namespace AsdOps