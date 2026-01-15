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
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/checktensor/check_tensor.h>
#include "asdops/params/params.h"
#include "tiling/adalayer_norm_tiling.h"
#include "tiling/tiling_data.h"

constexpr uint32_t SCALE_DIM_THREE = 3;

namespace AsdOps {
using namespace Mki;
class AdaLayerNormKernel : public KernelBase {
public:
    explicit AdaLayerNormKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
                     "transpose: param type invalid", return false);

        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);

        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto inTensor2 = launchParam.GetInTensor(2);
        auto outTensor0 = launchParam.GetOutTensor(0);
        if (!(CheckEmptyTensor(inTensor0) || CheckEmptyTensor(inTensor1) || CheckEmptyTensor(inTensor2))) {
            MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input0 format invalid", return false);
            MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16,
                        "input0 dtype invalid", return false);
            MKI_CHECK(inTensor1.desc.format == inTensor0.desc.format, "input1 format invalid", return false);
            MKI_CHECK(inTensor1.desc.dtype == inTensor0.desc.dtype, "input1 dtype invalid", return false);
            MKI_CHECK(inTensor2.desc.format == inTensor0.desc.format, "input1 format invalid", return false);
            MKI_CHECK(inTensor2.desc.dtype == inTensor0.desc.dtype, "input1 dtype invalid", return false);
        }
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "output0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == inTensor0.desc.dtype, "output0 dtype invalid", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(AdaLayerNormTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelBufferInfoAdaLayerNorm kernelBufferInfo;
        kernelBufferInfo.fp16BufNumForMulRow = 2 * 2; // 2: x & cast16 x（double buffer）
        kernelBufferInfo.fp32BufNum = 2;        // 2: temp fp32 Buffer x, y
        kernelBufferInfo.fp16BufNum = 2 * 2;        // 2: beta & gamma （double buffer）
        return AdaLayerNormTiling(launchParam, kernelInfo_, kernelBufferInfo);
    }
};
REG_KERNEL_BASE(AdaLayerNormKernel);
} // namespace AsdOps