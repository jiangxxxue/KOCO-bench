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
#include "tiling/cohere_layer_norm_tiling.h"
#include "tiling/tiling_data.h"
#include "kernels/utils/common.h"

namespace AsdOps {
class CohereLayernormKernel : public KernelBase {
public:
    explicit CohereLayernormKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        // According to the operator design document, the operator has two input tensors and one output tensor.
        //
        // Dimensions:
        // - input0: [batch_size, sequence_length, num_heads, head_size]
        // - input1: [num_heads, head_size]
        // - output0: [batch_size, sequence_length, num_heads, head_size]
        //
        // Note: The head_size should be a multiple of 32.
        //
        // Supported data types: FLOAT16 and BFLOAT16
        //
        // The input and output tensors are verified based on the following aspects:
        // 1. The number of input and output tensors.
        // 2. The dimensions of input tensors.
        // 3. The data type of input tensors.
        // 4. The size of head_size.

        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
                     "transpose: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::LAYER_NORM, "layernorm: param type invalid", return false);

        MKI_CHECK(launchParam.GetInTensorCount() == 2, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);

        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        
        MKI_CHECK((inTensor0.desc.dims.size() == 4) || (inTensor0.desc.dims.size() == 3),
                  "inTensor0's dim size should be equal to 3 or 4.", return false);
        MKI_CHECK(inTensor1.desc.dims.size() == 2, "inTensor0's dim size should be equal to 2.", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16,
                     "input0 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == inTensor0.desc.dtype, "input1 dtype invalid", return false);

        uint32_t inTensor0Col = launchParam.GetInTensor(0).desc.dims[launchParam.GetInTensor(0).desc.dims.size() - 1];
        MKI_CHECK((inTensor0Col % 32) == 0, "input0 is not a multiple of 32", return false);
        
        uint32_t inTensor1Col = launchParam.GetInTensor(1).desc.dims[launchParam.GetInTensor(1).desc.dims.size() - 1];
        MKI_CHECK((inTensor1Col % 32) == 0, "input1 is not a multiple of 32", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(CohereLayerNormTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelBufferInfoCohereLayerNorm kernelBufferInfo;
        kernelBufferInfo.fp32BufNum = 3; // 3: fp32 buffer num used for kernel calculation
        kernelBufferInfo.fp16BufNum = 1; // 1: gamma
        kernelBufferInfo.fp16BufNumForMulRow = 2; // 2: input, output
        return CohereLayerNormTiling(launchParam, kernelInfo_, kernelBufferInfo);
    }
};
REG_KERNEL_BASE(CohereLayernormKernel);
} // namespace AsdOps