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
#include "tiling/post_layer_norm_tiling.h"
#include "tiling/tiling_data.h"

namespace {
static const float THRESHOLD = 2e-38;
} // namespace

namespace AsdOps {
class PostLayerNormF16QuantKernel : public KernelBase {
public:
    explicit PostLayerNormF16QuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_CHECK(type == AsdOps::OpParam::Norm::LAYER_NORM, "layernorm: param type invalid", return false);
        MKI_CHECK(param.epsilon >= THRESHOLD, "epsilon invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 6, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 2, "output num invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto inTensor2 = launchParam.GetInTensor(2);
        auto inTensor3 = launchParam.GetInTensor(3);
        auto inTensor4 = launchParam.GetInTensor(4);
        auto inTensor5 = launchParam.GetInTensor(5);
        auto outTensor0 = launchParam.GetOutTensor(0);
        auto outTensor1 = launchParam.GetOutTensor(1);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "input0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "input0 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "input1 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "input1 format invalid", return false);
        MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_FLOAT16, "input2 dtype invalid", return false);
        MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "input2 format invalid", return false);
        MKI_CHECK(inTensor3.desc.format == TENSOR_FORMAT_ND, "input3 format invalid", return false);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_FLOAT16, "input3 dtype invalid", return false);
        MKI_CHECK(inTensor4.desc.dtype == TENSOR_DTYPE_FLOAT16, "input4 dtype invalid", return false);
        MKI_CHECK(inTensor5.desc.dtype == TENSOR_DTYPE_INT8, "input5 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT8, "output0 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "output0 format invalid", return false);
        MKI_CHECK(outTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "output1 dtype invalid", return false);
        MKI_CHECK(outTensor1.desc.format == TENSOR_FORMAT_ND, "output1 format invalid", return false);
        MKI_CHECK(outTensor1.desc.dims == inTensor0.desc.dims, "outtensor1 shape not same as intensor0", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PostLayerNormTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelBufferInfo kernelBufferInfo;
        kernelBufferInfo.fp16BufNumForMulRow = 3; // 3: in, resIn, output
        kernelBufferInfo.i8BufNumForMulRow = 1; // 1: output
        kernelBufferInfo.fp32BufNum = 2; // 2: temp fp32 Buffer x, y
        kernelBufferInfo.fp16BufNum = 2; // 2: beta & gamma
        return PostLayerNormTiling(launchParam, kernelInfo_, kernelBufferInfo);
    }
};
REG_KERNEL_BASE(PostLayerNormF16QuantKernel);
} // namespace AsdOps