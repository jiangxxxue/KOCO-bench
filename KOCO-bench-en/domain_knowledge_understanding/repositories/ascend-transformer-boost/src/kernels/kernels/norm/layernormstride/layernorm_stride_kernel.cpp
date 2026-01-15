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
#include "asdops/params/params.h"
#include "kernels/norm/layernormstride/tiling/layer_norm_stride_tiling.h"
#include "tiling/tiling_data.h"

namespace {
static const float THRESHOLD = 2e-38;
static const uint32_t NUM_TWO = 2;
static const uint32_t BYTES_ALIGN = 32;
static const uint32_t ELEMENTS_ALIGN = 16;
} // namespace

namespace AsdOps {
using namespace Mki;
class LayerNormStrideKernel : public KernelBase {
public:
    explicit LayerNormStrideKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        MKI_CHECK(!launchParam.GetInTensor(0).desc.IsContiguous(), "inTensor0 should be discontinuous", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Norm),
            "norm: param type invalid", return false);
        AsdOps::OpParam::Norm param = AnyCast<OpParam::Norm>(launchParam.GetParam());
        AsdOps::OpParam::Norm::NormType type = param.normType;
        MKI_LOG(DEBUG) << "norm type: " << type;
        MKI_CHECK(type == OpParam::Norm::LAYER_NORM, "layernorm: param type invalid", return false);
        MKI_CHECK(param.epsilon >= THRESHOLD, "epsilon invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        auto inTensor1 = launchParam.GetInTensor(1);
        auto inTensor2 = launchParam.GetInTensor(2);
        auto outTensor0 = launchParam.GetOutTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "inTensor0 format invalid", return false);
        MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "inTensor1 format invalid", return false);
        MKI_CHECK(inTensor2.desc.format == TENSOR_FORMAT_ND, "inTensor2 format invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "outTensor0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dims == inTensor0.desc.dims,
                  "outTensor0 shape not same as inTensor0", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16,
                  "input0 dtype invalid", return false);
        MKI_CHECK(inTensor1.desc.dtype == inTensor0.desc.dtype,
                  "inTensor1 dtype unsupported", return false);
        MKI_CHECK(inTensor2.desc.dtype == inTensor0.desc.dtype,
                  "inTensor2 dtype unsupported", return false);
        MKI_CHECK(launchParam.GetOutTensor(0).desc.dtype == inTensor0.desc.dtype,
                  "outTensor0 dtype unsupported", return false);
        const auto& xStrides = inTensor0.desc.strides;
        const auto& shape = inTensor0.desc.dims;
        auto inTensor0Row = shape.size();
        MKI_CHECK(!xStrides.empty(), "xStrides should not be empty", return false);
        MKI_CHECK(xStrides.size() == shape.size(), "mismatch in length of strides and shape", return false);
        MKI_CHECK(xStrides[inTensor0Row - 1] == 1,
                "the last dimension of strides should be 1", return false);
        MKI_CHECK(xStrides[inTensor0Row - NUM_TWO] >= shape[inTensor0Row - 1],
            "The penultimate element of the input0 strides \
            array must be greater than the last element of the shape array.", return false);
        MKI_CHECK((xStrides[inTensor0Row - NUM_TWO] % ELEMENTS_ALIGN) == 0,
            "padded input0 is not a multiple of " << BYTES_ALIGN << " btyes.", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(LayerNormStrideTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        KernelBufferInfoLayerNormStride kernelBufferInfo;
        kernelBufferInfo.fp16BufNumForMulRow = 2; // 2: x & cast16 x
        kernelBufferInfo.fp32BufNum = 3;        // 3: temp fp32 Buffer x, y, z
        kernelBufferInfo.fp16BufNum = 2;        // 2: beta & gamma
        return LayerNormStrideTiling(launchParam, kernelInfo_, kernelBufferInfo);
    }
};
REG_KERNEL_BASE(LayerNormStrideKernel);
} // namespace AsdOps