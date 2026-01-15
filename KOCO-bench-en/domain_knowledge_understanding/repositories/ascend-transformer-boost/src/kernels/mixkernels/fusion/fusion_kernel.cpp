/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/base/bishengir_kernel_base.h>
#include <mki/utils/const/op_const.h>
#include <mki/utils/log/log.h>
#include <mki_loader/op_register.h>
#include "mixkernels/utils/common.h"
#include "tiling/fusion_tiling.h"
#include "tiling/tiling_data.h"
#include "atbops/params/params.h"

static constexpr uint32_t TENSOR_INPUT_NUM_MATMUL_ADD = 3;
static constexpr uint32_t TENSOR_INPUT_NUM_MATMUL_ACTIVATE = 2;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AtbOps {
using namespace Mki;
class FusionKernel : public BishengIRKernelBase {
public:
    explicit FusionKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : BishengIRKernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        OpParam::Fusion fusionType = launchParam.GetParam<OpParam::Fusion>();
        if (fusionType.fusionType == OpParam::Fusion::MATMUL_ADD) {
            MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM_MATMUL_ADD, "in tensor num invalid",
                      return false);
            MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
            auto inTensor0 = launchParam.GetInTensor(DIM_0);
            MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 0 dtype invalid", return false);
            auto inTensor1 = launchParam.GetInTensor(DIM_1);
            MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 1 dtype invalid", return false);
            auto inTensor2 = launchParam.GetInTensor(DIM_2);
            MKI_CHECK(inTensor2.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 2 dtype invalid", return false);
        } else if (fusionType.fusionType == OpParam::Fusion::MATMUL_GELU) {
            MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM_MATMUL_ACTIVATE, "in tensor num invalid",
                      return false);
            MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
            auto inTensor0 = launchParam.GetInTensor(DIM_0);
            MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 0 dtype invalid", return false);
            auto inTensor1 = launchParam.GetInTensor(DIM_1);
            MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 1 dtype invalid", return false);
        }
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            auto inTensor = launchParam.GetInTensor(i);
            MKI_CHECK(inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor dtype invalid", return false);
            MKI_CHECK(inTensor.desc.format == TENSOR_FORMAT_ND, "in tensor " << i << " format invalid", return false);
            MKI_CHECK(inTensor.desc.dims.size() == DIM_2, "in tensor " << i << " dim num invalid", return false);
        }
        for (size_t i = 0; i < TENSOR_OUTPUT_NUM; i++) {
            auto outTensor = launchParam.GetOutTensor(i);
            MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor dtype invalid", return false);
            MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "out tensor " << i << " format invalid", return false);
            MKI_CHECK(outTensor.desc.dims.size() == DIM_2, "out tensor " << i << " dim num invalid", return false);
        }
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(FusionTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetHwsyncIdx(0);
        return FusionTiling(launchParam, kernelInfo_);
    }
};
using FusionMatmulAddKernel = FusionKernel;
using FusionMatmulGeluKernel = FusionKernel;
using FusionMatmulSigmoidKernel = FusionKernel;
using FusionMatmulSwiGluKernel = FusionKernel;
using FusionErasedKernel = FusionKernel;

REG_KERNEL_BASE(FusionMatmulAddKernel);
REG_KERNEL_BASE(FusionMatmulGeluKernel);
REG_KERNEL_BASE(FusionMatmulSigmoidKernel);
REG_KERNEL_BASE(FusionMatmulSwiGluKernel);
REG_KERNEL_BASE(FusionErasedKernel);
} // namespace AtbOps
