/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "tiling/swi_glu_quant_tiling.h"
#include "tiling/tiling_data.h"
#include "mixkernels/utils/common.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 1;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;

namespace AtbOps {
using namespace Mki;
class SwiGluQuantKernel : public KernelBase {
public:
    explicit SwiGluQuantKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_LOG(INFO) << " CanSupport START ";
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
        auto inTensor0 = launchParam.GetInTensor(DIM_0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "in tensor 0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16,
                  "in tensor 0 dtype invalid", return false);
        auto outTensor0 = launchParam.GetOutTensor(DIM_0);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "out tensor 0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_INT8, "out tensor 0 dtype invalid", return false);
        auto outTensor1 = launchParam.GetOutTensor(DIM_1);
        MKI_CHECK(outTensor1.desc.format == TENSOR_FORMAT_ND, "out tensor 0 format invalid", return false);
        MKI_CHECK(outTensor1.desc.dtype == TENSOR_DTYPE_FLOAT, "out tensor 0 dtype invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(SwiGluQuantTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        MKI_LOG(INFO) << " InitImpl START ";
        return SwiGluQuantTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(SwiGluQuantKernel);
}