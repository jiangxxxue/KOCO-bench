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
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "tiling/pad_tiling.h"
#include "tiling/tiling_data.h"
#include "mixkernels/utils/common.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;

namespace AtbOps {
using namespace Mki;
class PadKernel : public KernelBase {
public:
    explicit PadKernel(const std::string &kernelName, const BinHandle *handle) noexcept : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "out tensor num invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::Pad), "param type invalid",
                     return false);
        auto inTensor0 = launchParam.GetInTensor(0);
        MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "in tensor 0 format invalid", return false);
        MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "in tensor 0 dtype invalid", return false);
        for (size_t i = 1; i < TENSOR_INPUT_NUM - 1; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            MKI_CHECK(inTensor.desc.format == TENSOR_FORMAT_ND, "in tensor " << i << " format invalid",
                         return false);
            MKI_CHECK(inTensor.desc.dtype == TENSOR_DTYPE_INT32, "in tensor " << i << " dtype invalid",
                         return false);
            MKI_CHECK(inTensor.desc.dims.size() == DIM_2, "in tensor " << i << " dims num invalid", return false);
        }
        auto inTensor3 = launchParam.GetInTensor(3);
        MKI_CHECK(inTensor3.desc.dtype == TENSOR_DTYPE_INT64, "in tensor 3 dtype invalid", return false);
        auto outTensor0 = launchParam.GetOutTensor(0);
        MKI_CHECK(outTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16, "out tensor 0 dtype invalid", return false);
        MKI_CHECK(outTensor0.desc.format == TENSOR_FORMAT_ND, "out tensor 0 format invalid", return false);
        MKI_CHECK(outTensor0.desc.dims.size() == DIM_2, "out tensor 0 dims num invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(PadTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PadTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(PadKernel);
} // namespace AtbOps