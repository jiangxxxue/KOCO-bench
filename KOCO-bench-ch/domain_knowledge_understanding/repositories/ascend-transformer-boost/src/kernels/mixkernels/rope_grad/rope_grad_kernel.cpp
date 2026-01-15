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
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/rope_grad_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 2;
static constexpr uint32_t BTACH_LIMIT = 100000;
namespace AtbOps {
using namespace Mki;
class RopeGradKernel : public KernelBase {
public:
    explicit RopeGradKernel(const std::string &kernelName, const BinHandle *handle) : KernelBase(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "outTensor count invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RopeGrad), "opParam type invalid",
            return false);
        for (size_t i = 0; i < TENSOR_INPUT_NUM; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            MKI_CHECK(inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "inTensor dtype invalid", return false);
            MKI_CHECK(inTensor.desc.format == TENSOR_FORMAT_ND, "inTensor format invalid", return false);
            MKI_CHECK(inTensor.desc.dims.size() == 2, "inTensor dims invalid", return false);
        }
        for (size_t i = 0; i < TENSOR_OUTPUT_NUM; i++) {
            auto outTensor = launchParam.GetOutTensor(i);
            MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16, "outTensor dtype invalid", return false);
        }
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RopeGrad), "OpParam is invalid",
            return 0);
        uint32_t batch = AnyCast<OpParam::RopeGrad>(launchParam.GetParam()).qSeqLen.size();
        MKI_CHECK(batch > 0 && batch < BTACH_LIMIT, "OpParam is invalid", return 0);
        return sizeof(RopeGradTilingData) + sizeof(RopeGradSampleTilingData) * batch;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RopeGradTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(RopeGradKernel);
} // namespace AtbOps