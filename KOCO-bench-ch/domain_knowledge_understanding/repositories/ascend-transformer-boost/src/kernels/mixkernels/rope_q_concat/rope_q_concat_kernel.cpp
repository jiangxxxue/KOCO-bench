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
#include <mki/utils/log/log.h>
#include "atbops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/rope_q_concat_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 4;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
namespace AtbOps {
using namespace Mki;
class RopeQConcatKernel : public KernelBase {
public:
    explicit RopeQConcatKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "inTensor count invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM, "outTensor count invalid", return false);
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RopeQConcat), "opParam type invalid", return false);
        for (size_t i = 0; i < TENSOR_INPUT_NUM; i++) {
            auto inTensor = launchParam.GetInTensor(i);
            MKI_CHECK(inTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor.desc.dtype == TENSOR_DTYPE_BF16,
                      "inTensor dtype invalid", return false);
        }
        auto outTensor = launchParam.GetOutTensor(DIM_0);
        MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16,
                  "outTensor dtype invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::RopeQConcat), "OpParam is invalid",
            return 0);
        return sizeof(RopeQConcatTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return RopeQConcatTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(RopeQConcatKernel);
} // namespace AtbOps