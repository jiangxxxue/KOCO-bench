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
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "mixkernels/utils/common.h"
#include "tiling/logprobs_sample_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 3;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 1;
static constexpr uint32_t MAX_BATCH_SIZE = 512;

namespace AsdOps {
using namespace Mki;
class LogprobsSampleKernel : public KernelBase {
public:
    explicit LogprobsSampleKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
        launchBufferSize_ = sizeof(LogprobsSampleTilingData);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
                  "check outTensor count failed", return false);
        auto sortedProbsTensorDesc = launchParam.GetInTensor(INPUT_SORTED_PROBS_INDEX).desc;
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::LogprobsSample),
                  "check param type failed!", return false);
        MKI_CHECK(sortedProbsTensorDesc.dims.size() == 2, "check inTensor dims failed", return false);
        MKI_CHECK(sortedProbsTensorDesc.dims[0] <= MAX_BATCH_SIZE,
                  "batch size should be less than or equal to 512", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return launchBufferSize_ + sizeof(OpParam::LogprobsSample::logprobsSize);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        kernelInfo_.SetConstTensorOffset(launchBufferSize_);
        auto &param = AnyCast<OpParam::LogprobsSample>(launchParam.GetParam());
        MKI_LOG(DEBUG)<< "logprobs size is :"<< param.logprobsSize;
        auto ret = kernelInfo_.AddConstTensorData<int32_t>(INPUT_LOGPROBS_SIZE_INDEX, { param.logprobsSize });
        MKI_CHECK_NO_LOG(ret, return Status::FailStatus(1));
        return LogprobsSampleTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(LogprobsSampleKernel);
} // namespace AsdOps