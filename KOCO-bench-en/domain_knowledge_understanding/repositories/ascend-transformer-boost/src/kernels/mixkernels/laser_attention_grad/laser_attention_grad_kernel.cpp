/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <mki_loader/op_register.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/const/op_const.h>
#include "atbops/params/params.h"
#include "tiling/laser_attention_grad_tiling.h"
#include "tiling/tiling_data.h"

static constexpr uint32_t TENSOR_INPUT_NUM = 10;
static constexpr uint32_t TENSOR_OUTPUT_NUM = 3;

namespace AtbOps {
using namespace Mki;
class LaserAttentionGradKernel : public KernelBase {
public:
    explicit LaserAttentionGradKernel(const std::string &kernelName, const BinHandle *handle)
        : KernelBase(kernelName, handle) {}

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(launchParam.GetInTensorCount() == TENSOR_INPUT_NUM,
            "in tensor num is invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == TENSOR_OUTPUT_NUM,
            "out tensor num is invalid", return false);
        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        return sizeof(LaserAttentionGradTilingData);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        Status ret = LaserAttentionGradTiling(launchParam, kernelInfo_);
        MKI_CHECK_NO_LOG(ret.Ok(), return ret);
        kernelInfo_.SetHwsyncIdx(0);
        return Status::OkStatus();
    }
};

REG_KERNEL_BASE(LaserAttentionGradKernel);
} // namespace AtbOps
